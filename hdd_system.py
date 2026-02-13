import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta, timezone

# ===============================
# CONFIG
# ===============================

LAT = float(os.getenv("LAT", "40.7128"))
LON = float(os.getenv("LON", "-74.0060"))
BASE_F = 65.0

CSV_FILE = "ng_hdd_data.csv"
CHART_FILE = "ng_etf_chart.png"

TG_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")

WIN15 = 15

# ===============================
# WEATHER
# ===============================

def fetch_past_mean(days=35):
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = end - timedelta(days=days-1)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "temperature_2m_mean",
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    temps = data["daily"]["temperature_2m_mean"]
    return temps

def compute_hdd(temps):
    return [max(0, BASE_F - t) for t in temps]

# ===============================
# PRICE DATA
# ===============================

def fetch_ng_price():
    # Front month proxy: UNG
    data = yf.download("UNG", period="60d", interval="1d", progress=False)
    data["20MA"] = data["Close"].rolling(20).mean()
    data["TR"] = data["High"] - data["Low"]
    data["ATR10"] = data["TR"].rolling(10).mean()
    return data

# ===============================
# TELEGRAM
# ===============================

def send_msg(text):
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text})

def send_photo(path, caption):
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    with open(path, "rb") as f:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto",
            files={"photo": f},
            data={"chat_id": TG_CHAT_ID, "caption": caption},
        )

# ===============================
# MAIN
# ===============================

def run():

    today = datetime.now(timezone.utc).date().isoformat()

    # ---- Weather
    temps = fetch_past_mean()
    hdds = compute_hdd(temps)

    hdd_15 = sum(hdds[-WIN15:])
    hdd_prev = sum(hdds[-WIN15-1:-1])
    revision = hdd_15 - hdd_prev

    # ---- Price
    price_df = fetch_ng_price()
    latest = price_df.iloc[-1]

    price = latest["Close"]
    ma20 = latest["20MA"]
    atr = latest["ATR10"]

    atr_ratio = atr / price if price != 0 else 0

    # ---- Signal Logic (Aggressive)
    signal = "Neutral"
    if revision > 10 and price >= ma20:
        signal = "🔥 BOIL Bias"
    elif revision < -10 and price <= ma20:
        signal = "❄️ KOLD Bias"

    # ---- Chart
    plt.figure(figsize=(10,6))
    plt.plot(price_df["Close"], label="UNG")
    plt.plot(price_df["20MA"], label="20MA")
    plt.title("UNG Price & 20MA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_FILE)
    plt.close()

    # ---- Telegram
    msg = f"""
NG ETF Aggressive Signal ({today})

Weather Revision (15D): {revision:.2f}
Price: {price:.2f}
20MA: {ma20:.2f}
ATR10/Price: {atr_ratio:.3f}

Signal: {signal}
""".strip()

    print(msg)
    send_msg(msg)
    send_photo(CHART_FILE, "UNG Price vs 20MA")

if __name__ == "__main__":
    run()
