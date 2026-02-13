import os
import time
import requests
import pandas as pd
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_F = 65.0
FORECAST_DAYS = 15

CITIES = {
    "New_York": (40.7128, -74.0060, 0.20),
    "Chicago":  (41.8781, -87.6298, 0.20),
    "Boston":   (42.3601, -71.0589, 0.10),
    "Atlanta":  (33.7490, -84.3880, 0.15),
    "Dallas":   (32.7767, -96.7970, 0.15),
    "Denver":   (39.7392, -104.9903, 0.10),
    "LA":       (34.0522, -118.2437, 0.10),
}

FILE = "ng_hdd_data.csv"

def fetch_daily_mean_f(lat, lon, retries=5, backoff=2):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_mean"
        "&temperature_unit=fahrenheit"
        f"&forecast_days={FORECAST_DAYS}"
        "&timezone=UTC"
    )

    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.json()["daily"]["temperature_2m_mean"]
        except Exception as e:
            last_err = e
            time.sleep(backoff ** i)  # 1,2,4,8...秒
    raise RuntimeError(f"Open-Meteo failed after retries: {last_err}")

def hdd(temp_f):
    return max(0.0, BASE_F - temp_f)

def compute_15d_hdd():
    total = 0.0
    for lat, lon, weight in CITIES.values():
        temps = fetch_daily_mean_f(lat, lon)
        total += weight * sum(hdd(t) for t in temps)
    return total

def send_telegram(text: str):
    token = os.getenv("TG_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TG_CHAT_ID", "").strip()

    if not token or not chat_id:
        print("Telegram skipped: TG_BOT_TOKEN / TG_CHAT_ID not set.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    print("Telegram sent.")

def run_system():
    today = str(date.today())
    today_val = compute_15d_hdd()

    if os.path.exists(FILE):
        df = pd.read_csv(FILE)
        prev = float(df.iloc[-1]["hdd_15d"])
        delta = today_val - prev
    else:
        df = pd.DataFrame(columns=["date", "hdd_15d", "delta"])
        delta = 0.0

    new_row = pd.DataFrame([[today, today_val, delta]], columns=["date", "hdd_15d", "delta"])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(FILE, index=False)

    if delta > 5:
        signal = "🔥 Bullish Weather Revision"
    elif delta < -5:
        signal = "❄ Bearish Weather Revision"
    else:
        signal = "😐 Neutral"

    msg = (
        f"HDD 15D Update ({today})\n"
        f"15-Day Weighted HDD: {today_val:.2f}\n"
        f"Delta vs Yesterday: {delta:.2f}\n"
        f"Signal: {signal}"
    )

    print("\n==============================")
    print(msg)
    print("==============================")

    plt.figure()
    plt.plot(df["date"], df["hdd_15d"])
    plt.xticks(rotation=45)
    plt.title("15-Day Weighted HDD")
    plt.tight_layout()
    plt.savefig("hdd_chart.png")
    print("Chart saved as hdd_chart.png")

    send_telegram(msg)

if __name__ == "__main__":
    run_system()
