import os
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
CHART = "hdd_chart.png"


def fetch_daily_mean_f(lat, lon):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_mean"
        "&temperature_unit=fahrenheit"
        f"&forecast_days={FORECAST_DAYS}"
        "&timezone=UTC"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()["daily"]["temperature_2m_mean"]


def hdd(temp_f):
    return max(0.0, BASE_F - temp_f)


def compute_15d_hdd():
    total = 0.0
    for lat, lon, weight in CITIES.values():
        temps = fetch_daily_mean_f(lat, lon)
        total += weight * sum(hdd(t) for t in temps)
    return total


def signal_text(delta):
    if delta > 5:
        return "🔥 Bullish (HDD 上修)"
    elif delta < -5:
        return "❄ Bearish (HDD 下修)"
    else:
        return "😐 Neutral"


def format_msg(today, today_val, delta):
    return (
        f"🧊 HDD 15D Update ({today})\n"
        f"────────────────────\n"
        f"15-Day Weighted HDD: {today_val:.2f}\n"
        f"Δ vs Previous:       {delta:+.2f}\n"
        f"Signal:              {signal_text(delta)}\n"
    )


def tg_send_message(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()


def tg_send_photo(token, chat_id, photo_path, caption=None):
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    with open(photo_path, "rb") as f:
        files = {"photo": f}
        data = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption
        r = requests.post(url, data=data, files=files, timeout=60)
    r.raise_for_status()


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

    # chart
    plt.figure()
    plt.plot(df["date"], df["hdd_15d"])
    plt.xticks(rotation=45)
    plt.title("15-Day Weighted HDD")
    plt.tight_layout()
    plt.savefig(CHART)

    # console
    print(format_msg(today, today_val, delta))
    print(f"Saved: {FILE}, {CHART}")

    # telegram (optional)
    token = os.getenv("TG_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TG_CHAT_ID", "").strip()

    if token and chat_id:
        msg = format_msg(today, today_val, delta)
        try:
            tg_send_message(token, chat_id, msg)
            # 圖片用很短 caption，避免重複一堆文字
            tg_send_photo(token, chat_id, CHART, caption=f"📈 HDD Trend ({today})")
            print("Telegram: sent message + chart")
        except Exception as e:
            # 你如果希望 Telegram 失敗就讓整個 Actions 失敗，把這行改成 raise
            print(f"Telegram send failed: {e}")
    else:
        print("Telegram skipped: TG_BOT_TOKEN / TG_CHAT_ID not set")


if __name__ == "__main__":
    run_system()
