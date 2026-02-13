import os
import requests
import pandas as pd
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_F = 65.0
FORECAST_DAYS = 30  # ✅ 一次抓 30 天，15D/30D 都能算

CITIES = {
    "New_York": (40.7128, -74.0060, 0.20),
    "Chicago":  (41.8781, -87.6298, 0.20),
    "Boston":   (42.3601, -71.0589, 0.10),
    "Atlanta":  (33.7490, -84.3880, 0.15),
    "Dallas":   (32.7767, -96.7970, 0.15),
    "Denver":   (39.7392, -104.9903, 0.10),
    "LA":       (34.0522, -118.2437, 0.10),
}

DATA_FILE = "ng_hdd_data.csv"
CHART_FILE = "hdd_chart.png"
MARKET_FILE = "market_data.csv"

TG_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")

def fetch_daily_mean_f(lat, lon):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_mean"
        "&temperature_unit=fahrenheit"
        f"&past_days={PAST_DAYS}"
        f"&forecast_days={FORECAST_DAYS}"
        "&timezone=UTC"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    daily = r.json()["daily"]
    dates = daily["time"]
    temps = daily["temperature_2m_mean"]
    return dates, temps

def hdd(temp_f):
    return max(0.0, BASE_F - temp_f)

from datetime import date

def compute_hdd_15_30():
    today_str = str(date.today())

    total_15 = 0.0
    total_30 = 0.0

    for lat, lon, weight in CITIES.values():
        dates, temps = fetch_daily_mean_f(lat, lon)

        # 找「今天」在回傳陣列的位置
        try:
            i0 = dates.index(today_str)
        except ValueError:
            # 找不到就用中間當今天（保底）
            i0 = PAST_DAYS

        hdds = [hdd(t) for t in temps]

        # 15D：今天起算 15 天（i0 ~ i0+14）
        if i0 + 15 > len(hdds):
            raise RuntimeError("Not enough days for 15D window")
        h15 = sum(hdds[i0:i0+15])

        # 30D：過去14 + 今天 + 未來15（共30天）
        start = i0 - PAST_DAYS
        end = i0 + (30 - PAST_DAYS)  # i0+16
        if start < 0 or end > len(hdds):
            raise RuntimeError("Not enough days for 30D window")
        h30 = sum(hdds[start:end])

        total_15 += weight * h15
        total_30 += weight * h30

    return total_15, total_30

def signal_from_delta(delta):
    if delta > 5:
        return "🔥 Bullish Weather Revision"
    elif delta < -5:
        return "❄️ Bearish Weather Revision"
    else:
        return "🙂 Neutral"

def load_market(today_str):
    """
    可選：若 repo 有 market_data.csv，抓今天最新一筆（或最後一筆）。
    欄位：date,ng_price,storage_bcf
    """
    if not os.path.exists(MARKET_FILE):
        return None

    try:
        m = pd.read_csv(MARKET_FILE)
        if m.empty:
            return None
        # 先找今天，沒有就拿最後一筆
        row = m[m["date"].astype(str) == today_str]
        if not row.empty:
            r = row.iloc[-1]
        else:
            r = m.iloc[-1]
        return {
            "ng_price": float(r["ng_price"]) if "ng_price" in r and pd.notna(r["ng_price"]) else None,
            "storage_bcf": float(r["storage_bcf"]) if "storage_bcf" in r and pd.notna(r["storage_bcf"]) else None,
        }
    except Exception:
        return None

def send_telegram_message(text):
    if not TG_TOKEN or not TG_CHAT_ID:
        print("Telegram env not set. Skip sending message.")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()

def send_telegram_photo(photo_path, caption=""):
    if not TG_TOKEN or not TG_CHAT_ID:
        print("Telegram env not set. Skip sending photo.")
        return
    if not os.path.exists(photo_path):
        print(f"Chart not found: {photo_path}")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto"
    with open(photo_path, "rb") as f:
        files = {"photo": f}
        data = {"chat_id": TG_CHAT_ID, "caption": caption}
        r = requests.post(url, data=data, files=files, timeout=60)
        r.raise_for_status()

def plot_chart(df):
    plt.figure()
    plt.plot(df["date"], df["hdd_15d"])
    plt.plot(df["date"], df["hdd_30d"])
    plt.xticks(rotation=45)
    plt.title("HDD Trend (15D vs 30D)")
    plt.tight_layout()
    plt.savefig(CHART_FILE)
    print(f"Chart saved as {CHART_FILE}")

def run_system():
    today_str = str(date.today())

    h15, h30 = compute_hdd_15_30()

    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        prev15 = float(df.iloc[-1]["hdd_15d"])
        prev30 = float(df.iloc[-1]["hdd_30d"])
        d15 = h15 - prev15
        d30 = h30 - prev30
    else:
        df = pd.DataFrame(columns=["date", "hdd_15d", "delta_15d", "hdd_30d", "delta_30d"])
        d15 = 0.0
        d30 = 0.0

    new_row = pd.DataFrame(
        [[today_str, h15, d15, h30, d30]],
        columns=["date", "hdd_15d", "delta_15d", "hdd_30d", "delta_30d"],
    )
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

    sig15 = signal_from_delta(d15)

    # 可選市場資料
    market = load_market(today_str)

    # ✅ Telegram 文字更好懂（固定格式）
    lines = []
    lines.append(f"📊 <b>HDD Daily Report</b> ({today_str})")
    lines.append("")
    lines.append(f"15D Weighted HDD: <b>{h15:.2f}</b>  (Δ {d15:+.2f})")
    lines.append(f"30D Weighted HDD: <b>{h30:.2f}</b>  (Δ {d30:+.2f})")
    lines.append("")
    lines.append(f"Signal (15D): <b>{sig15}</b>")

    if market:
        ngp = market.get("ng_price")
        stg = market.get("storage_bcf")
        lines.append("")
        lines.append("📌 <b>Market</b>")
        if ngp is not None:
            lines.append(f"NG Price: <b>{ngp:.3f}</b>")
        if stg is not None:
            lines.append(f"Storage: <b>{stg:.0f}</b> bcf")

    msg = "\n".join(lines)

    print(msg.replace("<b>", "").replace("</b>", ""))

    # 先畫圖，再送訊息+圖
    plot_chart(df)
    send_telegram_message(msg)
    send_telegram_photo(CHART_FILE, caption=f"HDD Trend (15D vs 30D) • {today_str}")

if __name__ == "__main__":
    run_system()
