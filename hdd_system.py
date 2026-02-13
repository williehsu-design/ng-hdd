import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
BASE_F = float(os.getenv("BASE_F", "65.0"))

LAT = float(os.getenv("LAT", "40.7128"))
LON = float(os.getenv("LON", "-74.0060"))

CSV_FILE = os.getenv("CSV_FILE", "ng_hdd_data.csv")
CHART_FILE = os.getenv("CHART_FILE", "hdd_cdd_chart.png")

TG_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "").strip()

WIN15 = 15
WIN30 = 30

USER_AGENT = "hdd-cdd-bot/1.0"

# =========================
# HELPERS
# =========================
def retry_get(url: str, params: dict, tries: int = 4, timeout: int = 30) -> requests.Response:
    last_err = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if r.status_code in (502, 503, 504):
                time.sleep(1.5 ** i)
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(1.5 ** i)
    raise RuntimeError(f"HTTP failed after {tries} tries: {last_err}")

def hdd(temp_f: float) -> float:
    return max(0.0, BASE_F - temp_f)

def cdd(temp_f: float) -> float:
    return max(0.0, temp_f - BASE_F)

def utc_today_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()

def fetch_past_daily_mean_f(days: int = 35):
    """
    Use Open-Meteo ARCHIVE API for stable past observed daily mean temps.
    We'll fetch up to yesterday (UTC) to avoid partial-day issues.
    """
    end_date = (datetime.now(timezone.utc).date() - timedelta(days=1))
    start_date = end_date - timedelta(days=days - 1)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "temperature_2m_mean",
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
    }
    r = retry_get(url, params=params)
    data = r.json()

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    temps = daily.get("temperature_2m_mean", [])

    if not dates or not temps or len(dates) != len(temps):
        raise RuntimeError(f"Unexpected archive payload: {str(data)[:300]}")

    return dates, [float(t) for t in temps]

def send_telegram_message(text: str):
    if not TG_TOKEN or not TG_CHAT_ID:
        print("Telegram skipped: TG_BOT_TOKEN/TG_CHAT_ID not set.")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text, "disable_web_page_preview": True}
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()

def send_telegram_photo(photo_path: str, caption: str = ""):
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    if not os.path.exists(photo_path):
        print(f"Chart missing: {photo_path}")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto"
    with open(photo_path, "rb") as f:
        files = {"photo": f}
        data = {"chat_id": TG_CHAT_ID, "caption": caption}
        r = requests.post(url, data=data, files=files, timeout=60)
    r.raise_for_status()

def make_chart(df: pd.DataFrame):
    dfx = df.tail(90).copy()
    plt.figure(figsize=(10.5, 5.2))
    plt.plot(dfx["date"], dfx["hdd_15d"], label="HDD 15D")
    plt.plot(dfx["date"], dfx["cdd_15d"], label="CDD 15D")
    plt.plot(dfx["date"], dfx["ndd_15d"], label="NDD 15D (CDD-HDD)")
    plt.xticks(rotation=45, ha="right")
    plt.title("Degree Days Trend (15D)")
    plt.xlabel("Run Date (UTC)")
    plt.ylabel("Degree Days (sum)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=160)
    plt.close()

def run_system():
    run_date = utc_today_date()

    # Fetch past temps
    dates, temps = fetch_past_daily_mean_f(days=35)

    daily_hdd = [hdd(t) for t in temps]
    daily_cdd = [cdd(t) for t in temps]

    # last N days sums
    hdd_15 = sum(daily_hdd[-WIN15:])
    hdd_30 = sum(daily_hdd[-WIN30:])
    cdd_15 = sum(daily_cdd[-WIN15:])
    cdd_30 = sum(daily_cdd[-WIN30:])

    ndd_15 = cdd_15 - hdd_15
    ndd_30 = cdd_30 - hdd_30

    # Load/upgrade CSV
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame(columns=[
            "date",
            "hdd_15d","hdd_30d",
            "cdd_15d","cdd_30d",
            "ndd_15d","ndd_30d",
            "delta_hdd_15d","delta_cdd_15d","delta_ndd_15d"
        ])

    for col in [
        "date",
        "hdd_15d","hdd_30d",
        "cdd_15d","cdd_30d",
        "ndd_15d","ndd_30d",
        "delta_hdd_15d","delta_cdd_15d","delta_ndd_15d"
    ]:
        if col not in df.columns:
            df[col] = 0.0 if col != "date" else ""

    # Previous values
    if len(df) > 0:
        prev_h15 = float(df.iloc[-1].get("hdd_15d", 0.0))
        prev_c15 = float(df.iloc[-1].get("cdd_15d", 0.0))
        prev_n15 = float(df.iloc[-1].get("ndd_15d", 0.0))
    else:
        prev_h15 = prev_c15 = prev_n15 = 0.0

    delta_h15 = hdd_15 - prev_h15
    delta_c15 = cdd_15 - prev_c15
    delta_n15 = ndd_15 - prev_n15

    new_row = {
        "date": run_date,
        "hdd_15d": round(hdd_15, 3),
        "hdd_30d": round(hdd_30, 3),
        "cdd_15d": round(cdd_15, 3),
        "cdd_30d": round(cdd_30, 3),
        "ndd_15d": round(ndd_15, 3),
        "ndd_30d": round(ndd_30, 3),
        "delta_hdd_15d": round(delta_h15, 3),
        "delta_cdd_15d": round(delta_c15, 3),
        "delta_ndd_15d": round(delta_n15, 3),
    }

    if (df["date"].astype(str) == run_date).any():
        df.loc[df["date"].astype(str) == run_date, list(new_row.keys())] = list(new_row.values())
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(CSV_FILE, index=False)

    make_chart(df)

    regime = "❄️ HDD-dominant" if ndd_15 < 0 else "🔥 CDD-dominant"

    msg = (
        f"📊 HDD/CDD Daily Report ({run_date} UTC)\n"
        f"────────────────────────\n"
        f"15D HDD: {hdd_15:.2f}   (Δ {delta_h15:+.2f})\n"
        f"15D CDD: {cdd_15:.2f}   (Δ {delta_c15:+.2f})\n"
        f"15D NDD: {ndd_15:.2f}   (Δ {delta_n15:+.2f})\n"
        f"\n"
        f"30D HDD: {hdd_30:.2f}\n"
        f"30D CDD: {cdd_30:.2f}\n"
        f"30D NDD: {ndd_30:.2f}\n"
        f"\n"
        f"Regime: {regime}"
    )

    print(msg)

    send_telegram_message(msg)
    send_telegram_photo(CHART_FILE, caption=f"📈 Trend (HDD/CDD/NDD 15D) • {run_date} UTC")

if __name__ == "__main__":
    run_system()
