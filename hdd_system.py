# hdd_system.py
import os
import time
import math
from datetime import datetime, timezone
from typing import List, Tuple, Optional

import requests
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
BASE_F = float(os.getenv("BASE_F", "65"))  # HDD base temperature (F)
FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", "30"))  # we need 30 to compute 30D
CSV_PATH = os.getenv("CSV_PATH", "ng_hdd_data.csv")
CHART_PATH = os.getenv("CHART_PATH", "hdd_chart.png")

# Location (example: NYC-ish)
LAT = float(os.getenv("LAT", "40.7128"))
LON = float(os.getenv("LON", "-74.0060"))

# Telegram
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "").strip()

# Optional price inputs (if you want)
# You can later feed these from anywhere; for now read from env (Actions secrets or repo variables)
STORAGE_PRICE = os.getenv("STORAGE_PRICE", "").strip()  # e.g. "2.10"
NG_PRICE = os.getenv("NG_PRICE", "").strip()            # e.g. "2.45"


# =========================
# HELPERS
# =========================
def clamp_lat_lon(lat: float, lon: float) -> Tuple[float, float]:
    if not (-90 <= lat <= 90):
        raise ValueError(f"LAT out of range: {lat}")
    if not (-180 <= lon <= 180):
        raise ValueError(f"LON out of range: {lon}")
    return lat, lon


def safe_float(x: str) -> Optional[float]:
    try:
        if x is None:
            return None
        x = str(x).strip()
        if x == "":
            return None
        return float(x)
    except Exception:
        return None


def retry_get(url: str, params: dict, tries: int = 3, timeout: int = 20) -> requests.Response:
    last_err = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            # If API sometimes returns 502/503, retry
            if r.status_code in (502, 503, 504):
                time.sleep(1.5 * (i + 1))
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"HTTP request failed after {tries} tries: {last_err}")


def fetch_daily_mean_f(lat: float, lon: float, forecast_days: int) -> Tuple[List[str], List[float]]:
    """
    Open-Meteo forecast daily mean temperature (F) for N days.
    """
    lat, lon = clamp_lat_lon(lat, lon)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_mean",
        "temperature_unit": "fahrenheit",
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }

    r = retry_get(url, params=params, tries=3, timeout=20)
    data = r.json()

    if "daily" not in data or "time" not in data["daily"] or "temperature_2m_mean" not in data["daily"]:
        raise RuntimeError(f"Unexpected API response shape: {data}")

    dates = data["daily"]["time"]
    temps = data["daily"]["temperature_2m_mean"]

    if len(dates) < forecast_days or len(temps) < forecast_days:
        # still accept, but we need at least 30 for 30D
        pass

    return dates, temps


def compute_hdd_series(temps_f: List[float], base_f: float) -> List[float]:
    # HDD per day: max(base - mean_temp, 0)
    hdds = []
    for t in temps_f:
        h = max(base_f - float(t), 0.0)
        hdds.append(h)
    return hdds


def weighted_sum(hdds: List[float], days: int) -> float:
    """
    Weighted HDD like your earlier logic:
    - more weight to nearer days.
    Weight scheme: linear weights (days..1).
    """
    if len(hdds) < days:
        days = len(hdds)
    if days <= 0:
        return 0.0

    weights = list(range(days, 0, -1))  # e.g. 30..1
    s = 0.0
    wsum = 0.0
    for i in range(days):
        s += hdds[i] * weights[i]
        wsum += weights[i]
    return s / wsum if wsum else 0.0


def signal_from_delta(delta15: float, delta30: float) -> str:
    # Simple readable signal:
    # - if both up: bullish
    # - if both down: bearish
    # - else neutral
    up15 = delta15 > 0.5
    dn15 = delta15 < -0.5
    up30 = delta30 > 0.5
    dn30 = delta30 < -0.5

    if up15 and up30:
        return "🔥 Bullish (HDD up)"
    if dn15 and dn30:
        return "🧊 Bearish (HDD down)"
    return "😐 Neutral / Mixed"


def fmt(n: float, digits: int = 2) -> str:
    return f"{n:.{digits}f}"


def send_telegram_message(text: str) -> None:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("Telegram secrets missing (TG_BOT_TOKEN/TG_CHAT_ID). Skip sending.")
        return

    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }
    r = retry_get(url, params=payload, tries=3, timeout=20)
    _ = r.text


def send_telegram_photo(photo_path: str, caption: str = "") -> None:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("Telegram secrets missing (TG_BOT_TOKEN/TG_CHAT_ID). Skip sending photo.")
        return
    if not os.path.exists(photo_path):
        print(f"Chart not found: {photo_path}. Skip sending photo.")
        return

    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendPhoto"
    with open(photo_path, "rb") as f:
        files = {"photo": f}
        data = {"chat_id": TG_CHAT_ID, "caption": caption}
        # Use requests directly here because it's multipart
        resp = requests.post(url, data=data, files=files, timeout=30)
        resp.raise_for_status()


def plot_chart(df: pd.DataFrame, out_path: str) -> None:
    # Expect columns: date, hdd_15d, hdd_30d
    dfx = df.copy()
    dfx["date"] = pd.to_datetime(dfx["date"])
    dfx = dfx.sort_values("date")

    plt.figure(figsize=(10, 5))
    plt.plot(dfx["date"], dfx["hdd_15d"], label="HDD 15D")
    plt.plot(dfx["date"], dfx["hdd_30d"], label="HDD 30D")
    plt.title("HDD Trend (15D vs 30D)")
    plt.xlabel("Date")
    plt.ylabel("Weighted HDD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# =========================
# MAIN
# =========================
def run_system():
    # 1) fetch temps
    dates, temps = fetch_daily_mean_f(LAT, LON, FORECAST_DAYS)

    # Need at least 30 days for 30D
    if len(temps) < 30:
        raise RuntimeError(f"Not enough forecast days returned: got {len(temps)}")

    hdds = compute_hdd_series(temps, BASE_F)
    h15 = weighted_sum(hdds, 15)
    h30 = weighted_sum(hdds, 30)

    today = datetime.now(timezone.utc).date().isoformat()

    # 2) load / init csv
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.DataFrame(columns=["date", "hdd_15d", "hdd_30d", "delta_15d", "delta_30d", "storage_price", "ng_price"])

    # Make sure required cols exist (upgrade old csv)
    for c in ["date", "hdd_15d", "hdd_30d", "delta_15d", "delta_30d", "storage_price", "ng_price"]:
        if c not in df.columns:
            df[c] = 0.0 if c != "date" else ""

    # Convert numeric cols safely
    for c in ["hdd_15d", "hdd_30d", "delta_15d", "delta_30d", "storage_price", "ng_price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Previous values (yesterday row = last row, if any)
    if len(df) > 0:
        prev15 = float(df.iloc[-1].get("hdd_15d", 0.0))
        prev30 = float(df.iloc[-1].get("hdd_30d", 0.0))
    else:
        prev15, prev30 = 0.0, 0.0

    d15 = h15 - prev15
    d30 = h30 - prev30

    # Prices (optional)
    storage_p = safe_float(STORAGE_PRICE) or 0.0
    ng_p = safe_float(NG_PRICE) or 0.0

    # 3) write new row (replace if same date already exists)
    new_row = {
        "date": today,
        "hdd_15d": round(h15, 3),
        "hdd_30d": round(h30, 3),
        "delta_15d": round(d15, 3),
        "delta_30d": round(d30, 3),
        "storage_price": round(storage_p, 3),
        "ng_price": round(ng_p, 3),
    }

    # If today's row exists, overwrite
    if (df["date"] == today).any():
        df.loc[df["date"] == today, list(new_row.keys())] = list(new_row.values())
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Keep only last 120 days to keep repo small (optional)
    df["date"] = df["date"].astype(str)
    df = df.tail(120)

    # Ensure column order
    df = df.reindex(columns=["date", "hdd_15d", "hdd_30d", "delta_15d", "delta_30d", "storage_price", "ng_price"])
    df.to_csv(CSV_PATH, index=False)

    # 4) chart
    plot_chart(df, CHART_PATH)

    # 5) Telegram (clearer text)
    sig = signal_from_delta(d15, d30)

    msg_lines = [
        f"📌 HDD Update ({today})",
        f"• 15D Weighted HDD: {fmt(h15)}  (Δ {fmt(d15)})",
        f"• 30D Weighted HDD: {fmt(h30)}  (Δ {fmt(d30)})",
        f"• Signal: {sig}",
    ]

    # Optional prices
    if storage_p != 0.0 or ng_p != 0.0:
        msg_lines.append("")
        msg_lines.append("💲 Inputs (optional)")
        msg_lines.append(f"• Storage: {fmt(storage_p)}")
        msg_lines.append(f"• NG: {fmt(ng_p)}")

    text = "\n".join(msg_lines)

    send_telegram_message(text)
    send_telegram_photo(CHART_PATH, caption=f"HDD Trend (15D vs 30D) — {today}")

    print("Done.")
    print(text)


if __name__ == "__main__":
    run_system()
