#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import json
import time
import datetime as dt
from typing import List, Tuple, Optional, Dict

import requests
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
LAT = float(os.getenv("HDD_LAT", "40.7128"))     # Default: NYC
LON = float(os.getenv("HDD_LON", "-74.0060"))
BASE_F = float(os.getenv("HDD_BASE_F", "65.0"))  # HDD base temperature (F)

FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", "30"))  # we fetch 30 so we can compute 15/30
CSV_PATH = os.getenv("HDD_CSV_PATH", "ng_hdd_data.csv")
CHART_PATH = os.getenv("HDD_CHART_PATH", "hdd_chart.png")

# Optional pricing inputs (recommended to set via GitHub Secrets or repo variables)
# If not provided, we'll fallback to last CSV values.
STORAGE_PRICE = os.getenv("STORAGE_PRICE")  # e.g. "0.35"
NG_PRICE = os.getenv("NG_PRICE")            # e.g. "2.15"

# Telegram
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")

# Signal thresholds
DELTA_BULL = float(os.getenv("DELTA_BULL", "10.0"))
DELTA_BEAR = float(os.getenv("DELTA_BEAR", "-10.0"))

# Chart length (history rows)
CHART_HISTORY_ROWS = int(os.getenv("CHART_HISTORY_ROWS", "90"))

USER_AGENT = "ng-hdd-bot/1.0"


# -----------------------------
# HELPERS
# -----------------------------
def log(msg: str):
    print(msg, flush=True)


def retry_get(url: str, params: dict, tries: int = 3, timeout: int = 20, backoff: float = 1.5) -> requests.Response:
    last_err = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if r.status_code >= 400:
                # keep body for debugging
                raise requests.HTTPError(f"{r.status_code} {r.reason}: {r.text[:200]}", response=r)
            return r
        except Exception as e:
            last_err = e
            sleep_s = (backoff ** i)
            log(f"[WARN] HTTP attempt {i+1}/{tries} failed: {e}. sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"HTTP request failed after {tries} tries: {last_err}")


def fetch_daily_mean_f(lat: float, lon: float, days: int) -> Tuple[List[str], List[float]]:
    """
    Open-Meteo forecast endpoint does NOT support temperature_2m_mean.
    So we fetch max/min and compute mean = (max+min)/2.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "temperature_2m_min"],
        "temperature_unit": "fahrenheit",
        "forecast_days": int(days),
        "timezone": "UTC",
    }
    r = retry_get(url, params=params, tries=3, timeout=20)
    data = r.json()

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    tmax = daily.get("temperature_2m_max", [])
    tmin = daily.get("temperature_2m_min", [])

    if not dates or not tmax or not tmin:
        raise RuntimeError(f"Open-Meteo response missing fields: {json.dumps(data)[:300]}")

    if not (len(dates) == len(tmax) == len(tmin)):
        raise RuntimeError("Open-Meteo daily arrays length mismatch")

    tmean = [ (float(a) + float(b)) / 2.0 for a, b in zip(tmax, tmin) ]
    return dates, tmean


def daily_hdd(base_f: float, temp_mean_f: float) -> float:
    # HDD = max(0, base - mean_temp)
    return max(0.0, base_f - temp_mean_f)


def weighted_sum(values: List[float]) -> float:
    """
    Weighted sum with more weight on nearer days (front-loaded).
    Weights: N..1 (day1 gets N, dayN gets 1)
    """
    n = len(values)
    if n == 0:
        return 0.0
    weights = list(range(n, 0, -1))
    wsum = sum(w * v for w, v in zip(weights, values))
    wtot = sum(weights)
    return wsum / wtot


def safe_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(str(x).strip())
    except:
        return None


def load_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    return pd.DataFrame(columns=[
        "date",
        "hdd_15d",
        "hdd_30d",
        "delta_15d",
        "delta_30d",
        "signal",
        "storage_price",
        "ng_price",
        "cost_15d",
        "cost_30d",
    ])


def last_value(df: pd.DataFrame, col: str) -> Optional[float]:
    if df.empty or col not in df.columns:
        return None
    try:
        v = df.iloc[-1][col]
        if pd.isna(v):
            return None
        return float(v)
    except:
        return None


def compute_signal(delta15: float, delta30: float, h15: float, h30: float) -> str:
    # simple but readable
    if delta15 >= DELTA_BULL:
        return "🔥 Bullish (HDD up)"
    if delta15 <= DELTA_BEAR:
        return "🥶 Bearish (HDD down)"

    # neutral region: add a hint from 15 vs 30 slope
    if h15 > h30 * 1.05:
        return "🟡 Neutral (near-term colder than trend)"
    if h15 < h30 * 0.95:
        return "🟡 Neutral (near-term warmer than trend)"
    return "😐 Neutral"


def format_message(date_str: str, h15: float, h30: float, d15: float, d30: float,
                   storage_p: Optional[float], ng_p: Optional[float],
                   cost15: Optional[float], cost30: Optional[float],
                   signal: str) -> str:
    # Make it easy to read on Telegram
    lines = []
    lines.append(f"📊 HDD Update ({date_str})")
    lines.append("")
    lines.append(f"15D Weighted HDD: {h15:.2f}  ({d15:+.2f} vs prev)")
    lines.append(f"30D Weighted HDD: {h30:.2f}  ({d30:+.2f} vs prev)")
    lines.append("")
    if storage_p is not None or ng_p is not None:
        lines.append("💰 Pricing Inputs")
        if storage_p is not None:
            lines.append(f"Storage: {storage_p:.4f}")
        if ng_p is not None:
            lines.append(f"NG: {ng_p:.4f}")
        lines.append("")
    if cost15 is not None or cost30 is not None:
        lines.append("🧮 HDD × NG + Storage (Index)")
        if cost15 is not None:
            lines.append(f"Cost Index (15D): {cost15:.2f}")
        if cost30 is not None:
            lines.append(f"Cost Index (30D): {cost30:.2f}")
        lines.append("")
    lines.append(f"Signal: {signal}")
    return "\n".join(lines)


def tg_send_message(text: str) -> None:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        log("[INFO] Telegram not configured (TG_BOT_TOKEN / TG_CHAT_ID missing). Skip send.")
        return
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }
    r = retry_get(url, params=payload, tries=3, timeout=20)
    _ = r.text


def tg_send_photo(photo_path: str, caption: str = "") -> None:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        log("[INFO] Telegram not configured (TG_BOT_TOKEN / TG_CHAT_ID missing). Skip photo.")
        return
    if not os.path.exists(photo_path):
        log(f"[WARN] Chart not found: {photo_path}. Skip photo.")
        return
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendPhoto"
    with open(photo_path, "rb") as f:
        files = {"photo": f}
        data = {"chat_id": TG_CHAT_ID}
        if caption:
            data["caption"] = caption
        # use requests directly here (multipart)
        for i in range(3):
            try:
                rr = requests.post(url, data=data, files=files, timeout=30, headers={"User-Agent": USER_AGENT})
                if rr.status_code >= 400:
                    raise requests.HTTPError(f"{rr.status_code} {rr.reason}: {rr.text[:200]}")
                return
            except Exception as e:
                log(f"[WARN] Telegram photo attempt {i+1}/3 failed: {e}")
                time.sleep(1.5 ** i)
    raise RuntimeError("Telegram sendPhoto failed after 3 tries")


def make_chart(df: pd.DataFrame, out_path: str) -> None:
    # Lazy import so local run doesn’t require matplotlib until needed
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        return

    dfx = df.tail(CHART_HISTORY_ROWS).copy()
    # ensure date exists
    if "date" not in dfx.columns:
        return

    # parse date
    try:
        dfx["date"] = pd.to_datetime(dfx["date"])
    except:
        pass

    # Plot 15D & 30D HDD
    plt.figure(figsize=(10, 5))
    if "hdd_15d" in dfx.columns:
        plt.plot(dfx["date"], dfx["hdd_15d"], label="HDD 15D")
    if "hdd_30d" in dfx.columns:
        plt.plot(dfx["date"], dfx["hdd_30d"], label="HDD 30D")
    plt.title("Weighted HDD Trend (15D vs 30D)")
    plt.xlabel("Date")
    plt.ylabel("HDD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# -----------------------------
# MAIN
# -----------------------------
def run_system() -> None:
    today = dt.datetime.utcnow().date().isoformat()

    df = load_csv(CSV_PATH)

    # ---- pricing (from env -> else from last CSV)
    storage_p = safe_float(STORAGE_PRICE)
    ng_p = safe_float(NG_PRICE)

    if storage_p is None:
        storage_p = last_value(df, "storage_price")
    if ng_p is None:
        ng_p = last_value(df, "ng_price")

    # ---- fetch weather (30 days)
    # fallback if weather API fails: keep last values and still send a message
    dates: List[str] = []
    temps: List[float] = []
    weather_ok = True
    try:
        dates, temps = fetch_daily_mean_f(LAT, LON, FORECAST_DAYS)
    except Exception as e:
        weather_ok = False
        log(f"[ERROR] Weather fetch failed: {e}")

    if weather_ok:
        # compute daily HDD list for forecast horizon
        hdds = [daily_hdd(BASE_F, t) for t in temps]

        h15 = weighted_sum(hdds[:15])
        h30 = weighted_sum(hdds[:30])

    else:
        # fallback to last known values
        prev_h15 = last_value(df, "hdd_15d")
        prev_h30 = last_value(df, "hdd_30d")
        if prev_h15 is None or prev_h30 is None:
            raise RuntimeError("Weather API failed and no historical HDD in CSV to fallback.")
        h15 = float(prev_h15)
        h30 = float(prev_h30)

    # ---- deltas vs prev row
    prev_h15 = last_value(df, "hdd_15d")
    prev_h30 = last_value(df, "hdd_30d")
    d15 = (h15 - prev_h15) if prev_h15 is not None else 0.0
    d30 = (h30 - prev_h30) if prev_h30 is not None else 0.0

    # ---- combined index
    # Cost Index = HDD * NG + Storage
    cost15 = None
    cost30 = None
    if ng_p is not None:
        cost15 = (h15 * ng_p) + (storage_p or 0.0)
        cost30 = (h30 * ng_p) + (storage_p or 0.0)

    signal = compute_signal(d15, d30, h15, h30)

    # ---- upsert (avoid duplicate date)
    row = {
        "date": today,
        "hdd_15d": round(h15, 4),
        "hdd_30d": round(h30, 4),
        "delta_15d": round(d15, 4),
        "delta_30d": round(d30, 4),
        "signal": signal,
        "storage_price": (round(storage_p, 6) if storage_p is not None else ""),
        "ng_price": (round(ng_p, 6) if ng_p is not None else ""),
        "cost_15d": (round(cost15, 6) if cost15 is not None else ""),
        "cost_30d": (round(cost30, 6) if cost30 is not None else ""),
    }

    if not df.empty and "date" in df.columns and str(df.iloc[-1]["date"]) == today:
        # replace last row
        for k, v in row.items():
            df.at[df.index[-1], k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # ensure column order
    wanted_cols = [
        "date",
        "hdd_15d",
        "hdd_30d",
        "delta_15d",
        "delta_30d",
        "signal",
        "storage_price",
        "ng_price",
        "cost_15d",
        "cost_30d",
    ]
    for c in wanted_cols:
        if c not in df.columns:
            df[c] = ""
    df = df[wanted_cols]

    df.to_csv(CSV_PATH, index=False)

    # ---- chart + telegram
    make_chart(df, CHART_PATH)

    msg = format_message(
        date_str=today,
        h15=h15, h30=h30,
        d15=d15, d30=d30,
        storage_p=storage_p, ng_p=ng_p,
        cost15=cost15, cost30=cost30,
        signal=signal
    )

    # send text + chart
    tg_send_message(msg)
    tg_send_photo(CHART_PATH, caption="📈 HDD Trend (15D vs 30D)")

    log("✅ HDD system finished.")


if __name__ == "__main__":
    try:
        run_system()
    except Exception as e:
        log(f"[FATAL] {e}")
        raise
