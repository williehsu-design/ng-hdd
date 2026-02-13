#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HDD/CDD Daily Monitor
- Fetch weather daily mean temp (Open-Meteo) with past_days + forecast_days (forecast_days max 16)
- Compute HDD/CDD (base 65F)
- Compute 15D and 30D weighted (more weight to nearer days)
- Save CSV: ng_hdd_data.csv
- Plot chart: hdd_chart.png
- Send Telegram message + chart (sendMessage + sendPhoto)
Secrets:
  - TG_BOT_TOKEN
  - TG_CHAT_ID
"""

from __future__ import annotations

import os
import sys
import time
import math
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict

import requests
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Config
# =========================
LAT = float(os.getenv("LAT", "40.7128"))        # NYC default
LON = float(os.getenv("LON", "-74.0060"))
TEMP_UNIT = os.getenv("TEMP_UNIT", "fahrenheit")  # "fahrenheit" or "celsius"
BASE_F = float(os.getenv("BASE_F", "65.0"))

# Open-Meteo forecast API limits: forecast_days max 16
PAST_DAYS = int(os.getenv("PAST_DAYS", "30"))         # history to include (used for chart context)
FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", "16")) # must be <= 16

CSV_PATH = os.getenv("CSV_PATH", "ng_hdd_data.csv")
CHART_PATH = os.getenv("CHART_PATH", "hdd_chart.png")

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "").strip()

# Telegram options
TG_PARSE_MODE = os.getenv("TG_PARSE_MODE", "")  # leave empty to avoid formatting surprises
TG_DISABLE_WEB_PAGE_PREVIEW = True

# Behavior
HTTP_TIMEOUT = 25
HTTP_RETRIES = 3
RETRY_SLEEP_BASE = 1.0


# =========================
# Helpers
# =========================
def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def retry_get(url: str, params: Dict, tries: int = HTTP_RETRIES, timeout: int = HTTP_TIMEOUT) -> requests.Response:
    last_err = None
    for i in range(1, tries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code >= 400:
                # include response json if exists for easier debugging
                try:
                    msg = r.json()
                except Exception:
                    msg = r.text[:300]
                raise requests.HTTPError(f"{r.status_code} {r.reason}: {msg}", response=r)
            return r
        except Exception as e:
            last_err = e
            sleep_s = RETRY_SLEEP_BASE * (1.0 + i * 0.5)
            print(f"[WARN] HTTP attempt {i}/{tries} failed: {e}. sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"HTTP request failed after {tries} tries: {last_err}")


def safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return default
        return float(x)
    except Exception:
        return default


def weighted_avg(values: List[float], weights: List[float]) -> float:
    if not values or not weights or len(values) != len(weights):
        return 0.0
    s_w = sum(weights)
    if s_w == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / s_w


def make_linear_weights(n: int) -> List[float]:
    # 1..n (nearest day gets larger weight if we order oldest->newest)
    if n <= 0:
        return []
    return list(range(1, n + 1))


# =========================
# Weather + HDD/CDD
# =========================
def fetch_daily_mean_temp(lat: float, lon: float, past_days: int, forecast_days: int, temp_unit: str) -> Tuple[List[str], List[float]]:
    """
    Uses Open-Meteo Forecast API:
      - daily=temperature_2m_mean
      - past_days=...
      - forecast_days=... (max 16)
    """
    if forecast_days > 16:
        raise ValueError("FORECAST_DAYS must be <= 16 due to Open-Meteo API limitation.")

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_mean",
        "temperature_unit": temp_unit,
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }

    r = retry_get(url, params=params, tries=HTTP_RETRIES, timeout=HTTP_TIMEOUT)
    data = r.json()

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    temps = daily.get("temperature_2m_mean", [])

    if not dates or not temps or len(dates) != len(temps):
        raise RuntimeError(f"Unexpected weather payload: {json.dumps(data)[:400]}")

    temps_f = [safe_float(t) for t in temps]
    return dates, temps_f


def compute_hdd_cdd_series(temps_f: List[float], base_f: float) -> Tuple[List[float], List[float]]:
    hdd = [max(0.0, base_f - t) for t in temps_f]
    cdd = [max(0.0, t - base_f) for t in temps_f]
    return hdd, cdd


@dataclass
class Metrics:
    date: str
    hdd15: float
    hdd30: float
    cdd15: float
    cdd30: float
    delta_hdd15: float
    delta_hdd30: float
    signal: str


def derive_signal(delta_hdd15: float, delta_hdd30: float) -> str:
    # Very simple heuristic: you can refine later
    # Positive delta => colder => potentially bullish NG (more heating demand)
    if delta_hdd15 >= 10 or delta_hdd30 >= 10:
        return "🔥 Bullish (colder revision)"
    if delta_hdd15 >= 3 or delta_hdd30 >= 3:
        return "📈 Mild Bullish"
    if delta_hdd15 <= -10 or delta_hdd30 <= -10:
        return "🧊 Bearish (warmer revision)"
    if delta_hdd15 <= -3 or delta_hdd30 <= -3:
        return "📉 Mild Bearish"
    return "😐 Neutral"


def compute_metrics(dates: List[str], temps_f: List[float]) -> Metrics:
    hdd, cdd = compute_hdd_cdd_series(temps_f, BASE_F)

    # Use the MOST RECENT forecast window for 15/30 (from the end of the combined past+forecast series)
    # We take last N days from the available series.
    def w_metric(series: List[float], n: int) -> float:
        n = min(n, len(series))
        chunk = series[-n:]
        w = make_linear_weights(n)  # oldest->newest within chunk
        return weighted_avg(chunk, w)

    hdd15 = w_metric(hdd, 15)
    hdd30 = w_metric(hdd, 30)
    cdd15 = w_metric(cdd, 15)
    cdd30 = w_metric(cdd, 30)

    today = dates[-1]  # last date in series

    # For deltas, compare to yesterday's saved CSV if exists
    prev_hdd15 = None
    prev_hdd30 = None
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            if len(df) > 0:
                prev_hdd15 = safe_float(df.iloc[-1].get("hdd_15d"))
                prev_hdd30 = safe_float(df.iloc[-1].get("hdd_30d"))
        except Exception as e:
            print(f"[WARN] Failed reading previous CSV for delta: {e}")

    delta_hdd15 = (hdd15 - prev_hdd15) if prev_hdd15 is not None else 0.0
    delta_hdd30 = (hdd30 - prev_hdd30) if prev_hdd30 is not None else 0.0

    signal = derive_signal(delta_hdd15, delta_hdd30)

    return Metrics(
        date=today,
        hdd15=hdd15,
        hdd30=hdd30,
        cdd15=cdd15,
        cdd30=cdd30,
        delta_hdd15=delta_hdd15,
        delta_hdd30=delta_hdd30,
        signal=signal,
    )


# =========================
# Output: CSV + Chart
# =========================
def upsert_csv(m: Metrics) -> None:
    row = {
        "date": m.date,
        "hdd_15d": round(m.hdd15, 3),
        "hdd_30d": round(m.hdd30, 3),
        "cdd_15d": round(m.cdd15, 3),
        "cdd_30d": round(m.cdd30, 3),
        "delta_hdd_15d": round(m.delta_hdd15, 3),
        "delta_hdd_30d": round(m.delta_hdd30, 3),
        "signal": m.signal,
        "updated_utc": _now_utc_str(),
    }

    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    # If last row is same date, overwrite; else append
    if len(df) > 0 and str(df.iloc[-1].get("date")) == m.date:
        df.iloc[-1] = row
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(CSV_PATH, index=False)
    print(f"[OK] CSV updated: {CSV_PATH}")


def plot_chart(dates: List[str], temps_f: List[float], m: Metrics) -> None:
    hdd, cdd = compute_hdd_cdd_series(temps_f, BASE_F)

    # chart window: show last (PAST_DAYS + FORECAST_DAYS) but keep safe
    n = len(dates)
    x = list(range(n))

    plt.figure(figsize=(12, 6))
    plt.plot(x, hdd, label="Daily HDD (base 65F)")
    plt.plot(x, cdd, label="Daily CDD (base 65F)")

    # annotate metrics
    title = f"HDD/CDD Trend | {m.date} | 15D HDD={m.hdd15:.2f} / 30D HDD={m.hdd30:.2f} | 15D CDD={m.cdd15:.2f} / 30D CDD={m.cdd30:.2f}"
    plt.title(title)
    plt.xlabel("Day Index (Past -> Forecast)")
    plt.ylabel("Degree Days")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # label some x ticks
    tick_idx = list(range(0, n, max(1, n // 8)))
    plt.xticks(tick_idx, [dates[i] for i in tick_idx], rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=160)
    plt.close()
    print(f"[OK] Chart saved: {CHART_PATH}")


# =========================
# Telegram
# =========================
def tg_send_message(text: str) -> None:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("[INFO] Telegram secrets not set; skip sendMessage.")
        return

    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": text,
        "disable_web_page_preview": TG_DISABLE_WEB_PAGE_PREVIEW,
    }
    if TG_PARSE_MODE:
        payload["parse_mode"] = TG_PARSE_MODE

    r = retry_get(url, params=payload, tries=HTTP_RETRIES, timeout=HTTP_TIMEOUT)
    _ = r.json()
    print("[OK] Telegram message sent.")


def tg_send_photo(photo_path: str, caption: str = "") -> None:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("[INFO] Telegram secrets not set; skip sendPhoto.")
        return
    if not os.path.exists(photo_path):
        print(f"[WARN] Chart not found for Telegram: {photo_path}")
        return

    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendPhoto"
    with open(photo_path, "rb") as f:
        files = {"photo": f}
        data = {
            "chat_id": TG_CHAT_ID,
            "caption": caption[:1024],  # Telegram caption limit
        }
        if TG_PARSE_MODE:
            data["parse_mode"] = TG_PARSE_MODE
        r = requests.post(url, data=data, files=files, timeout=HTTP_TIMEOUT)
        if r.status_code >= 400:
            try:
                msg = r.json()
            except Exception:
                msg = r.text[:300]
            raise RuntimeError(f"Telegram sendPhoto failed: {r.status_code} {r.reason}: {msg}")
    print("[OK] Telegram photo sent.")


def build_pretty_message(m: Metrics) -> str:
    # nicer, easier-to-read formatting
    arrow15 = "⬆️" if m.delta_hdd15 > 0 else ("⬇️" if m.delta_hdd15 < 0 else "➡️")
    arrow30 = "⬆️" if m.delta_hdd30 > 0 else ("⬇️" if m.delta_hdd30 < 0 else "➡️")

    lines = [
        f"🛰️ HDD/CDD Update ({m.date})",
        "",
        f"🔥 HDD (base 65F)",
        f"• 15D Weighted: {m.hdd15:.2f}  ({arrow15} Δ {m.delta_hdd15:+.2f})",
        f"• 30D Weighted: {m.hdd30:.2f}  ({arrow30} Δ {m.delta_hdd30:+.2f})",
        "",
        f"🌤️ CDD (base 65F)",
        f"• 15D Weighted: {m.cdd15:.2f}",
        f"• 30D Weighted: {m.cdd30:.2f}",
        "",
        f"📌 Signal: {m.signal}",
        "",
        f"⏱ Updated: {_now_utc_str()}",
    ]
    return "\n".join(lines)


# =========================
# Main
# =========================
def run() -> int:
    # guard forecast_days limit
    if FORECAST_DAYS > 16:
        print("[WARN] FORECAST_DAYS > 16 is invalid for Open-Meteo; forcing to 16.")
        fd = 16
    else:
        fd = FORECAST_DAYS

    # Fetch weather
    dates, temps = fetch_daily_mean_temp(LAT, LON, PAST_DAYS, fd, TEMP_UNIT)

    # Metrics
    m = compute_metrics(dates, temps)

    # Save CSV + chart
    upsert_csv(m)
    plot_chart(dates, temps, m)

    # Telegram: send text + chart
    msg = build_pretty_message(m)
    tg_send_message(msg)
    tg_send_photo(CHART_PATH, caption=f"HDD/CDD Trend ({m.date})")

    print("[OK] Done.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)
