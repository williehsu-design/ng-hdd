# -*- coding: utf-8 -*-
"""
HDD/CDD Monitor (15D / 30D) + Telegram + Trend Chart + EIA Storage helper

- Weather: Open-Meteo daily temperature_2m_mean (F)
  * Uses past_days + forecast_days
  * NOTE: Open-Meteo 'forecast_days' has an upper limit (often 16). We respect it.
- Metrics:
  * Daily HDD = max(0, BASE_F - TmeanF)
  * Daily CDD = max(0, TmeanF - BASE_F)
  * Weighted 15D / 30D using linear weights (recent days weigh more)
- Storage:
  * Pulls EIA Weekly Natural Gas Storage Report JSON
  * Uses EIA release_date as truth (holiday weeks shift automatically)
- Output:
  * Append row to CSV
  * Generate trend chart PNG
  * Send Telegram text + photo
"""

import os
import sys
import json
import time
import math
import csv
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Optional

import requests
import pandas as pd
import matplotlib.pyplot as plt

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None


# =========================
# User-configurable settings
# =========================

# Location (example: New York area). Change if needed.
LAT = float(os.getenv("LAT", "40.7128"))
LON = float(os.getenv("LON", "-74.0060"))

# Degree-day base temperature (F)
BASE_F = float(os.getenv("BASE_F", "65.0"))

# Weather window controls
# We want 30D metric. We'll build it using (past_days + forecast_days) >= 30.
# Open-Meteo often limits forecast_days <= 16, so we default forecast_days=16, past_days=20 (36 total).
FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", "16"))  # keep <= 16 to avoid 400 errors
PAST_DAYS = int(os.getenv("PAST_DAYS", "20"))          # adjust so past+forecast >= 30

# Output files
CSV_PATH = os.getenv("CSV_PATH", "ng_hdd_data.csv")
CHART_PATH = os.getenv("CHART_PATH", "hdd_chart.png")

# Telegram (GitHub Secrets)
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "").strip()

# EIA storage JSON (official)
EIA_WNGSR_JSON = "https://ir.eia.gov/ngs/wngsr.json"

# Open-Meteo API
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


# =========================
# Helpers
# =========================

def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def today_utc_date_str() -> str:
    return utcnow().date().isoformat()

def sleep_backoff(attempt: int) -> None:
    # attempt starts at 1
    base = 0.8
    t = base * (1.8 ** (attempt - 1))
    time.sleep(min(t, 6.0))

def retry_get(url: str, params: dict, tries: int = 3, timeout: int = 20) -> requests.Response:
    last_err = None
    for i in range(1, tries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code >= 400:
                # Print body to help debugging
                try:
                    body = r.text[:300]
                except Exception:
                    body = "<no body>"
                print(f"[WARN] HTTP attempt {i}/{tries} failed: {r.status_code} {r.reason}: {body}")
                r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            if i < tries:
                sleep_backoff(i)
            else:
                raise RuntimeError(f"HTTP request failed after {tries} tries: {last_err}") from last_err

def linear_weights(n: int) -> List[float]:
    """1..n normalized (recent = larger if list is chronological)"""
    if n <= 0:
        return []
    w = list(range(1, n + 1))
    s = float(sum(w))
    return [x / s for x in w]

def weighted_sum(values: List[float]) -> float:
    w = linear_weights(len(values))
    return float(sum(v * wi for v, wi in zip(values, w)))

def dd_from_temp(base_f: float, tmean_f: float) -> Tuple[float, float]:
    hdd = max(0.0, base_f - tmean_f)
    cdd = max(0.0, tmean_f - base_f)
    return hdd, cdd


# =========================
# Weather
# =========================

def fetch_daily_mean_f(lat: float, lon: float, past_days: int, forecast_days: int) -> Tuple[List[str], List[float]]:
    """
    Returns (dates[], tmean_f[]) in UTC-date strings, from past_days back through forecast_days forward.
    We request timezone=UTC so dates are stable and NOT affected by US DST.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_mean",
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
        "past_days": str(past_days),
        "forecast_days": str(forecast_days),
    }
    r = retry_get(OPEN_METEO_URL, params=params, tries=3, timeout=25)
    j = r.json()
    daily = j.get("daily", {})
    dates = daily.get("time", [])
    temps = daily.get("temperature_2m_mean", [])
    if not dates or not temps or len(dates) != len(temps):
        raise RuntimeError(f"Weather payload invalid: dates={len(dates)} temps={len(temps)}")
    return dates, [float(x) for x in temps]


def compute_hdd_cdd_15_30(base_f: float, dates: List[str], temps_f: List[float]) -> dict:
    """
    Uses the latest 15 and 30 daily values from the (past+forecast) sequence.
    Sequence is chronological in Open-Meteo (old -> new).
    """
    daily_hdd = []
    daily_cdd = []
    for t in temps_f:
        h, c = dd_from_temp(base_f, t)
        daily_hdd.append(h)
        daily_cdd.append(c)

    if len(daily_hdd) < 30:
        raise RuntimeError(f"Need at least 30 days to compute 30D, got {len(daily_hdd)}")

    h15 = daily_hdd[-15:]
    h30 = daily_hdd[-30:]
    c15 = daily_cdd[-15:]
    c30 = daily_cdd[-30:]

    return {
        "daily_dates": dates,
        "daily_hdd": daily_hdd,
        "daily_cdd": daily_cdd,
        "hdd_15d": weighted_sum(h15),
        "hdd_30d": weighted_sum(h30),
        "cdd_15d": weighted_sum(c15),
        "cdd_30d": weighted_sum(c30),
    }


# =========================
# EIA Storage (C1)
# =========================

@dataclass
class StorageInfo:
    release_date_utc: Optional[dt.datetime]  # based on EIA JSON release_date (date-only effectively)
    current_week: Optional[str]             # e.g. "2026-02-06"
    total_bcf: Optional[float]              # Lower 48 total (bcf)
    net_change_bcf: Optional[float]         # calculated net_change in JSON
    fiveyr_avg_bcf: Optional[float]         # calculated 5yr-avg
    ok: bool
    reason: str

def fetch_eia_storage() -> StorageInfo:
    """
    Pull EIA WNGSR JSON and return L48 total + net change + release date.
    Holiday weeks are already handled by EIA release schedule: the JSON updates on the actual release.
    """
    try:
        r = retry_get(EIA_WNGSR_JSON, params={}, tries=3, timeout=25)
        j = r.json()

        release_date_str = j.get("release_date")  # e.g. "2026-Feb-12 00:00:00"
        release_dt_utc = None
        if release_date_str:
            # release_date in JSON is not explicitly tz; treat it as UTC date boundary (good enough for “new release” detection)
            # We'll parse it as naive then attach UTC.
            try:
                release_dt_utc = dt.datetime.strptime(release_date_str, "%Y-%b-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
            except Exception:
                release_dt_utc = None

        current_week = j.get("current_week")
        series = j.get("series", [])

        # Find "total lower 48 states"
        total_series = None
        for s in series:
            if str(s.get("name", "")).strip().lower() == "total lower 48 states":
                total_series = s
                break

        if not total_series:
            return StorageInfo(release_dt_utc, current_week, None, None, None, False, "EIA JSON missing lower 48 series")

        data = total_series.get("data", [])
        calc = total_series.get("calculated", {})

        total_bcf = None
        if data and isinstance(data, list) and len(data) > 0:
            # first item is [current_week_date, value]
            total_bcf = float(data[0][1])

        net_change = calc.get("net_change")
        net_change_bcf = float(net_change) if net_change is not None else None

        fiveyr = calc.get("5yr-avg")
        fiveyr_avg_bcf = float(fiveyr) if fiveyr is not None else None

        return StorageInfo(release_dt_utc, current_week, total_bcf, net_change_bcf, fiveyr_avg_bcf, True, "ok")
    except Exception as e:
        return StorageInfo(None, None, None, None, None, False, f"Storage fetch failed: {e}")


# =========================
# CSV + Chart
# =========================

def load_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame(columns=[
        "run_utc",
        "hdd_15d", "hdd_30d",
        "cdd_15d", "cdd_30d",
        "storage_release_utc", "storage_week",
        "storage_total_bcf", "storage_net_change_bcf", "storage_5yr_avg_bcf",
    ])

def append_row(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    df2 = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    # sort by run_utc if possible
    if "run_utc" in df2.columns:
        try:
            df2["run_utc"] = pd.to_datetime(df2["run_utc"], utc=True)
            df2 = df2.sort_values("run_utc").reset_index(drop=True)
            df2["run_utc"] = df2["run_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    return df2

def make_trend_chart(dates: List[str], daily_hdd: List[float], daily_cdd: List[float],
                     title: str, out_path: str) -> None:
    """
    Plot last 30 points (or all if shorter) for daily HDD & CDD.
    """
    n = min(30, len(dates))
    x = dates[-n:]
    h = daily_hdd[-n:]
    c = daily_cdd[-n:]

    plt.figure(figsize=(12, 5))
    plt.plot(x, h, label=f"Daily HDD (base {BASE_F:.0f}F)")
    plt.plot(x, c, label=f"Daily CDD (base {BASE_F:.0f}F)")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Degree Days")
    plt.xlabel("Day (UTC)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# =========================
# Telegram
# =========================

def tg_send_message(token: str, chat_id: str, text: str) -> None:
    if not token or not chat_id:
        print("[INFO] Telegram token/chat_id missing; skip message")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    r = retry_get(url, params=payload, tries=3, timeout=25)
    _ = r.text  # consume

def tg_send_photo(token: str, chat_id: str, photo_path: str, caption: str = "") -> None:
    if not token or not chat_id:
        print("[INFO] Telegram token/chat_id missing; skip photo")
        return
    if not os.path.exists(photo_path):
        print("[WARN] Chart not found; skip photo")
        return

    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    with open(photo_path, "rb") as f:
        files = {"photo": f}
        data = {"chat_id": chat_id, "caption": caption}
        # use requests directly (multipart)
        for i in range(1, 4):
            try:
                rr = requests.post(url, data=data, files=files, timeout=40)
                if rr.status_code >= 400:
                    print(f"[WARN] TG photo attempt {i}/3 failed: {rr.status_code} {rr.text[:200]}")
                    rr.raise_for_status()
                return
            except Exception as e:
                if i < 3:
                    sleep_backoff(i)
                else:
                    raise RuntimeError(f"Telegram photo failed: {e}") from e


# =========================
# Signal logic (simple)
# =========================

def pick_signal(hdd15_delta: float) -> str:
    """
    Very simple: HDD up => Bullish (colder), down => Bearish (warmer)
    """
    if hdd15_delta > 2.0:
        return "🔥 Bullish (colder revision)"
    if hdd15_delta < -2.0:
        return "🧊 Bearish (warmer revision)"
    return "😐 Neutral"

def fmt(x: Optional[float], nd=2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


# =========================
# Main
# =========================

def run_system() -> None:
    run_dt = utcnow()
    run_utc_str = run_dt.strftime("%Y-%m-%d %H:%M:%S")

    # 1) Load history
    df = load_csv(CSV_PATH)

    # previous values (for delta)
    prev_h15 = None
    prev_h30 = None
    prev_c15 = None
    prev_c30 = None
    if len(df) > 0:
        try:
            prev_h15 = float(df.iloc[-1]["hdd_15d"]) if "hdd_15d" in df.columns else None
            prev_h30 = float(df.iloc[-1]["hdd_30d"]) if "hdd_30d" in df.columns else None
            prev_c15 = float(df.iloc[-1]["cdd_15d"]) if "cdd_15d" in df.columns else None
            prev_c30 = float(df.iloc[-1]["cdd_30d"]) if "cdd_30d" in df.columns else None
        except Exception:
            pass

    # 2) Weather -> HDD/CDD
    dates, temps = fetch_daily_mean_f(LAT, LON, past_days=PAST_DAYS, forecast_days=FORECAST_DAYS)
    metrics = compute_hdd_cdd_15_30(BASE_F, dates, temps)

    h15 = float(metrics["hdd_15d"])
    h30 = float(metrics["hdd_30d"])
    c15 = float(metrics["cdd_15d"])
    c30 = float(metrics["cdd_30d"])

    d_h15 = (h15 - prev_h15) if prev_h15 is not None else 0.0
    d_h30 = (h30 - prev_h30) if prev_h30 is not None else 0.0

    # 3) Storage (C1): fetch; decide “new release” by comparing release_date to last stored
    storage = fetch_eia_storage()

    last_storage_release = None
    if "storage_release_utc" in df.columns and len(df) > 0:
        try:
            # keep as string in csv; parse back
            s = str(df.iloc[-1].get("storage_release_utc", "")).strip()
            if s and s != "nan":
                last_storage_release = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            last_storage_release = None

    storage_is_new = False
    if storage.ok and storage.release_date_utc is not None:
        if last_storage_release is None or storage.release_date_utc > last_storage_release:
            storage_is_new = True

    # 4) Append row
    row = {
        "run_utc": run_utc_str,
        "hdd_15d": h15,
        "hdd_30d": h30,
        "cdd_15d": c15,
        "cdd_30d": c30,
        "storage_release_utc": storage.release_date_utc.isoformat().replace("+00:00", "Z") if storage.release_date_utc else "",
        "storage_week": storage.current_week or "",
        "storage_total_bcf": storage.total_bcf if storage.total_bcf is not None else "",
        "storage_net_change_bcf": storage.net_change_bcf if storage.net_change_bcf is not None else "",
        "storage_5yr_avg_bcf": storage.fiveyr_avg_bcf if storage.fiveyr_avg_bcf is not None else "",
    }
    df2 = append_row(df, row)
    df2.to_csv(CSV_PATH, index=False)

    # 5) Chart
    chart_title = f"HDD/CDD Trend · {today_utc_date_str()} UTC"
    make_trend_chart(metrics["daily_dates"], metrics["daily_hdd"], metrics["daily_cdd"], chart_title, CHART_PATH)

    # 6) Telegram message (cleaner)
    signal = pick_signal(d_h15)
    lines = []
    lines.append(f"📌 HDD/CDD Update ({today_utc_date_str()})")
    lines.append("")
    lines.append(f"🔥 HDD (base {BASE_F:.0f}F)")
    lines.append(f" • 15D Weighted: {fmt(h15)}   (Δ {fmt(d_h15)})")
    lines.append(f" • 30D Weighted: {fmt(h30)}   (Δ {fmt(d_h30)})")
    lines.append("")
    lines.append(f"🌤️ CDD (base {BASE_F:.0f}F)")
    lines.append(f" • 15D Weighted: {fmt(c15)}")
    lines.append(f" • 30D Weighted: {fmt(c30)}")
    lines.append("")
    if storage.ok and storage.total_bcf is not None:
        tag = "🟢 NEW" if storage_is_new else "ℹ️"
        lines.append(f"🧱 Storage (EIA WNGSR) {tag}")
        lines.append(f" • Week: {storage.current_week or 'NA'}")
        lines.append(f" • Total (L48): {fmt(storage.total_bcf, 0)} bcf")
        if storage.net_change_bcf is not None:
            lines.append(f" • Net change: {fmt(storage.net_change_bcf, 0)} bcf")
        if storage.fiveyr_avg_bcf is not None:
            lines.append(f" • 5Y avg: {fmt(storage.fiveyr_avg_bcf, 0)} bcf")
    else:
        lines.append(f"🧱 Storage: NA ({storage.reason})")
    lines.append("")
    lines.append(f"📍 Signal: {signal}")
    lines.append(f"⏱️ Updated: {run_utc_str} UTC")

    msg = "\n".join(lines)

    tg_send_message(TG_BOT_TOKEN, TG_CHAT_ID, msg)
    tg_send_photo(TG_BOT_TOKEN, TG_CHAT_ID, CHART_PATH, caption=f"📈 Trend (HDD/CDD) · {today_utc_date_str()} UTC")

    print("[OK] Updated CSV + chart + Telegram sent")


if __name__ == "__main__":
    run_system()
