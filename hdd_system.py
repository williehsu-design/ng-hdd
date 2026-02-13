#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import math
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Optional

import requests
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Config
# =========================
LAT = float(os.getenv("LAT", "40.7128"))
LON = float(os.getenv("LON", "-74.0060"))
BASE_F = float(os.getenv("BASE_F", "65.0"))

CSV_PATH = os.getenv("CSV_PATH", "ng_hdd_data.csv")
CHART_PATH = os.getenv("CHART_PATH", "hdd_chart.png")

# Rolling windows
WIN_15 = int(os.getenv("WIN_15", "15"))
WIN_30 = int(os.getenv("WIN_30", "30"))

# Telegram
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "").strip()

# Optional prices
# (你可以在 GitHub Actions secrets 或 variables 設 NG_PRICE / STORAGE_PRICE)
NG_PRICE = os.getenv("NG_PRICE", "").strip()
STORAGE_PRICE = os.getenv("STORAGE_PRICE", "").strip()

# If you still want “forecast look-ahead” you can add another flow later.
# This version focuses on stable "past observed" HDD to avoid forecast_days limits.


# =========================
# Helpers
# =========================
def utc_today() -> dt.date:
    return dt.datetime.utcnow().date()


def to_float_or_none(x: str) -> Optional[float]:
    try:
        if x is None or str(x).strip() == "":
            return None
        return float(x)
    except Exception:
        return None


def retry_get(url: str, params: dict, tries: int = 3, timeout: int = 25) -> requests.Response:
    last_err = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code >= 400:
                # include response body for easier debugging
                raise RuntimeError(f"{r.status_code} {r.reason}: {r.text}")
            return r
        except Exception as e:
            last_err = e
            sleep_s = 1.0 + i * 1.3
            print(f"[WARN] HTTP attempt {i+1}/{tries} failed: {e}. sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"HTTP request failed after {tries} tries: {last_err}")


def compute_hdd_series(mean_temps_f: List[float], base_f: float) -> List[float]:
    # HDD = max(0, base - temp)
    return [max(0.0, base_f - float(t)) for t in mean_temps_f]


def weighted_avg(values: List[float]) -> float:
    # Linear weights: older low, newer high (1..N)
    n = len(values)
    if n <= 0:
        return float("nan")
    weights = list(range(1, n + 1))
    s = sum(v * w for v, w in zip(values, weights))
    w = sum(weights)
    return s / w


def signal_from_delta(delta: float) -> str:
    # You can tune these thresholds
    if delta >= 10:
        return "🔥 Bullish"
    if delta <= -10:
        return "🧊 Bearish"
    return "😐 Neutral"


def fmt(x: Optional[float], nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{nd}f}"


# =========================
# Weather Fetch (ARCHIVE)
# =========================
def fetch_daily_mean_f_archive(lat: float, lon: float, days: int) -> Tuple[List[str], List[float]]:
    """
    Fetch past 'days' daily mean temperatures (F) using Open-Meteo Archive API.
    End at yesterday UTC (archive is stable).
    """
    if days <= 0:
        raise ValueError("days must be > 0")

    end_date = utc_today() - dt.timedelta(days=1)
    start_date = end_date - dt.timedelta(days=days - 1)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "temperature_2m_mean",
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
    }

    r = retry_get(url, params=params, tries=3, timeout=25)
    data = r.json()

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    temps = daily.get("temperature_2m_mean", [])

    if not dates or not temps or len(dates) != len(temps):
        raise RuntimeError(f"Archive API returned unexpected payload: {json.dumps(data)[:400]}")

    return dates, [float(t) for t in temps]


# =========================
# Telegram
# =========================
def tg_send_message(text: str) -> None:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("[INFO] Telegram not configured (missing TG_BOT_TOKEN/TG_CHAT_ID). Skip send_message.")
        return
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    r = requests.post(url, data=payload, timeout=25)
    if r.status_code >= 400:
        raise RuntimeError(f"Telegram sendMessage failed: {r.status_code} {r.text}")


def tg_send_photo(photo_path: str, caption: str = "") -> None:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("[INFO] Telegram not configured (missing TG_BOT_TOKEN/TG_CHAT_ID). Skip send_photo.")
        return
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendPhoto"
    with open(photo_path, "rb") as f:
        files = {"photo": f}
        data = {"chat_id": TG_CHAT_ID, "caption": caption}
        r = requests.post(url, data=data, files=files, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"Telegram sendPhoto failed: {r.status_code} {r.text}")


# =========================
# Chart
# =========================
def plot_chart(df: pd.DataFrame, out_path: str) -> None:
    # Plot last 90 rows (if exists)
    view = df.tail(90).copy()

    plt.figure(figsize=(10.5, 5.2))
    plt.plot(view["date"], view["hdd_15d"], label="HDD 15D (weighted)")
    plt.plot(view["date"], view["hdd_30d"], label="HDD 30D (weighted)")

    plt.xticks(rotation=45, ha="right")
    plt.title("HDD Trend (15D / 30D weighted)")
    plt.xlabel("Date (UTC)")
    plt.ylabel("HDD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# =========================
# Main
# =========================
def run_system() -> None:
    # We need enough days to compute both windows.
    need_days = max(WIN_15, WIN_30)

    try:
        dates, temps = fetch_daily_mean_f_archive(LAT, LON, need_days)
    except Exception as e:
        # Fallback: if archive fails, try to continue from existing CSV (if any)
        print(f"[ERROR] Weather fetch failed: {e}")
        if os.path.exists(CSV_PATH):
            print("[WARN] Using historical CSV only (no new update).")
            df = pd.read_csv(CSV_PATH)
            if df.empty:
                raise RuntimeError("CSV exists but empty; cannot fallback.")
            # Still re-render chart & telegram using last row
            plot_chart(df, CHART_PATH)
            last = df.iloc[-1].to_dict()
            send_summary(df, last, chart_path=CHART_PATH)
            return
        raise RuntimeError("Weather API failed and no historical HDD in CSV to fallback.")

    hdds = compute_hdd_series(temps, BASE_F)

    # Build a small dataframe for the fetched window
    dff = pd.DataFrame({
        "date": dates,
        "temp_mean_f": temps,
        "hdd": hdds,
    })

    # Compute weighted windows
    # For each date i, use last WIN_N values up to i
    def rolling_weighted(series: List[float], win: int) -> List[float]:
        out = []
        for i in range(len(series)):
            start = max(0, i - win + 1)
            chunk = series[start:i+1]
            out.append(weighted_avg(chunk))
        return out

    dff["hdd_15d"] = rolling_weighted(dff["hdd"].tolist(), WIN_15)
    dff["hdd_30d"] = rolling_weighted(dff["hdd"].tolist(), WIN_30)

    # Load existing CSV, append/merge by date
    if os.path.exists(CSV_PATH):
        old = pd.read_csv(CSV_PATH)
        if "date" in old.columns:
            df = pd.concat([old, dff], ignore_index=True)
            df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
        else:
            df = dff.copy()
    else:
        df = dff.copy()

    # Compute deltas vs yesterday (previous row)
    df["delta_15"] = df["hdd_15d"].diff()
    df["delta_30"] = df["hdd_30d"].diff()

    # Signals
    df["signal_15"] = df["delta_15"].apply(lambda x: signal_from_delta(x) if pd.notna(x) else "—")
    df["signal_30"] = df["delta_30"].apply(lambda x: signal_from_delta(x) if pd.notna(x) else "—")

    # Prices (optional)
    ng_price = to_float_or_none(NG_PRICE)
    storage_price = to_float_or_none(STORAGE_PRICE)
    df["ng_price"] = ng_price if ng_price is not None else ""
    df["storage_price"] = storage_price if storage_price is not None else ""

    # A simple combined index (optional):
    # combo = HDD_15D * NG_PRICE + STORAGE_PRICE
    # (你想要不同公式我再幫你換)
    if ng_price is not None or storage_price is not None:
        df["combo_index"] = (
            (df["hdd_15d"] * (ng_price if ng_price is not None else 0.0))
            + (storage_price if storage_price is not None else 0.0)
        )
    else:
        df["combo_index"] = ""

    # Save CSV
    df.to_csv(CSV_PATH, index=False)

    # Plot chart
    plot_chart(df, CHART_PATH)

    # Telegram summary
    last = df.iloc[-1].to_dict()
    send_summary(df, last, chart_path=CHART_PATH)


def send_summary(df: pd.DataFrame, last_row: dict, chart_path: str) -> None:
    d = str(last_row.get("date", "—"))
    h15 = last_row.get("hdd_15d", None)
    h30 = last_row.get("hdd_30d", None)
    d15 = last_row.get("delta_15", None)
    d30 = last_row.get("delta_30", None)
    s15 = str(last_row.get("signal_15", "—"))
    s30 = str(last_row.get("signal_30", "—"))

    ng_price = last_row.get("ng_price", "")
    storage_price = last_row.get("storage_price", "")
    combo = last_row.get("combo_index", "")

    # Cleaner, easier-to-read message (HTML)
    lines = []
    lines.append(f"<b>✅ HDD Update</b> <code>{d}</code>")
    lines.append("")
    lines.append(f"• <b>15D HDD (weighted)</b>: <b>{fmt(h15)}</b>  (Δ {fmt(d15)})  {s15}")
    lines.append(f"• <b>30D HDD (weighted)</b>: <b>{fmt(h30)}</b>  (Δ {fmt(d30)})  {s30}")

    # Optional price block
    if str(ng_price).strip() != "" or str(storage_price).strip() != "":
        lines.append("")
        lines.append("<b>Prices</b>")
        lines.append(f"• NG: <b>{ng_price if str(ng_price).strip()!='' else '—'}</b>")
        lines.append(f"• Storage: <b>{storage_price if str(storage_price).strip()!='' else '—'}</b>")
        if str(combo).strip() != "":
            try:
                combo_f = float(combo)
                lines.append(f"• <b>Combo index</b> (HDD15*NG + Storage): <b>{combo_f:.3f}</b>")
            except Exception:
                lines.append(f"• <b>Combo index</b>: <b>{combo}</b>")

    msg = "\n".join(lines)

    # Send message + chart
    tg_send_message(msg)
    if os.path.exists(chart_path):
        tg_send_photo(chart_path, caption=f"HDD Trend (15D/30D) {d}")


if __name__ == "__main__":
    run_system()
