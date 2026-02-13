import os
import os

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
EIA_API_KEY = os.getenv("EIA_API_KEY")
import json
import time
import math
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo


# =========================
# CONFIG (edit if you want)
# =========================
BASE_F = float(os.getenv("BASE_F", "65.0"))

# NYC (example). You can change to other hub lat/lon (e.g., Chicago)
LAT = float(os.getenv("LAT", "40.7128"))
LON = float(os.getenv("LON", "-74.0060"))

# For 30-day window we will build:
#   past_days = 14 (actuals) + forecast_days = 16  => 30 total
PAST_DAYS = int(os.getenv("PAST_DAYS", "14"))
FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", "16"))  # Open-Meteo forecast_days max is often 16

CSV_PATH = os.getenv("CSV_PATH", "ng_hdd_data.csv")
CHART_PATH = os.getenv("CHART_PATH", "hdd_cdd_chart.png")

# Telegram envs (GitHub secrets)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# EIA env (GitHub secrets)
EIA_API_KEY = os.getenv("EIA_API_KEY", "")

# Price ticker
# NG=F is Henry Hub Natural Gas futures on Yahoo Finance
NG_TICKER = os.getenv("NG_TICKER", "NG=F")


# =========================
# Utils
# =========================
def log(msg: str):
    print(msg, flush=True)

def retry_get(url, params=None, tries=3, timeout=25, backoff=1.5):
    last_err = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code >= 400:
                # include JSON if present
                try:
                    detail = r.json()
                except Exception:
                    detail = r.text[:300]
                raise requests.HTTPError(f"{r.status_code} {r.reason}: {detail}", response=r)
            return r
        except Exception as e:
            last_err = e
            sleep_s = (backoff ** i) + (0.3 * i)
            log(f"[WARN] HTTP attempt {i+1}/{tries} failed: {e}. sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"HTTP request failed after {tries} tries: {last_err}")


# =========================
# Weather -> temps
# We use:
#  - Archive API for past actual daily mean
#  - Forecast API for next daily mean
# =========================
def fetch_past_daily_mean_f(lat: float, lon: float, past_days: int):
    """
    Returns (dates[], tempsF[]) for past_days ending yesterday.
    """
    # open-meteo archive (no key)
    # We request daily mean temp in C then convert.
    # archive range: [today - past_days, yesterday]
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=past_days)
    end = today - timedelta(days=1)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "temperature_2m_mean",
        "timezone": "UTC",
    }
    r = retry_get(url, params=params, tries=3, timeout=25)
    data = r.json()

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    temps_c = daily.get("temperature_2m_mean", [])

    if not dates or len(dates) != len(temps_c):
        raise RuntimeError(f"Past weather data malformed: {json.dumps(data)[:500]}")

    temps_f = [c * 9/5 + 32 for c in temps_c]
    return dates, temps_f


def fetch_forecast_daily_mean_f(lat: float, lon: float, forecast_days: int):
    """
    Returns (dates[], tempsF[]) for next forecast_days starting today.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_mean",
        "temperature_unit": "fahrenheit",
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }
    r = retry_get(url, params=params, tries=3, timeout=25)
    data = r.json()

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    temps_f = daily.get("temperature_2m_mean", [])

    if not dates or len(dates) != len(temps_f):
        raise RuntimeError(f"Forecast data malformed: {json.dumps(data)[:500]}")
    return dates, temps_f


def compute_hdd_cdd_series(dates, temps_f, base_f=65.0):
    temps = np.array(temps_f, dtype=float)
    hdd = np.maximum(0.0, base_f - temps)
    cdd = np.maximum(0.0, temps - base_f)
    df = pd.DataFrame({"date": pd.to_datetime(dates), "temp_f": temps, "hdd": hdd, "cdd": cdd})
    df["date"] = df["date"].dt.date
    return df


def weighted_avg(values: np.ndarray, kind="linear"):
    """
    Weighted average where near-term gets more weight.
    kind:
      - linear: weights from 1 down to 0.2
      - exp: exponential decay
    """
    n = len(values)
    if n == 0:
        return float("nan")
    if kind == "exp":
        w = np.exp(-np.linspace(0, 2.0, n))
    else:
        w = np.linspace(1.0, 0.2, n)
    w = w / w.sum()
    return float(np.sum(values * w))


def compute_15d_30d(past_days=14, forecast_days=16):
    # build 30-day window: past actual + forecast
    past_dates, past_t = fetch_past_daily_mean_f(LAT, LON, past_days)
    fc_dates, fc_t = fetch_forecast_daily_mean_f(LAT, LON, forecast_days)

    df_past = compute_hdd_cdd_series(past_dates, past_t, BASE_F)
    df_fc = compute_hdd_cdd_series(fc_dates, fc_t, BASE_F)

    df = pd.concat([df_past, df_fc], ignore_index=True)
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)

    # ensure we have at least 30 rows ideally
    # (sometimes archive missing a day; still works)
    last_30 = df.tail(30).copy()
    last_15 = df.tail(15).copy()

    hdd15 = weighted_avg(last_15["hdd"].to_numpy(), kind="linear")
    hdd30 = weighted_avg(last_30["hdd"].to_numpy(), kind="linear")
    cdd15 = weighted_avg(last_15["cdd"].to_numpy(), kind="linear")
    cdd30 = weighted_avg(last_30["cdd"].to_numpy(), kind="linear")

    return last_30, hdd15, hdd30, cdd15, cdd30


# =========================
# EIA Storage (WNGSR)
# We fetch latest weekly storage.
# If holiday shifts, "latest available period" still works.
# =========================
def fetch_eia_storage_latest():
    """
    Returns dict:
      {
        "week": "YYYY-MM-DD",
        "total_bcf": float,
        "net_change_bcf": float or None,
      }
    If EIA_API_KEY missing -> return None
    """
    if not EIA_API_KEY:
        return None

    # EIA v2 endpoint for weekly working gas storage (Lower 48)
    # dataset: natural-gas/stor/wkly
    url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
    params = {
        "api_key": EIA_API_KEY,
        "frequency": "weekly",
        "data[0]": "value",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 2,  # get 2 rows for net change calc
    }

    r = retry_get(url, params=params, tries=3, timeout=25)
    data = r.json()

    # expect data["response"]["data"] list
    rows = (((data or {}).get("response") or {}).get("data")) or []
    if not rows:
        raise RuntimeError(f"EIA returned no data: {json.dumps(data)[:500]}")

    # "period" should be YYYY-MM-DD, "value" is Bcf
    latest = rows[0]
    week = latest.get("period")
    total = latest.get("value")

    net_change = None
    if len(rows) >= 2 and rows[1].get("value") is not None and total is not None:
        try:
            net_change = float(total) - float(rows[1]["value"])
        except Exception:
            net_change = None

    return {
        "week": str(week) if week else None,
        "total_bcf": float(total) if total is not None else None,
        "net_change_bcf": float(net_change) if net_change is not None else None,
    }


# =========================
# Price (NG=F)
# We'll try yfinance; if not available, fallback to Stooq.
# =========================
def fetch_ng_price():
    # 1) yfinance
    try:
        import yfinance as yf
        t = yf.Ticker(NG_TICKER)
        hist = t.history(period="10d", interval="1d")
        if hist is not None and len(hist) > 0:
            close = float(hist["Close"].iloc[-1])
            # simple 5-day MA
            ma5 = float(hist["Close"].tail(5).mean()) if len(hist) >= 5 else float("nan")
            above_5ma = (close > ma5) if not math.isnan(ma5) else None

            # 3-day breakout check
            last3 = hist["Close"].tail(3)
            prev3 = hist["Close"].iloc[-4:-1] if len(hist) >= 4 else None
            break_high = None
            break_low = None
            if prev3 is not None and len(prev3) == 3:
                break_high = close > float(prev3.max())
                break_low = close < float(prev3.min())

            return {
                "close": close,
                "ma5": ma5 if not math.isnan(ma5) else None,
                "above_5ma": above_5ma,
                "break_3d_high": break_high,
                "break_3d_low": break_low,
            }
    except Exception as e:
        log(f"[WARN] yfinance failed: {e}")

    # 2) fallback: stooq (not perfect but keeps pipeline alive)
    try:
        # stooq uses different symbols; futures not always available.
        # We'll just skip if not found.
        return None
    except Exception:
        return None


# =========================
# Telegram
# =========================
def telegram_send_message(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log("[WARN] Telegram token/chat_id not set; skip sending message.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    retry_get(url, params=payload, tries=3, timeout=25)


def telegram_send_photo(photo_path: str, caption: str = ""):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log("[WARN] Telegram token/chat_id not set; skip sending photo.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(photo_path, "rb") as f:
        files = {"photo": f}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
        r = requests.post(url, data=data, files=files, timeout=35)
        if r.status_code >= 400:
            raise RuntimeError(f"Telegram sendPhoto failed: {r.status_code} {r.text[:300]}")


# =========================
# Chart
# =========================
def build_chart(df_30: pd.DataFrame, run_label: str, out_path: str):
    # df_30 columns: date, temp_f, hdd, cdd
    x = pd.to_datetime(df_30["date"])
    plt.figure(figsize=(11, 4.8))
    plt.plot(x, df_30["hdd"].to_numpy(), label=f"Daily HDD (base {BASE_F:.0f}F)")
    plt.plot(x, df_30["cdd"].to_numpy(), label=f"Daily CDD (base {BASE_F:.0f}F)")
    plt.title(f"HDD/CDD Trend • {run_label}")
    plt.xlabel("Day (UTC)")
    plt.ylabel("Degree Days")
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# =========================
# CSV history
# =========================
def load_history(csv_path: str):
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        log(f"[WARN] CSV read failed ({e}), starting fresh.")
        return pd.DataFrame()


def append_history(csv_path: str, row: dict):
    df = load_history(csv_path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv_path, index=False)


def fmt_arrow(delta: float):
    if delta is None or (isinstance(delta, float) and math.isnan(delta)):
        return "—"
    if abs(delta) < 1e-9:
        return "⏺"
    return "⬆️" if delta > 0 else "⬇️"


# =========================
# Main signal logic
# =========================
def compute_signal(hdd15_delta, hdd30_delta, price_info):
    """
    Simple logic:
      - If both HDD 15/30 up: bullish weather
      - If both down: bearish weather
      - Otherwise: mixed -> WAIT
    Add price confirmation if available (above 5MA).
    """
    if hdd15_delta is None or hdd30_delta is None:
        return ("WAIT", "No delta history yet")

    up15 = hdd15_delta > 0
    up30 = hdd30_delta > 0
    down15 = hdd15_delta < 0
    down30 = hdd30_delta < 0

    if up15 and up30:
        base = "🔥 Bullish (colder revision)"
    elif down15 and down30:
        base = "🧊 Bearish (warmer revision)"
    else:
        base = "🎯 WAIT"
    conf = []

    if price_info and price_info.get("above_5ma") is not None:
        conf.append(f"Price above 5MA: {'YES' if price_info['above_5ma'] else 'NO'}")
    else:
        conf.append("Price confirmation: NA")

    if base.startswith("🎯"):
        conf.insert(0, "Weather mixed (15D/30D disagree)")

    return (base, " / ".join(conf))


def run():
    run_utc = datetime.now(timezone.utc)
    run_label = run_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    # 1) HDD/CDD
    df_30, hdd15, hdd30, cdd15, cdd30 = compute_15d_30d(PAST_DAYS, FORECAST_DAYS)

    # 2) History deltas (vs previous run)
    hist = load_history(CSV_PATH)
    prev = hist.iloc[-1] if len(hist) > 0 else None

    def get_prev_float(col):
        if prev is None:
            return None
        try:
            return float(prev[col])
        except Exception:
            return None

    prev_hdd15 = get_prev_float("hdd15_wtd")
    prev_hdd30 = get_prev_float("hdd30_wtd")
    prev_cdd15 = get_prev_float("cdd15_wtd")
    prev_cdd30 = get_prev_float("cdd30_wtd")

    hdd15_delta = (hdd15 - prev_hdd15) if (prev_hdd15 is not None) else None
    hdd30_delta = (hdd30 - prev_hdd30) if (prev_hdd30 is not None) else None
    cdd15_delta = (cdd15 - prev_cdd15) if (prev_cdd15 is not None) else None
    cdd30_delta = (cdd30 - prev_cdd30) if (prev_cdd30 is not None) else None

    # 3) Storage (EIA)
    storage_note = ""
    storage = None
    if EIA_API_KEY:
        try:
            storage = fetch_eia_storage_latest()
        except Exception as e:
            storage = None
            storage_note = f"Storage fetch failed: {e}"
    else:
        storage_note = "EIA_API_KEY not set (storage skipped)"

    # 4) Price
    price_info = fetch_ng_price()

    # 5) Chart
    build_chart(df_30, run_label, CHART_PATH)

    # 6) Compose Telegram text (easy to read)
    def line(name, val, delta):
        d = "NA" if delta is None else f"{fmt_arrow(delta)} Δ {delta:+.2f}"
        return f"• {name}: <b>{val:.2f}</b>  ({d})"

    msg_lines = []
    msg_lines.append(f"📌 <b>HDD/CDD Update</b> ({run_utc.strftime('%Y-%m-%d')})")
    msg_lines.append("")
    msg_lines.append(f"🔥 <b>HDD</b> (base {BASE_F:.0f}F)")
    msg_lines.append(line("15D Wtd", hdd15, hdd15_delta))
    msg_lines.append(line("30D Wtd", hdd30, hdd30_delta))
    msg_lines.append("")
    msg_lines.append(f"🌤️ <b>CDD</b> (base {BASE_F:.0f}F)")
    msg_lines.append(line("15D Wtd", cdd15, cdd15_delta))
    msg_lines.append(line("30D Wtd", cdd30, cdd30_delta))
    msg_lines.append("")

    # Storage section
    msg_lines.append("🧱 <b>Storage (EIA WNGSR)</b>")
    if storage and storage.get("week") and storage.get("total_bcf") is not None:
        msg_lines.append(f"• Week: <b>{storage['week']}</b>")
        msg_lines.append(f"• Total: <b>{storage['total_bcf']:.0f}</b> bcf")
        if storage.get("net_change_bcf") is not None:
            msg_lines.append(f"• Net chg: <b>{storage['net_change_bcf']:+.0f}</b> bcf")
        else:
            msg_lines.append("• Net chg: NA")
    else:
        msg_lines.append("• Storage: <b>NA</b>")
        if storage_note:
            msg_lines.append(f"• Note: {storage_note}")
    msg_lines.append("")

    # Price section
    msg_lines.append("📈 <b>Price</b> (NG=F)")
    if price_info and price_info.get("close") is not None:
        msg_lines.append(f"• Close: <b>{price_info['close']:.3f}</b>")
        msg_lines.append(f"• Above 5MA: <b>{'YES' if price_info.get('above_5ma') else 'NO'}</b>" if price_info.get("above_5ma") is not None else "• Above 5MA: NA")
        bh = price_info.get("break_3d_high")
        bl = price_info.get("break_3d_low")
        if bh is not None and bl is not None:
            msg_lines.append(f"• Break 3D High: <b>{'YES' if bh else 'NO'}</b> / Low: <b>{'YES' if bl else 'NO'}</b>")
        else:
            msg_lines.append("• Break 3D High/Low: NA")
    else:
        msg_lines.append("• Close: NA (price fetch failed)")
    msg_lines.append("")

    # Signal
    signal, confidence = compute_signal(hdd15_delta, hdd30_delta, price_info)
    msg_lines.append(f"🎯 <b>Signal:</b> {signal}")
    msg_lines.append(f"🧠 <b>Confidence:</b> {confidence}")
    msg_lines.append(f"🕒 <b>Updated:</b> {run_label}")

    msg = "\n".join(msg_lines)

    # 7) Save row
    row = {
        "run_utc": run_label,
        "hdd15_wtd": hdd15,
        "hdd30_wtd": hdd30,
        "cdd15_wtd": cdd15,
        "cdd30_wtd": cdd30,
        "storage_week": (storage.get("week") if storage else None),
        "storage_total_bcf": (storage.get("total_bcf") if storage else None),
        "storage_net_change_bcf": (storage.get("net_change_bcf") if storage else None),
        "ng_close": (price_info.get("close") if price_info else None),
    }
    append_history(CSV_PATH, row)

    # 8) Telegram push
    telegram_send_message(msg)
    telegram_send_photo(CHART_PATH, caption=f"📉 HDD/CDD Trend • {run_utc.strftime('%Y-%m-%d')} UTC")

    log("[OK] Done.")


if __name__ == "__main__":
    run()
