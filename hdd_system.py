import os
import io
import math
import time
import json
import datetime as dt
from dataclasses import dataclass

import requests
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# =========================
# Config (env)
# =========================
LAT = float(os.getenv("LAT", "40.7128"))        # NYC default
LON = float(os.getenv("LON", "-74.0060"))
BASE_F = float(os.getenv("BASE_F", "65.0"))     # degree-day base
# We want 30 days total for HDD/CDD:
# Open-Meteo forecast_days allowed 0..16. We'll do: past_days=14 + forecast_days=16 => 30.
PAST_DAYS = int(os.getenv("PAST_DAYS", "14"))
FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", "16"))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

EIA_API_KEY = os.getenv("EIA_API_KEY")  # <-- 你加在 secrets 後，要靠 workflow env 傳進來
# EIA series id (weekly storage, Lower 48). If you already know your own series id, set env EIA_SERIES_ID.
EIA_SERIES_ID = os.getenv("EIA_SERIES_ID", "NG.NW2_EPG0_SWO_R48_BCF.W")  # common weekly storage series
NG_TICKER = os.getenv("NG_TICKER", "NG=F")

DATA_CSV = os.getenv("DATA_CSV", "ng_hdd_data.csv")
CHART_PNG = os.getenv("CHART_PNG", "hdd_cdd_chart.png")
TIMEZONE = "UTC"  # keep everything in UTC to avoid DST confusion


# =========================
# Helpers
# =========================
def utcnow():
    return dt.datetime.now(dt.timezone.utc)

def iso_utc(ts: dt.datetime) -> str:
    return ts.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def sign_emoji(x: float) -> str:
    if x is None:
        return "—"
    if x > 0:
        return "⬆️"
    if x < 0:
        return "⬇️"
    return "—"

def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"

def safe_getenv(name: str):
    v = os.getenv(name)
    return v if v and v.strip() else None

def retry_get(url, params=None, headers=None, tries=3, timeout=25):
    last_err = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            # keep body for error debugging
            if r.status_code >= 400:
                raise requests.HTTPError(f"{r.status_code} {r.reason}: {r.text}", response=r)
            return r
        except Exception as e:
            last_err = e
            sleep_s = 1.0 + i * 0.8
            print(f"[WARN] HTTP attempt {i+1}/{tries} failed: {e}. sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"HTTP request failed after {tries} tries: {last_err}")


# =========================
# Weather / HDD / CDD
# =========================
def fetch_daily_mean_f(lat: float, lon: float, past_days: int, forecast_days: int):
    """
    Open-Meteo forecast API:
    - forecast_days allowed 0..16
    - past_days can be used to include recent history
    We request daily temperature_2m_mean in Fahrenheit, timezone UTC.
    """
    if forecast_days < 0 or forecast_days > 16:
        raise ValueError("FORECAST_DAYS must be 0..16 for Open-Meteo forecast API.")
    if past_days < 0 or past_days > 92:
        # Open-Meteo supports up to 92 in many cases; keep safe.
        raise ValueError("PAST_DAYS should be 0..92")

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_mean",
        "temperature_unit": "fahrenheit",
        "timezone": TIMEZONE,
        "past_days": past_days,
        "forecast_days": forecast_days,
    }
    r = retry_get(url, params=params, tries=3, timeout=25)
    js = r.json()

    dates = js.get("daily", {}).get("time", [])
    temps = js.get("daily", {}).get("temperature_2m_mean", [])
    if not dates or not temps or len(dates) != len(temps):
        raise RuntimeError(f"Unexpected weather payload: {json.dumps(js)[:300]}")

    # Convert to list of (date_str, temp_f)
    return dates, [float(x) for x in temps]

def compute_hdd_cdd(temps_f, base_f=65.0):
    hdd = [max(0.0, base_f - t) for t in temps_f]
    cdd = [max(0.0, t - base_f) for t in temps_f]
    return hdd, cdd

def weighted_sum_last_n(values, n):
    """
    Simple linear weighting: latest day weight = n, previous = n-1, ... oldest=1
    """
    if len(values) < n:
        return None
    tail = values[-n:]
    weights = list(range(1, n + 1))
    # align oldest->1, newest->n
    s = sum(v * w for v, w in zip(tail, weights))
    denom = sum(weights)
    return s / denom

def build_hdd_cdd_metrics(dates, temps_f, base_f):
    hdd, cdd = compute_hdd_cdd(temps_f, base_f=base_f)
    out = {
        "dates": dates,
        "temps_f": temps_f,
        "hdd": hdd,
        "cdd": cdd,
        "hdd15": weighted_sum_last_n(hdd, 15),
        "hdd30": weighted_sum_last_n(hdd, 30),
        "cdd15": weighted_sum_last_n(cdd, 15),
        "cdd30": weighted_sum_last_n(cdd, 30),
    }
    return out


# =========================
# EIA Storage (weekly)
# =========================
@dataclass
class StorageData:
    week_ending: str  # YYYY-MM-DD
    total_bcf: float
    net_change_bcf: float | None
    is_new: bool

def fetch_eia_storage(eia_api_key: str, series_id: str) -> StorageData | None:
    """
    Uses legacy EIA v1 series endpoint (simple and stable).
    Returns latest weekly point and delta vs previous week.
    """
    if not eia_api_key:
        return None

    url = "https://api.eia.gov/series/"
    params = {"api_key": eia_api_key, "series_id": series_id}
    r = retry_get(url, params=params, tries=3, timeout=25)
    js = r.json()

    series = js.get("series")
    if not series:
        raise RuntimeError(f"EIA response missing series. Payload: {json.dumps(js)[:300]}")

    data = series[0].get("data", [])
    # data format: [[YYYYMMDD, value], ...] (often most recent first)
    if not data or len(data) < 1:
        raise RuntimeError("EIA series data empty.")

    # ensure sorted by date ascending for delta
    def parse_key(k):
        # weekly: YYYYMMDD
        s = str(k)
        return dt.datetime.strptime(s, "%Y%m%d").date()

    # data might be most recent first; normalize
    norm = [(parse_key(k), float(v)) for k, v in data if k and v is not None]
    norm.sort(key=lambda x: x[0])

    last_date, last_val = norm[-1]
    prev_val = norm[-2][1] if len(norm) >= 2 else None
    net = (last_val - prev_val) if prev_val is not None else None

    return StorageData(
        week_ending=last_date.isoformat(),
        total_bcf=last_val,
        net_change_bcf=net,
        is_new=True,  # we'll decide "new" by comparing to last stored CSV later
    )


# =========================
# Price / Simple signals
# =========================
def fetch_ng_price(ticker="NG=F"):
    """
    Use yfinance daily close. We'll pull last ~10 days.
    """
    try:
        df = yf.download(ticker, period="14d", interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None, None, None, None
        # yfinance returns columns like ['Open','High','Low','Close',...]
        close = float(df["Close"].dropna().iloc[-1])
        sma5 = float(df["Close"].dropna().rolling(5).mean().iloc[-1]) if len(df["Close"].dropna()) >= 5 else None

        # 3-day breakout check
        last3 = df["High"].dropna().tail(3)
        low3 = df["Low"].dropna().tail(3)
        break_high = bool(close > float(last3.max())) if len(last3) == 3 else False
        break_low = bool(close < float(low3.min())) if len(low3) == 3 else False

        return close, sma5, break_high, break_low
    except Exception as e:
        print(f"[WARN] Price fetch failed: {e}")
        return None, None, None, None


def decide_signal(delta_hdd15, delta_hdd30, delta_cdd15, delta_cdd30, price_above_sma5):
    """
    Very simple logic (你之後可以再調):
    - HDD up => bullish gas (colder)
    - HDD down => bearish gas (warmer)
    - CDD up can be bullish in summer (more cooling demand)
    - Mixed => WAIT/Neutral
    """
    score = 0
    # HDD has bigger weight in winter
    if delta_hdd15 is not None:
        score += 2 if delta_hdd15 > 0 else (-2 if delta_hdd15 < 0 else 0)
    if delta_hdd30 is not None:
        score += 1 if delta_hdd30 > 0 else (-1 if delta_hdd30 < 0 else 0)

    # CDD smaller weight (depends season)
    if delta_cdd15 is not None:
        score += 1 if delta_cdd15 > 0 else (-1 if delta_cdd15 < 0 else 0)
    if delta_cdd30 is not None:
        score += 1 if delta_cdd30 > 0 else (-1 if delta_cdd30 < 0 else 0)

    # price filter
    if price_above_sma5 is True:
        score += 1
    elif price_above_sma5 is False:
        score -= 1

    if score >= 3:
        return "🔥 BULLISH", "Weather/price supportive"
    if score <= -3:
        return "🧊 BEARISH", "Warmer revision / price weak"
    if -2 <= score <= 2:
        return "🎯 WAIT", "Mixed signals (weather 15D/30D or price mixed)"
    return "😐 Neutral", "Unclear"


# =========================
# CSV / History
# =========================
def load_history(csv_path):
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    return pd.read_csv(csv_path)

def append_history(csv_path, row: dict):
    df = load_history(csv_path)
    new_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    new_df.to_csv(csv_path, index=False)
    return new_df

def last_row(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    return df.iloc[-1].to_dict()


# =========================
# Chart
# =========================
def plot_trend_png(dates, hdd, cdd, out_path, title):
    # dates are YYYY-MM-DD strings
    x = [dt.datetime.strptime(d, "%Y-%m-%d").date() for d in dates]

    plt.figure(figsize=(12, 5))
    plt.plot(x, hdd, label="Daily HDD (base 65F)")
    plt.plot(x, cdd, label="Daily CDD (base 65F)")
    plt.title(title)
    plt.xlabel("Day (UTC)")
    plt.ylabel("Degree Days")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# =========================
# Telegram
# =========================
def tg_send_message(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    r = retry_get(url, params=payload, tries=3, timeout=25)
    return r.json()

def tg_send_photo(token, chat_id, photo_path, caption=None):
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    with open(photo_path, "rb") as f:
        files = {"photo": f}
        data = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption
        r = requests.post(url, data=data, files=files, timeout=40)
        if r.status_code >= 400:
            raise RuntimeError(f"Telegram photo failed: {r.status_code} {r.text}")
        return r.json()


# =========================
# Main
# =========================
def run():
    now = utcnow()

    # 1) Weather
    dates, temps = fetch_daily_mean_f(LAT, LON, PAST_DAYS, FORECAST_DAYS)  # 30 points total
    metrics = build_hdd_cdd_metrics(dates, temps, BASE_F)

    hdd15 = metrics["hdd15"]
    hdd30 = metrics["hdd30"]
    cdd15 = metrics["cdd15"]
    cdd30 = metrics["cdd30"]

    # 2) Price
    close, sma5, break_high, break_low = fetch_ng_price(NG_TICKER)
    price_above_sma5 = None
    if close is not None and sma5 is not None:
        price_above_sma5 = (close > sma5)

    # 3) Load history and compute deltas
    hist = load_history(DATA_CSV)
    prev = last_row(hist)

    def get_prev_float(key):
        if not prev:
            return None
        v = prev.get(key)
        try:
            return float(v)
        except Exception:
            return None

    prev_hdd15 = get_prev_float("hdd15_wtd")
    prev_hdd30 = get_prev_float("hdd30_wtd")
    prev_cdd15 = get_prev_float("cdd15_wtd")
    prev_cdd30 = get_prev_float("cdd30_wtd")

    d_hdd15 = (hdd15 - prev_hdd15) if (hdd15 is not None and prev_hdd15 is not None) else 0.0 if prev else 0.0
    d_hdd30 = (hdd30 - prev_hdd30) if (hdd30 is not None and prev_hdd30 is not None) else 0.0 if prev else 0.0
    d_cdd15 = (cdd15 - prev_cdd15) if (cdd15 is not None and prev_cdd15 is not None) else 0.0 if prev else 0.0
    d_cdd30 = (cdd30 - prev_cdd30) if (cdd30 is not None and prev_cdd30 is not None) else 0.0 if prev else 0.0

    # 4) Storage (EIA)
    storage_note = None
    storage = None
    if EIA_API_KEY:
        try:
            storage = fetch_eia_storage(EIA_API_KEY, EIA_SERIES_ID)
            # mark NEW only if week differs from last stored
            prev_week = prev.get("storage_week") if prev else None
            if prev_week and storage and storage.week_ending == str(prev_week):
                storage = StorageData(storage.week_ending, storage.total_bcf, storage.net_change_bcf, is_new=False)
        except Exception as e:
            storage_note = f"Storage fetch failed: {e}"
            storage = None
    else:
        storage_note = "EIA_API_KEY not set (storage skipped)"

    # 5) Signal
    signal, confidence = decide_signal(d_hdd15, d_hdd30, d_cdd15, d_cdd30, price_above_sma5)

    # 6) Save chart
    title = f"HDD/CDD Trend · {now.strftime('%Y-%m-%d')} {TIMEZONE}"
    plot_trend_png(metrics["dates"], metrics["hdd"], metrics["cdd"], CHART_PNG, title=title)

    # 7) Append history
    row = {
        "run_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
        "hdd15_wtd": hdd15,
        "hdd30_wtd": hdd30,
        "cdd15_wtd": cdd15,
        "cdd30_wtd": cdd30,
        "storage_week": storage.week_ending if storage else "",
        "storage_total_bcf": storage.total_bcf if storage else "",
        "storage_net_bcf": storage.net_change_bcf if storage else "",
        "ng_close": close if close is not None else "",
        "signal": signal,
    }
    hist2 = append_history(DATA_CSV, row)

    # 8) Compose Telegram message (更好讀)
    # Weather section
    msg = []
    msg.append(f"📌 <b>HDD/CDD Update</b> ({now.strftime('%Y-%m-%d')})")
    msg.append("")
    msg.append(f"🔥 <b>HDD</b> (base {int(BASE_F)}F)")
    msg.append(f"• 15D Wtd: {fmt(hdd15)}  ({sign_emoji(d_hdd15)} Δ {fmt(d_hdd15)})")
    msg.append(f"• 30D Wtd: {fmt(hdd30)}  ({sign_emoji(d_hdd30)} Δ {fmt(d_hdd30)})")
    msg.append("")
    msg.append(f"🌤️ <b>CDD</b> (base {int(BASE_F)}F)")
    msg.append(f"• 15D Wtd: {fmt(cdd15)}  ({sign_emoji(d_cdd15)} Δ {fmt(d_cdd15)})")
    msg.append(f"• 30D Wtd: {fmt(cdd30)}  ({sign_emoji(d_cdd30)} Δ {fmt(d_cdd30)})")
    msg.append("")

    # Storage section
    msg.append("🧱 <b>Storage</b> (EIA)")
    if storage:
        new_tag = "🟢 NEW" if storage.is_new else "🟡 no change (maybe holiday/unchanged)"
        msg.append(f"• Week: {storage.week_ending}  {new_tag}")
        msg.append(f"• Total: {fmt(storage.total_bcf, 0)} bcf")
        msg.append(f"• Net chg: {fmt(storage.net_change_bcf, 0)} bcf")
    else:
        msg.append("• Storage: NA")
        if storage_note:
            msg.append(f"• Note: {storage_note}")
    msg.append("")

    # Price section
    msg.append(f"📈 <b>Price</b> ({NG_TICKER})")
    msg.append(f"• Close: {fmt(close, 3)}")
    if sma5 is not None and close is not None:
        msg.append(f"• Above 5MA: {'YES' if close > sma5 else 'NO'}")
    msg.append(f"• Break 3D High: {'YES' if break_high else 'NO'} / Low: {'YES' if break_low else 'NO'}")
    msg.append("")

    # Signal
    msg.append(f"🎯 <b>Signal</b>: {signal}")
    msg.append(f"🧠 Confidence: {confidence}")
    msg.append(f"🕒 Updated: {iso_utc(now)}")

    text = "\n".join(msg)

    # 9) Send Telegram
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        tg_send_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, text)
        tg_send_photo(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, CHART_PNG, caption=f"📉 Trend (HDD/CDD) · {now.strftime('%Y-%m-%d')} {TIMEZONE}")
    else:
        print("[WARN] Telegram token/chat_id not set; skipping send.")
        print(text)


if __name__ == "__main__":
    run()
