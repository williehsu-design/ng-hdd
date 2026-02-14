import os
import json
import time
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
LAT = 40.7128
LON = -74.0060
BASE_F = 65.0

CSV_PATH = "ng_hdd_data.csv"
CHART_PATH = "hdd_cdd_chart.png"  # align with workflow git add

PAST_DAYS = 14
FORECAST_DAYS = 16  # 14 + 16 + today ≈ 31 points

# Price sources
PRICE_SYMBOL_PRIMARY = "NG=F"      # Yahoo (often blocked on Actions)
PRICE_SYMBOL_FALLBACK = "UNG"      # Yahoo fallback
PRICE_FRED_SERIES = "DHHNGSP"      # ✅ FRED Henry Hub spot (no key)
PRICE_STOOQ_FALLBACK = "ng.f"      # last-resort

# =========================
# ENV (GitHub Secrets)
# =========================
TG_BOT_TOKEN = (os.getenv("TG_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or "").strip()
TG_CHAT_ID = (os.getenv("TG_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID") or "").strip()
EIA_API_KEY = (os.getenv("EIA_API_KEY") or "").strip()

# =========================
# HELPERS
# =========================
def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def fmt_utc(ts: dt.datetime) -> str:
    return ts.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def retry_get(url: str, params: dict, tries: int = 3, timeout: int = 20) -> requests.Response:
    last_err = None
    headers = {"User-Agent": "Mozilla/5.0 (GitHubActions; HDDCDDMonitor/1.0)"}
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=headers)
            if r.status_code >= 400:
                raise requests.HTTPError(f"{r.status_code} {r.reason}: {r.text[:300]}", response=r)
            return r
        except Exception as e:
            last_err = e
            sleep_s = 1.0 + i * 0.7
            print(f"[WARN] HTTP attempt {i+1}/{tries} failed: {e}. sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"HTTP request failed after {tries} tries: {last_err}")

def fmt_arrow(delta: float) -> str:
    if delta > 0.001:
        return "⬆️"
    if delta < -0.001:
        return "⬇️"
    return "➖"

# =========================
# WEATHER / HDD / CDD
# =========================
def fetch_daily_mean_f(lat: float, lon: float, past_days: int, forecast_days: int) -> Tuple[List[str], List[float]]:
    """
    Open-Meteo daily temps (F). Robust against nulls:
    mean -> (max+min)/2 -> ffill/bfill.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min",
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
        "past_days": past_days,
        "forecast_days": forecast_days,
    }
    r = retry_get(url, params=params, tries=3, timeout=20)
    data = r.json()
    daily = data.get("daily", {})

    dates = daily.get("time", []) or []
    tmean = daily.get("temperature_2m_mean", []) or []
    tmax = daily.get("temperature_2m_max", []) or []
    tmin = daily.get("temperature_2m_min", []) or []

    if not dates:
        raise RuntimeError(f"Open-Meteo invalid payload: {json.dumps(daily)[:400]}")

    n = len(dates)

    def pad(arr):
        arr = list(arr)
        if len(arr) < n:
            arr += [None] * (n - len(arr))
        return arr[:n]

    tmean = pad(tmean)
    tmax = pad(tmax)
    tmin = pad(tmin)

    temps: List[Optional[float]] = []
    for i in range(n):
        v = tmean[i]
        if v is None:
            mx, mn = tmax[i], tmin[i]
            if mx is not None and mn is not None:
                v = (float(mx) + float(mn)) / 2.0
        temps.append(float(v) if v is not None else None)

    # forward fill
    last_valid = None
    for i in range(n):
        if temps[i] is None and last_valid is not None:
            temps[i] = last_valid
        elif temps[i] is not None:
            last_valid = temps[i]

    # backward fill
    next_valid = None
    for i in range(n - 1, -1, -1):
        if temps[i] is not None:
            next_valid = temps[i]
        elif temps[i] is None and next_valid is not None:
            temps[i] = next_valid

    if any(v is None for v in temps):
        raise RuntimeError(f"Open-Meteo temps still contain nulls: {temps}")

    return dates, [float(x) for x in temps]

def compute_hdd_cdd_series(dates: List[str], temps_f: List[float], base_f: float) -> pd.DataFrame:
    df = pd.DataFrame({"date": pd.to_datetime(dates), "tmean_f": temps_f})
    df = df.sort_values("date").reset_index(drop=True)
    df["hdd"] = np.maximum(0.0, base_f - df["tmean_f"])
    df["cdd"] = np.maximum(0.0, df["tmean_f"] - base_f)
    return df

def weighted_avg_recent(values: np.ndarray) -> float:
    n = len(values)
    if n == 0:
        return float("nan")
    w = np.linspace(0.5, 1.0, n)
    return float(np.sum(values * w) / np.sum(w))

def compute_15_30(df: pd.DataFrame) -> Dict[str, float]:
    last15 = df.tail(15).sort_values("date")
    last30 = df.tail(30).sort_values("date")
    if len(last30) < 30:
        raise RuntimeError(f"Not enough days for 30D metrics (need 30, got {len(last30)})")
    return {
        "hdd_15d": weighted_avg_recent(last15["hdd"].to_numpy()),
        "hdd_30d": weighted_avg_recent(last30["hdd"].to_numpy()),
        "cdd_15d": weighted_avg_recent(last15["cdd"].to_numpy()),
        "cdd_30d": weighted_avg_recent(last30["cdd"].to_numpy()),
    }

# =========================
# STORAGE (EIA v2)
# =========================
@dataclass
class StorageInfo:
    week: Optional[str] = None
    total_bcf: Optional[float] = None
    bias: str = "NA"
    note: str = ""

def fetch_storage_eia(api_key: str) -> StorageInfo:
    if not api_key:
        return StorageInfo(note="EIA_API_KEY not set (storage skipped)")

    url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
    params = {
        "api_key": api_key,
        "frequency": "weekly",
        "data[0]": "value",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 1,
    }

    try:
        r = retry_get(url, params=params, tries=3, timeout=20)
        j = r.json()
        rows = j.get("response", {}).get("data", [])
        if not rows:
            return StorageInfo(note="EIA response empty")
        row = rows[0]
        week = str(row.get("period", "")) or None
        total = row.get("value", None)
        return StorageInfo(
            week=week,
            total_bcf=float(total) if total is not None else None,
            bias="NA",
            note="EIA ok",
        )
    except Exception as e:
        return StorageInfo(note=f"Storage fetch failed: {e}")

# =========================
# PRICE
# =========================
@dataclass
class PriceInfo:
    symbol: str
    price: Optional[float] = None
    ma5: Optional[float] = None
    break3_high: Optional[bool] = None
    break3_low: Optional[bool] = None
    note: str = ""

def _calc_price_signals(close: pd.Series, symbol: str, note: str) -> PriceInfo:
    close = close.dropna()
    if len(close) < 6:
        return PriceInfo(symbol=symbol, note=f"{note}: not enough bars")

    last = float(close.iloc[-1])
    ma5 = float(close.tail(5).mean())
    prior3 = close.iloc[-4:-1]
    break3_high = bool(last > float(prior3.max()))
    break3_low = bool(last < float(prior3.min()))
    return PriceInfo(symbol=symbol, price=last, ma5=ma5, break3_high=break3_high, break3_low=break3_low, note="ok")

def fetch_price_yfinance(symbol: str) -> PriceInfo:
    try:
        import yfinance as yf
    except Exception as e:
        return PriceInfo(symbol=symbol, note=f"yfinance not available: {e}")

    try:
        df = yf.download(symbol, period="1mo", interval="1d", progress=False, threads=False)
        if df is None or df.empty or "Close" not in df.columns:
            return PriceInfo(symbol=symbol, note=f"{symbol}: empty")
        return _calc_price_signals(df["Close"], symbol, "yfinance")
    except Exception as e:
        return PriceInfo(symbol=symbol, note=f"{symbol}: failed: {e}")

def fetch_price_fred(series_id: str = "DHHNGSP") -> PriceInfo:
    """
    ✅ FRED CSV (no api key):
    https://fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP
    """
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
        params = {"id": series_id}
        r = retry_get(url, params=params, tries=3, timeout=20)

        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        if df is None or df.empty or df.shape[1] < 2:
            return PriceInfo(symbol=f"FRED:{series_id}", note="fred: empty")

        # columns: DATE, DHHNGSP
        col = df.columns[1]
        close = pd.to_numeric(df[col], errors="coerce")
        return _calc_price_signals(close, f"FRED:{series_id}", "fred")
    except Exception as e:
        return PriceInfo(symbol=f"FRED:{series_id}", note=f"fred failed: {e}")

def fetch_price_stooq(symbol_stooq: str) -> PriceInfo:
    try:
        url = "https://stooq.com/q/d/l/"
        params = {"s": symbol_stooq, "i": "d"}
        r = retry_get(url, params=params, tries=3, timeout=20)
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        if df is None or df.empty or "Close" not in df.columns:
            return PriceInfo(symbol=symbol_stooq, note="stooq: empty/not-csv")
        return _calc_price_signals(df["Close"], symbol_stooq, "stooq")
    except Exception as e:
        return PriceInfo(symbol=symbol_stooq, note=f"stooq failed: {e}")

def fetch_price_with_fallback() -> PriceInfo:
    # 1) Yahoo primary
    p = fetch_price_yfinance(PRICE_SYMBOL_PRIMARY)
    if p.note == "ok":
        return p

    # 2) Yahoo fallback
    p2 = fetch_price_yfinance(PRICE_SYMBOL_FALLBACK)
    if p2.note == "ok":
        return p2

    # 3) ✅ FRED (very stable)
    pf = fetch_price_fred(PRICE_FRED_SERIES)
    if pf.note == "ok":
        return pf

    # 4) last-resort Stooq
    ps = fetch_price_stooq(PRICE_STOOQ_FALLBACK)
    if ps.note == "ok":
        return ps

    return PriceInfo(
        symbol=PRICE_SYMBOL_PRIMARY,
        note=f"primary failed ({p.note}); fallback failed ({p2.note}); fred failed ({pf.note}); stooq failed ({ps.note})"
    )

# =========================
# SIGNAL LOGIC
# =========================
def decide_trade_2_5d_boilkold(
    d_hdd15: float,
    d_hdd30: float,
    storage: StorageInfo,
    price: PriceInfo,
) -> Tuple[str, str]:
    if d_hdd15 > 0 and d_hdd30 > 0:
        weather_dir = +1
        accel = (d_hdd15 > d_hdd30)
    elif d_hdd15 < 0 and d_hdd30 < 0:
        weather_dir = -1
        accel = (d_hdd15 < d_hdd30)
    else:
        return ("WAIT", "Weather mixed (15D/30D disagree)")

    if price.price is None or price.ma5 is None or price.break3_high is None or price.break3_low is None:
        return ("WAIT", f"No price confirmation ({price.note})")

    price_ok_long = (price.price > price.ma5) and bool(price.break3_high)
    price_ok_short = (price.price < price.ma5) and bool(price.break3_low)

    stor_dir = 0
    if "bull" in (storage.bias or "").lower():
        stor_dir = +1
    elif "bear" in (storage.bias or "").lower():
        stor_dir = -1

    if weather_dir == +1:
        if not price_ok_long:
            return ("WAIT", "No price confirmation for long")
        conf = 7.0 + (1.0 if accel else 0.0) + (1.0 if stor_dir == +1 else 0.0) - (1.0 if stor_dir == -1 else 0.0)
        conf = max(1.0, min(10.0, conf))
        return ("BOIL LONG (2–5D)", f"{conf:.1f}/10")
    else:
        if not price_ok_short:
            return ("WAIT", "No price confirmation for short")
        conf = 7.0 + (1.0 if accel else 0.0) + (1.0 if stor_dir == -1 else 0.0) - (1.0 if stor_dir == +1 else 0.0)
        conf = max(1.0, min(10.0, conf))
        return ("KOLD LONG (2–5D)", f"{conf:.1f}/10")

# =========================
# CSV / CHART / TELEGRAM
# =========================
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding="utf-8-sig")

def append_row(path: str, row: dict) -> pd.DataFrame:
    df = load_csv(path)
    out = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    out.to_csv(path, index=False)
    return out

def make_chart(weather_df: pd.DataFrame, run_date_utc: str, out_path: str) -> None:
    last30 = weather_df.tail(30).copy()
    if last30.empty:
        return
    fig = plt.figure(figsize=(10, 4.3))
    ax = plt.gca()
    ax.plot(last30["date"], last30["hdd"], label=f"Daily HDD (base {BASE_F:.0f}F)")
    ax.plot(last30["date"], last30["cdd"], label=f"Daily CDD (base {BASE_F:.0f}F)")
    ax.set_title(f"HDD/CDD Trend · {run_date_utc}")
    ax.set_xlabel("Day (UTC)")
    ax.set_ylabel("Degree Days")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)

def tg_send_message(token: str, chat_id: str, text: str) -> None:
    if not token or not chat_id:
        print("[INFO] Telegram secrets missing; skip message.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    r = requests.post(url, data=payload, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"Telegram sendMessage failed: {r.status_code} {r.text[:200]}")

def tg_send_photo(token: str, chat_id: str, photo_path: str, caption: str) -> None:
    if not token or not chat_id:
        print("[INFO] Telegram secrets missing; skip photo.")
        return
    if not os.path.exists(photo_path):
        print("[INFO] Chart not found; skip photo.")
        return
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    with open(photo_path, "rb") as f:
        files = {"photo": f}
        data = {"chat_id": chat_id, "caption": caption}
        r = requests.post(url, data=data, files=files, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"Telegram sendPhoto failed: {r.status_code} {r.text[:200]}")

# =========================
# MAIN
# =========================
def run():
    now = utc_now()
    run_date = now.strftime("%Y-%m-%d")
    run_ts = fmt_utc(now)

    dates, temps = fetch_daily_mean_f(LAT, LON, PAST_DAYS, FORECAST_DAYS)
    wdf = compute_hdd_cdd_series(dates, temps, BASE_F)
    m = compute_15_30(wdf)

    hist = load_csv(CSV_PATH)
    if not hist.empty and all(k in hist.columns for k in ["hdd_15d", "hdd_30d", "cdd_15d", "cdd_30d"]):
        prev_hdd15 = float(hist.iloc[-1]["hdd_15d"])
        prev_hdd30 = float(hist.iloc[-1]["hdd_30d"])
        prev_cdd15 = float(hist.iloc[-1]["cdd_15d"])
        prev_cdd30 = float(hist.iloc[-1]["cdd_30d"])
    else:
        prev_hdd15, prev_hdd30, prev_cdd15, prev_cdd30 = m["hdd_15d"], m["hdd_30d"], m["cdd_15d"], m["cdd_30d"]

    d_hdd15 = m["hdd_15d"] - prev_hdd15
    d_hdd30 = m["hdd_30d"] - prev_hdd30
    d_cdd15 = m["cdd_15d"] - prev_cdd15
    d_cdd30 = m["cdd_30d"] - prev_cdd30

    storage = fetch_storage_eia(EIA_API_KEY)
    price = fetch_price_with_fallback()

    signal, conf = decide_trade_2_5d_boilkold(d_hdd15, d_hdd30, storage, price)

    row = {
        "run_utc": run_ts,
        "date_utc": run_date,
        "hdd_15d": round(m["hdd_15d"], 2),
        "hdd_30d": round(m["hdd_30d"], 2),
        "cdd_15d": round(m["cdd_15d"], 2),
        "cdd_30d": round(m["cdd_30d"], 2),
        "delta_hdd15": round(d_hdd15, 2),
        "delta_hdd30": round(d_hdd30, 2),
        "delta_cdd15": round(d_cdd15, 2),
        "delta_cdd30": round(d_cdd30, 2),
        "storage_week": storage.week or "",
        "storage_total_bcf": storage.total_bcf if storage.total_bcf is not None else "",
        "storage_bias": storage.bias,
        "price_symbol": price.symbol,
        "price": price.price if price.price is not None else "",
        "ma5": price.ma5 if price.ma5 is not None else "",
        "break3_high": price.break3_high if price.break3_high is not None else "",
        "break3_low": price.break3_low if price.break3_low is not None else "",
        "signal": signal,
        "confidence": conf,
        "notes": f"storage_note={storage.note}; price_note={price.note}",
    }
    append_row(CSV_PATH, row)

    make_chart(wdf, f"{run_date} UTC", CHART_PATH)

    lines = []
    lines.append(f"📌 <b>HDD/CDD Update ({run_date})</b>")
    lines.append("")
    lines.append(f"🔥 <b>HDD</b> (base {BASE_F:.0f}F)")
    lines.append(f"• 15D Wtd: <b>{m['hdd_15d']:.2f}</b>  ({fmt_arrow(d_hdd15)} Δ {d_hdd15:+.2f})")
    lines.append(f"• 30D Wtd: <b>{m['hdd_30d']:.2f}</b>  ({fmt_arrow(d_hdd30)} Δ {d_hdd30:+.2f})")
    lines.append("")
    lines.append(f"🌤 <b>CDD</b> (base {BASE_F:.0f}F)")
    lines.append(f"• 15D Wtd: <b>{m['cdd_15d']:.2f}</b>  ({fmt_arrow(d_cdd15)} Δ {d_cdd15:+.2f})")
    lines.append(f"• 30D Wtd: <b>{m['cdd_30d']:.2f}</b>  ({fmt_arrow(d_cdd30)} Δ {d_cdd30:+.2f})")
    lines.append("")

    if storage.week and storage.total_bcf is not None:
        lines.append("🧱 <b>Storage</b> (EIA)")
        lines.append(f"• Week: {storage.week}")
        lines.append(f"• Total: {storage.total_bcf:.0f} bcf")
        lines.append(f"• Bias: <b>{storage.bias}</b>")
        lines.append("")
    else:
        lines.append("🧱 <b>Storage</b>: NA")
        if storage.note:
            lines.append(f"• Note: {storage.note}")
        lines.append("")

    if price.price is not None and price.ma5 is not None:
        above = "YES" if price.price > price.ma5 else "NO"
        b3h = "YES" if price.break3_high else "NO"
        b3l = "YES" if price.break3_low else "NO"
        lines.append(f"📈 <b>Price</b> ({price.symbol})")
        lines.append(f"• Close: {price.price:.3f}")
        lines.append(f"• Above 5MA: {above}")
        lines.append(f"• Break 3D High: {b3h} / Low: {b3l}")
        lines.append("")
    else:
        lines.append(f"📈 <b>Price</b>: NA ({price.note})")
        lines.append("")

    lines.append(f"🎯 <b>Signal</b>: {signal}")
    lines.append(f"🧠 Confidence: {conf}")
    lines.append(f"🕒 Updated: {run_ts}")

    msg = "\n".join(lines)
    tg_send_message(TG_BOT_TOKEN, TG_CHAT_ID, msg)
    tg_send_photo(TG_BOT_TOKEN, TG_CHAT_ID, CHART_PATH, caption=f"📈 Trend (HDD/CDD) · {run_date} UTC")

    # Actions log debug
    print("[INFO] ENV present:",
          f"TG_BOT_TOKEN={'yes' if TG_BOT_TOKEN else 'no'}",
          f"TG_CHAT_ID={'yes' if TG_CHAT_ID else 'no'}",
          f"EIA_API_KEY={'yes' if EIA_API_KEY else 'no'}")
    print(f"[INFO] Price source: {price.symbol} ({price.note})")
    print(f"[INFO] Storage note: {storage.note}")
    print("[OK] Done.")

if __name__ == "__main__":
    run()
