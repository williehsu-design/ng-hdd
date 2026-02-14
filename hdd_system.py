import os
import json
import time
import re
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
CHART_PATH = "hdd_cdd_chart.png"

PAST_DAYS = 14
FORECAST_DAYS = 16  # 14 + 16 + today ≈ 31 points

# Price: use FRED (stable, no key)
PRICE_FRED_SERIES = "DHHNGSP"  # Henry Hub spot

# Delta display / signal threshold (avoid +0.00 / -0.00 noise)
EPS = 0.01  # treat |delta| < 0.01 as 0

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

def _norm_delta(x: float) -> float:
    return 0.0 if abs(x) < EPS else float(x)

def fmt_arrow(delta: float) -> str:
    delta = _norm_delta(delta)
    if delta == 0.0:
        return "➖"
    return "⬆️" if delta > 0 else "⬇️"

def run_tag_from_utc(now: dt.datetime) -> str:
    # schedule is 00:20 & 12:20 UTC → label them AM/PM
    return "AM" if now.hour < 12 else "PM"

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
    w = np.linspace(0.5, 1.0, n)  # increasing weights
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
# STORAGE (Lower 48 Total Working Gas)
# =========================
@dataclass
class StorageInfo:
    week: Optional[str] = None         # YYYY-MM-DD
    total_bcf: Optional[float] = None
    wow_bcf: Optional[float] = None
    bias: str = "NA"                   # DRAW / BUILD / FLAT / NA
    note: str = ""

def _parse_eia_storage_report_fallback() -> StorageInfo:
    """
    Fallback scrape (no key): parse eia.gov/naturalgas/storage text.
    """
    url = "https://www.eia.gov/naturalgas/storage/"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    html = r.text

    m = re.search(
        r"Working gas in storage was\s+([\d,]+)\s+Bcf\s+as of Friday,\s+([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})",
        html
    )
    if not m:
        return StorageInfo(note="fallback parse failed (pattern not found)")

    bcf = float(m.group(1).replace(",", ""))
    month_name = m.group(2)
    day = int(m.group(3))
    year = int(m.group(4))

    week = None
    try:
        d = dt.datetime.strptime(f"{month_name} {day} {year}", "%B %d %Y").date()
        week = d.isoformat()
    except Exception:
        pass

    return StorageInfo(week=week, total_bcf=bcf, bias="NA", note="fallback: eia.gov/storage")

def fetch_storage_eia(api_key: str) -> StorageInfo:
    """
    Primary: EIA series endpoint for Lower 48 total working gas (Bcf).
      series_id = NG.NW2_EPG0_SWO_R48_BCF.W
    """
    if not api_key:
        return StorageInfo(note="EIA_API_KEY not set (storage skipped)")

    url = "https://api.eia.gov/series/"
    params = {
        "api_key": api_key,
        "series_id": "NG.NW2_EPG0_SWO_R48_BCF.W",
        "out": "json",
    }

    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        j = r.json()

        series = (j.get("series") or [])
        if not series or not series[0].get("data"):
            raise RuntimeError("EIA series empty")

        data = series[0]["data"]  # newest first: [[YYYYMMDD, value], ...]
        latest = data[0]
        prev = data[1] if len(data) > 1 else None

        date_raw = str(latest[0])  # YYYYMMDD
        val = float(latest[1])

        week = None
        if len(date_raw) == 8 and date_raw.isdigit():
            week = f"{date_raw[0:4]}-{date_raw[4:6]}-{date_raw[6:8]}"

        wow = None
        bias = "NA"
        if prev is not None:
            prev_val = float(prev[1])
            wow = val - prev_val
            if abs(wow) < 0.5:
                bias = "FLAT"
            elif wow < 0:
                bias = "DRAW"
            else:
                bias = "BUILD"

        return StorageInfo(
            week=week,
            total_bcf=val,
            wow_bcf=wow,
            bias=bias,
            note="ok: NG.NW2_EPG0_SWO_R48_BCF.W"
        )
    except Exception as e:
        fb = _parse_eia_storage_report_fallback()
        fb.note = f"primary failed ({e}); {fb.note}"
        return fb

# =========================
# PRICE (FRED)
# =========================
@dataclass
class PriceInfo:
    symbol: str
    price: Optional[float] = None
    ma5: Optional[float] = None
    break3_high: Optional[bool] = None
    break3_low: Optional[bool] = None
    note: str = ""

def _calc_price_signals(close: pd.Series, symbol: str, source_note: str) -> PriceInfo:
    close = pd.to_numeric(close, errors="coerce").dropna()
    if len(close) < 6:
        return PriceInfo(symbol=symbol, note=f"{source_note}: not enough bars")
    last = float(close.iloc[-1])
    ma5 = float(close.tail(5).mean())
    prior3 = close.iloc[-4:-1]
    break3_high = bool(last > float(prior3.max()))
    break3_low = bool(last < float(prior3.min()))
    return PriceInfo(symbol=symbol, price=last, ma5=ma5, break3_high=break3_high, break3_low=break3_low, note="ok")

def fetch_price_fred(series_id: str) -> PriceInfo:
    """
    FRED CSV (no key):
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

        col = df.columns[1]
        close = df[col]
        return _calc_price_signals(close, f"FRED:{series_id}", "fred")
    except Exception as e:
        return PriceInfo(symbol=f"FRED:{series_id}", note=f"fred failed: {e}")

# =========================
# SIGNAL LOGIC
# =========================
def decide_trade_2_5d_boilkold(
    d_hdd15: float,
    d_hdd30: float,
    storage: StorageInfo,
    price: PriceInfo,
) -> Tuple[str, str]:
    d_hdd15 = _norm_delta(d_hdd15)
    d_hdd30 = _norm_delta(d_hdd30)

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
    if storage.bias == "DRAW":
        stor_dir = +1
    elif storage.bias == "BUILD":
        stor_dir = -1

    if weather_dir == +1:
        if not price_ok_long:
            return ("WAIT", "No price confirmation for long")
        conf = 7.0 + (1.0 if accel else 0.0) + (0.5 if stor_dir == +1 else 0.0) - (0.5 if stor_dir == -1 else 0.0)
        conf = max(1.0, min(10.0, conf))
        return ("BOIL LONG (2–5D)", f"{conf:.1f}/10")
    else:
        if not price_ok_short:
            return ("WAIT", "No price confirmation for short")
        conf = 7.0 + (1.0 if accel else 0.0) + (0.5 if stor_dir == -1 else 0.0) - (0.5 if stor_dir == +1 else 0.0)
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

def make_chart(weather_df: pd.DataFrame, run_label: str, out_path: str) -> None:
    last30 = weather_df.tail(30).copy()
    if last30.empty:
        return
    fig = plt.figure(figsize=(10, 4.3))
    ax = plt.gca()
    ax.plot(last30["date"], last30["hdd"], label=f"Daily HDD (base {BASE_F:.0f}F)")
    ax.plot(last30["date"], last30["cdd"], label=f"Daily CDD (base {BASE_F:.0f}F)")
    ax.set_title(f"HDD/CDD Trend · {run_label}")
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
    tag = run_tag_from_utc(now)  # AM / PM
    run_label = f"{run_date} {tag} UTC"

    # 1) Weather
    dates, temps = fetch_daily_mean_f(LAT, LON, PAST_DAYS, FORECAST_DAYS)
    wdf = compute_hdd_cdd_series(dates, temps, BASE_F)

    # 2) Metrics
    m = compute_15_30(wdf)

    # 3) Delta vs previous row (compare to last record, regardless AM/PM)
    hist = load_csv(CSV_PATH)
    if not hist.empty and all(k in hist.columns for k in ["hdd_15d", "hdd_30d", "cdd_15d", "cdd_30d"]):
        prev_hdd15 = float(hist.iloc[-1]["hdd_15d"])
        prev_hdd30 = float(hist.iloc[-1]["hdd_30d"])
        prev_cdd15 = float(hist.iloc[-1]["cdd_15d"])
        prev_cdd30 = float(hist.iloc[-1]["cdd_30d"])
    else:
        prev_hdd15, prev_hdd30, prev_cdd15, prev_cdd30 = m["hdd_15d"], m["hdd_30d"], m["cdd_15d"], m["cdd_30d"]

    d_hdd15 = _norm_delta(m["hdd_15d"] - prev_hdd15)
    d_hdd30 = _norm_delta(m["hdd_30d"] - prev_hdd30)
    d_cdd15 = _norm_delta(m["cdd_15d"] - prev_cdd15)
    d_cdd30 = _norm_delta(m["cdd_30d"] - prev_cdd30)

    # 4) Storage (Lower 48 total)
    storage = fetch_storage_eia(EIA_API_KEY)

    # 5) Price (FRED)
    price = fetch_price_fred(PRICE_FRED_SERIES)

    # 6) Signal
    signal, conf = decide_trade_2_5d_boilkold(d_hdd15, d_hdd30, storage, price)

    # 7) Save CSV
    row = {
        "run_utc": run_ts,
        "run_tag": tag,  # AM/PM
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
        "storage_wow_bcf": storage.wow_bcf if storage.wow_bcf is not None else "",
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

    # 8) Chart (same filename, overwritten each run)
    make_chart(wdf, run_label, CHART_PATH)

    # 9) Telegram message
    lines = []
    lines.append(f"📌 <b>HDD/CDD Update ({run_date})</b>")
    lines.append(f"• Run: <b>{tag}</b> (UTC)")
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
        lines.append("🧱 <b>Storage</b> (EIA · Lower 48 Total)")
        lines.append(f"• Week: {storage.week}")
        lines.append(f"• Total: {storage.total_bcf:.0f} bcf")
        if storage.wow_bcf is not None:
            sign = "+" if storage.wow_bcf >= 0 else ""
            lines.append(f"• WoW: {sign}{storage.wow_bcf:.0f} bcf")
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
    tg_send_photo(TG_BOT_TOKEN, TG_CHAT_ID, CHART_PATH, caption=f"📈 Trend (HDD/CDD) · {run_label}")

    print("[INFO] ENV present:",
          f"TG_BOT_TOKEN={'yes' if TG_BOT_TOKEN else 'no'}",
          f"TG_CHAT_ID={'yes' if TG_CHAT_ID else 'no'}",
          f"EIA_API_KEY={'yes' if EIA_API_KEY else 'no'}")
    print(f"[INFO] Storage note: {storage.note}")
    print(f"[INFO] Price source: {price.symbol} ({price.note})")
    print("[OK] Done.")

if __name__ == "__main__":
    run()
