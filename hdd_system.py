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
# CONFIG (EDITABLE)
# =========================
LAT = 40.7128
LON = -74.0060
BASE_F = 65.0

CSV_PATH = "ng_composite_data.csv"
CHART_PATH = "ng_composite_chart.png"

PAST_DAYS = 14
FORECAST_DAYS = 16

PRICE_SYMBOL_PRIMARY = "NG=F"
PRICE_SYMBOL_FALLBACK = "UNG"
FRED_SERIES_FALLBACK = "DHHNGSP"

# =========================
# ENV (GitHub Secrets)
# =========================
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "").strip()
EIA_API_KEY = os.getenv("EIA_API_KEY", "").strip()
QUANDL_API_KEY = os.getenv("QUANDL_API_KEY", "").strip()
ENABLE_COT = os.getenv("ENABLE_COT", "0").strip() == "1"
COT_DATASET_CODE = os.getenv("COT_DATASET_CODE", "CFTC/067651_F_L_ALL").strip()

# =========================
# HELPERS
# =========================
@dataclass
class StorageInfo:
    week: Optional[str] = None
    total_bcf: Optional[float] = None
    wow_bcf: Optional[float] = None
    bias: str = "NA"
    note: str = ""

@dataclass
class PriceInfo:
    source: str
    symbol: str
    close: Optional[float] = None
    ma20: Optional[float] = None
    rsi14: Optional[float] = None
    vol10: Optional[float] = None
    note: str = ""

@dataclass
class COTInfo:
    net_managed_money: Optional[float] = None
    note: str = "disabled"

def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def fmt_utc(ts: dt.datetime) -> str:
    return ts.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def fmt_arrow(delta: float) -> str:
    if delta > 0.1: return "⬆️"
    if delta < -0.1: return "⬇️"
    return "➖"

def retry_get(url: str, params: dict = None, headers: dict = None, tries: int = 3, timeout: int = 25) -> requests.Response:
    last_err = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
            if r.status_code >= 400:
                raise requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
            return r
        except Exception as e:
            last_err = e
            time.sleep(1.5 + i)
    raise RuntimeError(f"HTTP failed: {last_err}")

def safe_float_list(xs: List) -> List[float]:
    return [float(x) if x is not None else float("nan") for x in xs]

def tg_send_message(token: str, chat_id: str, text: str) -> None:
    if not token or not chat_id: return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    requests.post(url, data=payload, timeout=25)

def tg_send_photo(token: str, chat_id: str, photo_path: str, caption: str) -> None:
    if not token or not chat_id or not os.path.exists(photo_path): return
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    with open(photo_path, "rb") as f:
        files = {"photo": f}
        data = {"chat_id": chat_id, "caption": caption}
        requests.post(url, data=data, files=files, timeout=40)

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path): return pd.DataFrame()
    return pd.read_csv(path)

def append_row(path: str, row: dict) -> pd.DataFrame:
    df = load_csv(path)
    out = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    out.to_csv(path, index=False)
    return out

# =========================
# WEATHER / HDD / CDD
# =========================
def fetch_daily_mean_f(lat: float, lon: float, past_days: int, forecast_days: int) -> Tuple[List[str], List[float]]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_mean",
        "temperature_unit": "fahrenheit", "timezone": "UTC",
        "past_days": past_days, "forecast_days": forecast_days,
    }
    r = retry_get(url, params=params)
    data = r.json().get("daily", {})
    return data.get("time", []), safe_float_list(data.get("temperature_2m_mean", []))

def compute_hdd_cdd_series(dates: List[str], temps_f: List[float], base_f: float) -> pd.DataFrame:
    df = pd.DataFrame({"date": pd.to_datetime(dates), "tmean_f": temps_f})
    df["tmean_f"] = df["tmean_f"].interpolate(limit_direction="both").ffill().bfill()
    df["hdd"] = np.maximum(0.0, base_f - df["tmean_f"])
    df["cdd"] = np.maximum(0.0, df["tmean_f"] - base_f)
    df["date"] = df["date"].dt.tz_localize(None)
    return df

def fut_sums(df: pd.DataFrame) -> Dict[str, float]:
    today = pd.Timestamp(dt.datetime.now(dt.timezone.utc).date()).tz_localize(None)
    fut = df[df["date"] >= today].sort_values("date").copy()
    if fut.empty: return {"hdd_fut7": 0.0, "hdd_fut15": 0.0, "cdd_fut7": 0.0, "cdd_fut15": 0.0}
    return {
        "hdd_fut7": float(fut.head(7)["hdd"].sum()),
        "hdd_fut15": float(fut.head(15)["hdd"].sum()),
        "cdd_fut7": float(fut.head(7)["cdd"].sum()),
        "cdd_fut15": float(fut.head(15)["cdd"].sum()),
    }

# =========================
# STORAGE (EIA v2 修正版)
# =========================
def fetch_storage_eia_v2(api_key: str) -> StorageInfo:
    if not api_key: return StorageInfo(note="No API Key")
    url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
    params = {
        "api_key": api_key,
        "frequency": "weekly",
        "data[0]": "value",
        "facets[series][]": "NW_NW_SWO_NG_R48_BCF", # 鎖定全美總量
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": 2
    }
    try:
        r = retry_get(url, params=params)
        data = r.json().get("response", {}).get("data", [])
        if not data: return StorageInfo(note="Empty data")
        
        curr_val = float(data[0]["value"])
        prev_val = float(data[1]["value"]) if len(data) > 1 else curr_val
        wow = curr_val - prev_val
        bias = "DRAW" if wow < 0 else "BUILD" if wow > 0 else "FLAT"
        return StorageInfo(week=data[0]["period"], total_bcf=curr_val, wow_bcf=wow, bias=bias, note="ok")
    except Exception as e:
        return StorageInfo(note=str(e)[:50])

# =========================
# PRICE & COT
# =========================
def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean().replace(0, np.nan)
    rs = roll_up / roll_down
    return float(100 - (100 / (1 + rs)).iloc[-1])

def fetch_price_yfinance(symbol: str) -> Optional[pd.Series]:
    try:
        import yfinance as yf
        df = yf.download(symbol, period="3mo", interval="1d", progress=False)
        if df is None or df.empty: return None
        return df["Close"].dropna()
    except: return None

def build_price_info() -> PriceInfo:
    close = fetch_price_yfinance(PRICE_SYMBOL_PRIMARY)
    if close is None: close = fetch_price_yfinance(PRICE_SYMBOL_FALLBACK)
    if close is not None and len(close) >= 20:
        return PriceInfo("YF", PRICE_SYMBOL_PRIMARY, float(close.iloc[-1]), float(close.tail(20).mean()), compute_rsi(close), note="ok")
    return PriceInfo("NA", "NA", note="Price fail")

def fetch_cot_quandl(dataset_code: str, api_key: str) -> COTInfo:
    if not api_key: return COTInfo(note="No Key")
    url = f"https://data.nasdaq.com/api/v3/datasets/{dataset_code}.json"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = retry_get(url, params={"api_key": api_key, "rows": 2}, headers=headers)
        ds = r.json().get("dataset", {})
        data = ds.get("data", [])
        cols = ds.get("column_names", [])
        if not data: return COTInfo(note="No data")
        df = pd.DataFrame(data, columns=cols)
        cand = [c for c in df.columns if "Managed" in c and "Net" in c]
        if cand: return COTInfo(net_managed_money=float(df.iloc[0][cand[0]]), note="ok")
        return COTInfo(note="Col not found")
    except Exception as e:
        return COTInfo(note=str(e)[:30])

# =========================
# SCORE & CHART
# =========================
def score_system(d_hdd_fut7, storage, price, cot) -> Tuple[int, int, int, int, int, str]:
    w = 2 if d_hdd_fut7 > 0.5 else -2 if d_hdd_fut7 < -0.5 else 0
    s = 2 if (storage.wow_bcf or 0) < -10 else -2 if (storage.wow_bcf or 0) > 10 else 0
    p = 2 if (price.close or 0) > (price.ma20 or 0) else -2
    c = 1 if (cot.net_managed_money or 0) > 0 else -1
    total = w + s + p + c
    sig = "BOIL LONG" if total >= 3 else "KOLD LONG" if total <= -3 else "WAIT"
    return w, s, p, c, total, sig

def make_chart(weather_df, run_tag, out_path):
    last30 = weather_df.tail(30)
    plt.figure(figsize=(10, 5))
    plt.plot(last30["date"], last30["hdd"], label="HDD")
    plt.plot(last30["date"], last30["cdd"], label="CDD")
    plt.title(f"NG Monitor - {run_tag}")
    plt.legend()
    plt.savefig(out_path)
    plt.close()

# =========================
# MAIN RUN
# =========================
def run():
    now = utc_now()
    run_date, run_ts = now.strftime("%Y-%m-%d"), fmt_utc(now)
    run_tag = "AM" if now.hour < 12 else "PM"
    
    dates, temps = fetch_daily_mean_f(LAT, LON, PAST_DAYS, FORECAST_DAYS)
    wdf = compute_hdd_cdd_series(dates, temps, BASE_F)
    f = fut_sums(wdf)
    
    hist = load_csv(CSV_PATH)
    prev_hdd = f["hdd_fut7"]
    if not hist.empty:
        same = hist[hist["run_tag"] == run_tag]
        if not same.empty: prev_hdd = float(same.iloc[-1]["hdd_fut7"])
    
    d_hdd = f["hdd_fut7"] - prev_hdd
    storage = fetch_storage_eia_v2(EIA_API_KEY)
    price = build_price_info()
    cot = fetch_cot_quandl(COT_DATASET_CODE, QUANDL_API_KEY) if ENABLE_COT else COTInfo()
    
    w_s, s_s, p_s, c_s, total, sig = score_system(d_hdd, storage, price, cot)
    
    row = {"run_utc": run_ts, "run_tag": run_tag, "hdd_fut7": f["hdd_fut7"], "delta_hdd": d_hdd, 
           "storage_total": storage.total_bcf, "price_close": price.close, "score_total": total, "signal": sig}
    append_row(CSV_PATH, row)
    make_chart(wdf, f"{run_date} {run_tag}", CHART_PATH)
    
    msg = f"📌 <b>NG Update ({run_date} {run_tag})</b>\n\n" \
          f"🌡️ HDD Fut7: {f['hdd_fut7']:.1f} ({fmt_arrow(d_hdd)} {d_hdd:+.1f})\n" \
          f"🧱 Storage: {storage.total_bcf or 'NA'} bcf (WoW: {storage.wow_bcf or 'NA'})\n" \
          f"📈 Price: {price.close or 'NA'}\n" \
          f"🧮 Score: {total} | 🎯 <b>{sig}</b>"
    
    tg_send_message(TG_BOT_TOKEN, TG_CHAT_ID, msg)
    tg_send_photo(TG_BOT_TOKEN, TG_CHAT_ID, CHART_PATH, f"Chart {run_date}")

if __name__ == "__main__":
    run()
