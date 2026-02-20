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
def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def fmt_utc(ts: dt.datetime) -> str:
    return ts.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def fmt_arrow(delta: float) -> str:
    if delta > 0.001: return "⬆️"
    if delta < -0.001: return "⬇️"
    return "➖"

def retry_get(url: str, params: dict = None, headers: dict = None, tries: int = 3, timeout: int = 25):
    last_err = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
            if r.status_code >= 400:
                raise requests.HTTPError(f"{r.status_code} {r.reason}: {r.text[:200]}", response=r)
            return r
        except Exception as e:
            last_err = e
            time.sleep(1.0 + i * 0.8)
    return str(last_err)

def safe_float_list(xs: List) -> List[float]:
    out = []
    for x in xs:
        if x is None:
            out.append(float("nan"))
        else:
            try:
                out.append(float(x))
            except:
                out.append(float("nan"))
    return out

def tg_send_message(token: str, chat_id: str, text: str) -> None:
    if not token or not chat_id: return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    requests.post(url, data=payload, timeout=25)

def tg_send_photo(token: str, chat_id: str, photo_path: str, caption: str) -> None:
    if not token or not chat_id or not os.path.exists(photo_path): return
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    with open(photo_path, "rb") as f:
        requests.post(url, data={"chat_id": chat_id, "caption": caption}, files={"photo": f}, timeout=40)

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path): return pd.DataFrame()
    try: return pd.read_csv(path)
    except: return pd.read_csv(path, encoding="utf-8-sig")

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
        "latitude": lat, "longitude": lon, "daily": "temperature_2m_mean",
        "temperature_unit": "fahrenheit", "timezone": "UTC",
        "past_days": past_days, "forecast_days": forecast_days,
    }
    r = retry_get(url, params=params)
    if isinstance(r, str): return [], []
    data = r.json().get("daily", {})
    return data.get("time", []), safe_float_list(data.get("temperature_2m_mean", []))

def compute_hdd_cdd_series(dates: List[str], temps_f: List[float], base_f: float) -> pd.DataFrame:
    df = pd.DataFrame({"date": pd.to_datetime(dates), "tmean_f": temps_f}).sort_values("date").reset_index(drop=True)
    df["tmean_f"] = df["tmean_f"].astype(float).interpolate(limit_direction="both").ffill().bfill()
    df["hdd"] = np.maximum(0.0, base_f - df["tmean_f"])
    df["cdd"] = np.maximum(0.0, df["tmean_f"] - base_f)
    return df

def weighted_avg_recent(values: np.ndarray) -> float:
    if len(values) == 0: return float("nan")
    w = np.linspace(0.5, 1.0, len(values))
    return float(np.sum(values * w) / np.sum(w))

def compute_15_30(df: pd.DataFrame) -> Dict[str, float]:
    last15, last30 = df.tail(15).sort_values("date"), df.tail(30).sort_values("date")
    return {
        "hdd_15d": weighted_avg_recent(last15["hdd"].to_numpy()),
        "hdd_30d": weighted_avg_recent(last30["hdd"].to_numpy()),
        "cdd_15d": weighted_avg_recent(last15["cdd"].to_numpy()),
        "cdd_30d": weighted_avg_recent(last30["cdd"].to_numpy()),
    }

def fut_sums(df: pd.DataFrame) -> Dict[str, float]:
    current_date = pd.Timestamp(dt.datetime.now(dt.timezone.utc).date())
    fut = df[df["date"] >= current_date].sort_values("date").reset_index(drop=True)
    if fut.empty:
        return {"hdd_fut7": 0.0, "hdd_fut15": 0.0, "cdd_fut7": 0.0, "cdd_fut15": 0.0}
    return {
        "hdd_fut7": float(fut.head(7)["hdd"].sum()),
        "hdd_fut15": float(fut.head(15)["hdd"].sum()),
        "cdd_fut7": float(fut.head(7)["cdd"].sum()),
        "cdd_fut15": float(fut.head(15)["cdd"].sum()),
    }

# =========================
# STORAGE (EIA v2 終極無腦抓取版)
# =========================
@dataclass
class StorageInfo:
    week: Optional[str] = None
    total_bcf: Optional[float] = None
    wow_bcf: Optional[float] = None
    bias: str = "NA"
    note: str = ""

def fetch_storage_eia_v2(api_key: str) -> StorageInfo:
    if not api_key: return StorageInfo(note="EIA_API_KEY not set")
    url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
    # 完全移除 facets 篩選，直接拿最新 50 筆原始數據
    params = {
        "api_key": api_key,
        "frequency": "weekly",
        "data[0]": "value",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": 50
    }
    r = retry_get(url, params=params)
    if isinstance(r, str): return StorageInfo(note=f"API Error: {r[:40]}")
    
    data = r.json().get("response", {}).get("data", [])
    if not data: return StorageInfo(note="EIA empty data")
    
    # 動態分組找最大值：因為 Total Lower 48 必定大於任何單一地區
    period_max = {}
    for d in data:
        p = str(d.get("period", ""))
        try:
            val_raw = d.get("value")
            if val_raw is None: continue
            v = float(val_raw)
            if p not in period_max or v > period_max[p]:
                period_max[p] = v
        except:
            continue
            
    sorted_periods = sorted(period_max.keys(), reverse=True)
    if not sorted_periods: return StorageInfo(note="No valid periods")
    
    curr_period = sorted_periods[0]
    curr_val = period_max[curr_period]
    prev_val = period_max[sorted_periods[1]] if len(sorted_periods) > 1 else curr_val
    wow = curr_val - prev_val
    
    return StorageInfo(week=curr_period, total_bcf=curr_val, wow_bcf=wow, bias="DRAW" if wow < 0 else "BUILD", note="ok")

# =========================
# PRICE
# =========================
@dataclass
class PriceInfo:
    source: str
    symbol: str
    close: Optional[float] = None
    ma20: Optional[float] = None
    rsi14: Optional[float] = None
    vol10: Optional[float] = None
    note: str = ""

def compute_rsi(series: np.ndarray, period: int = 14) -> float:
    delta = np.diff(series)
    up, down = np.clip(delta, 0, None), -np.clip(delta, None, 0)
    if len(up) < period: return float("nan")
    roll_up = pd.Series(up).rolling(period).mean().iloc[-1]
    roll_down = pd.Series(down).rolling(period).mean().iloc[-1]
    if roll_down == 0: return 100.0
    return float(100.0 - (100.0 / (1.0 + (roll_up / roll_down))))

def fetch_price_yfinance(symbol: str) -> Optional[np.ndarray]:
    try:
        import yfinance as yf
        df = yf.download(symbol, period="3mo", interval="1d", progress=False)
        if df is None or df.empty: return None
        return df["Close"].squeeze().dropna().values
    except: return None

def build_price_info() -> PriceInfo:
    close_arr = fetch_price_yfinance(PRICE_SYMBOL_PRIMARY)
    sym = PRICE_SYMBOL_PRIMARY
    if close_arr is None:
        close_arr = fetch_price_yfinance(PRICE_SYMBOL_FALLBACK)
        sym = PRICE_SYMBOL_FALLBACK

    if close_arr is not None and len(close_arr) >= 25:
        close_val = float(close_arr[-1])
        ma20_val = float(np.mean(close_arr[-20:]))
        rsi_val = compute_rsi(close_arr, 14)
        ret = pd.Series(close_arr).pct_change().dropna()
        vol_val = float(ret.tail(10).std() * np.sqrt(252))
        return PriceInfo(source="YF", symbol=sym, close=close_val, ma20=ma20_val, rsi14=rsi_val, vol10=vol_val, note="ok")
    return PriceInfo(source="NA", symbol="NA", note="price fail")

# =========================
# COT (Nasdaq)
# =========================
@dataclass
class COTInfo:
    net_managed_money: Optional[float] = None
    note: str = "disabled"

def fetch_cot_quandl(dataset_code: str, api_key: str) -> COTInfo:
    if not api_key: return COTInfo(note="No Key")
    url = f"https://data.nasdaq.com/api/v3/datasets/{dataset_code}.json"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = retry_get(url, params={"api_key": api_key, "rows": 5}, headers=headers)
        if isinstance(r, str): return COTInfo(note=r[:30])
        ds = r.json().get("dataset", {})
        cols, data = ds.get("column_names", []), ds.get("data", [])
        if not cols or not data: return COTInfo(note="Empty")
        df = pd.DataFrame(data, columns=cols)
        cand = [c for c in df.columns if "Managed" in c and "Net" in c]
        if cand: return COTInfo(net_managed_money=float(df.iloc[0][cand[0]]), note="ok")
        return COTInfo(note="Col missing")
    except Exception as e:
        err = str(e)
        return COTInfo(note="403 Forbidden" if "403" in err else err[:30])

# =========================
# CHART
# =========================
def make_chart(weather_df: pd.DataFrame, run_tag: str, out_path: str) -> None:
    last30 = weather_df.tail(30).copy()
    if last30.empty: return
    fig = plt.figure(figsize=(10, 4.3))
    ax = plt.gca()
    ax.plot(last30["date"], last30["hdd"], label=f"Daily HDD (base {BASE_F:.0f}F)")
    ax.plot(last30["date"], last30["cdd"], label=f"Daily CDD (base {BASE_F:.0f}F)")
    ax.set_title(f"NG Composite · HDD/CDD · {run_tag}")
    ax.set_xlabel("Day (UTC)")
    ax.set_ylabel("Degree Days")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)

# =========================
# SIGNAL
# =========================
def score_system(d_hdd_fut7, storage, price, cot) -> Tuple[int, int, int, int, int, str]:
    w = 2 if d_hdd_fut7 > 0.1 else -2 if d_hdd_fut7 < -0.1 else 0
    s = 2 if storage.wow_bcf is not None and storage.wow_bcf < -10 else -2 if storage.wow_bcf is not None and storage.wow_bcf > 10 else 0
    p = 2 if price.close is not None and price.ma20 is not None and price.close > price.ma20 else -2 if price.close is not None else 0
    c = 1 if cot.net_managed_money is not None and cot.net_managed_money > 0 else -1 if cot.net_managed_money is not None else 0
    total = w + s + p + c
    sig = "BOIL LONG (2–5D)" if total >= 4 else "KOLD LONG (2–5D)" if total <= -4 else "WAIT"
    return w, s, p, c, total, sig

# =========================
# MAIN
# =========================
def run():
    now = utc_now()
    run_date, run_ts = now.strftime("%Y-%m-%d"), fmt_utc(now)
    run_tag = "AM (UTC)" if now.hour < 12 else "PM (UTC)"

    dates, temps = fetch_daily_mean_f(LAT, LON, PAST_DAYS, FORECAST_DAYS)
    wdf = compute_hdd_cdd_series(dates, temps, BASE_F)
    m = compute_15_30(wdf)
    f = fut_sums(wdf)

    hist = load_csv(CSV_PATH)
    prev = hist[hist["run_tag"] == run_tag].iloc[-1].to_dict() if not hist.empty and run_tag in hist["run_tag"].values else {}
    
    def prevv(k, fb): return float(prev[k]) if k in prev and pd.notna(prev[k]) else fb
    d_hdd_fut7 = f["hdd_fut7"] - prevv("hdd_fut7", f["hdd_fut7"])
    d_hdd_fut15 = f["hdd_fut15"] - prevv("hdd_fut15", f["hdd_fut15"])
    d_cdd_fut7 = f["cdd_fut7"] - prevv("cdd_fut7", f["cdd_fut7"])
    d_cdd_fut15 = f["cdd_fut15"] - prevv("cdd_fut15", f["cdd_fut15"])

    storage = fetch_storage_eia_v2(EIA_API_KEY)
    price = build_price_info()
    cot = fetch_cot_quandl(COT_DATASET_CODE, QUANDL_API_KEY) if ENABLE_COT else COTInfo()

    w_score, s_score, p_score, c_score, total_score, signal = score_system(d_hdd_fut7, storage, price, cot)

    row = {
        "run_utc": run_ts, "date_utc": run_date, "run_tag": run_tag, "regime": "WINTER" if now.month in [11, 12, 1, 2, 3] else "SUMMER",
        "hdd_15d": round(m["hdd_15d"], 2), "hdd_30d": round(m["hdd_30d"], 2),
        "cdd_15d": round(m["cdd_15d"], 2), "cdd_30d": round(m["cdd_30d"], 2),
        "hdd_fut7": round(f["hdd_fut7"], 1), "hdd_fut15": round(f["hdd_fut15"], 1),
        "cdd_fut7": round(f["cdd_fut7"], 1), "cdd_fut15": round(f["cdd_fut15"], 1),
        "delta_hdd_fut7": round(d_hdd_fut7, 2), "delta_hdd_fut15": round(d_hdd_fut15, 2),
        "delta_cdd_fut7": round(d_cdd_fut7, 2), "delta_cdd_fut15": round(d_cdd_fut15, 2),
        "storage_week": storage.week or "", "storage_total_bcf": storage.total_bcf if storage.total_bcf is not None else "",
        "storage_wow_bcf": storage.wow_bcf if storage.wow_bcf is not None else "", "storage_bias": storage.bias,
        "price_symbol": price.symbol, "price_close": price.close if price.close is not None else "",
        "price_ma20": price.ma20 if price.ma20 is not None else "", "price_rsi14": price.rsi14 if price.rsi14 is not None else "",
        "price_vol10": price.vol10 if price.vol10 is not None else "", "cot_net_managed_money": cot.net_managed_money if cot.net_managed_money is not None else "",
        "score_weather": w_score, "score_storage": s_score, "score_price": p_score, "score_cot": c_score,
        "score_total": total_score, "signal": signal, "notes": f"storage={storage.note}; price={price.note}"
    }
    append_row(CSV_PATH, row)
    make_chart(wdf, f"{run_date} · {run_tag}", CHART_PATH)

    wow_str = "NA" if storage.wow_bcf is None else f"{storage.wow_bcf:+.0f} bcf"
    p_close_str = f"{price.close:.3f}" if price.close is not None else "NA"
    p_ma20_str = f"{price.ma20:.3f}" if price.ma20 is not None else "NA"
    p_rsi_str = f"{price.rsi14:.1f}" if price.rsi14 is not None and not np.isnan(price.rsi14) else "NA"
    p_vol_str = f"{price.vol10*100:.2f}%" if price.vol10 is not None and not np.isnan(price.vol10) else "NA"

    lines = [
        f"📌 <b>NG Composite Update ({run_date})</b>",
        f"• Run: <b>{run_tag}</b>",
        "",
        f"🌡️ <b>Composite HDD/CDD</b> (base {BASE_F:.0f}F)",
        f"• HDD 15D: <b>{m['hdd_15d']:.2f}</b> | 30D: <b>{m['hdd_30d']:.2f}</b>",
        f"• CDD 15D: <b>{m['cdd_15d']:.2f}</b> | 30D: <b>{m['cdd_30d']:.2f}</b>",
        "",
        "🧊/🔥 <b>Forecast Revision</b>",
        f"• HDD Fut7: <b>{f['hdd_fut7']:.1f}</b> ({fmt_arrow(d_hdd_fut7)} {d_hdd_fut7:+.2f})",
        f"• CDD Fut7: <b>{f['cdd_fut7']:.1f}</b> ({fmt_arrow(d_cdd_fut7)} {d_cdd_fut7:+.2f})",
        "",
    ]

    if storage.week and storage.total_bcf is not None:
        lines.extend([
            "🧱 <b>Storage</b> (EIA · Lower 48 Total)",
            f"• Week: {storage.week} | Total: {storage.total_bcf:.0f} bcf",
            f"• WoW: {wow_str} | Bias: <b>{storage.bias}</b>",
            ""
        ])
    else:
        lines.extend(["🧱 <b>Storage</b>: NA", f"• Note: {storage.note}", ""])

    lines.extend([
        f"📈 <b>Price</b> ({price.symbol})",
        f"• Close: {p_close_str} | MA20: {p_ma20_str}",
        f"• RSI14: {p_rsi_str} | Vol10: {p_vol_str}",
        "",
        "🧮 <b>Score</b> (Weather / Storage / Price / COT)",
        f"• {w_score} / {s_score} / {p_score} / {c_score}  → Total: <b>{total_score}</b>",
        "",
        f"🎯 <b>Signal</b>: {signal}",
        f"🕒 Updated: {run_ts}"
    ])

    msg = "\n".join(lines)
    tg_send_message(TG_BOT_TOKEN, TG_CHAT_ID, msg)
    tg_send_photo(TG_BOT_TOKEN, TG_CHAT_ID, CHART_PATH, caption=f"📈 NG Composite Chart · {run_date}")
    print("[OK] Done.")

if __name__ == "__main__":
    run()
