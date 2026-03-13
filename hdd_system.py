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

# =========================
# HELPERS
# =========================
def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def fmt_utc(ts: dt.datetime) -> str:
    return ts.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def fmt_arrow(delta: float) -> str:
    if delta > 0.1: return "⬆️"
    if delta < -0.1: return "⬇️"
    return "➖"

def retry_get(url: str, params: dict = None, headers: dict = None, tries: int = 3, timeout: int = 25):
    last_err = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
            if r.status_code >= 400:
                raise requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
            return r
        except Exception as e:
            last_err = e
            time.sleep(1.0 + i * 0.8)
    return str(last_err)

def safe_float_list(xs: List) -> List[float]:
    out = []
    for x in xs:
        if x is None: out.append(float("nan"))
        else:
            try: out.append(float(x))
            except: out.append(float("nan"))
    return out

def tg_send_message(token: str, chat_id: str, text: str) -> None:
    if not token or not chat_id: return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    r = requests.post(url, data=payload, timeout=25)
    if r.status_code >= 400:
        print(f"TG Msg Error: {r.text}")

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
        "hdd_15d": weighted_avg_recent(last15["hdd"].to_numpy()), "hdd_30d": weighted_avg_recent(last30["hdd"].to_numpy()),
        "cdd_15d": weighted_avg_recent(last15["cdd"].to_numpy()), "cdd_30d": weighted_avg_recent(last30["cdd"].to_numpy()),
    }

def fut_sums(df: pd.DataFrame) -> Dict[str, float]:
    current_date = pd.Timestamp(dt.datetime.now(dt.timezone.utc).date())
    fut = df[df["date"] >= current_date].sort_values("date").reset_index(drop=True)
    if fut.empty: return {"hdd_fut7": 0.0, "hdd_fut15": 0.0, "cdd_fut7": 0.0, "cdd_fut15": 0.0}
    return {
        "hdd_fut7": float(fut.head(7)["hdd"].sum()), "hdd_fut15": float(fut.head(15)["hdd"].sum()),
        "cdd_fut7": float(fut.head(7)["cdd"].sum()), "cdd_fut15": float(fut.head(15)["cdd"].sum()),
    }

# =========================
# STORAGE
# =========================
@dataclass
class StorageInfo:
    week: Optional[str] = None
    total_bcf: Optional[float] = None
    wow_bcf: Optional[float] = None
    wow_5yr_avg: Optional[float] = None
    bias: str = "NA"
    note: str = ""

def fetch_storage_eia_v2(api_key: str) -> StorageInfo:
    if not api_key: return StorageInfo(note="EIA_API_KEY not set")
    url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
    params = {
        "api_key": api_key, "frequency": "weekly", "data[0]": "value",
        "sort[0][column]": "period", "sort[0][direction]": "desc", "length": 2500
    }
    r = retry_get(url, params=params)
    if isinstance(r, str): return StorageInfo(note=f"API Error: {r[:40]}")
    
    data = r.json().get("response", {}).get("data", [])
    if not data: return StorageInfo(note="EIA empty data")
    
    period_max = {}
    for d in data:
        p = str(d.get("period", ""))
        try:
            v = float(d.get("value"))
            if p not in period_max or v > period_max[p]: period_max[p] = v
        except: continue
            
    sorted_periods = sorted(period_max.keys(), reverse=True)
    if not sorted_periods: return StorageInfo(note="No valid periods")
    
    curr_period = sorted_periods[0]
    curr_val = period_max[curr_period]
    prev_val = period_max[sorted_periods[1]] if len(sorted_periods) > 1 else curr_val
    wow = curr_val - prev_val
    
    wow_5yr_avg = None
    try:
        curr_date = dt.datetime.strptime(curr_period, "%Y-%m-%d")
        curr_week_num = curr_date.isocalendar()[1]
        historical_wows = []
        
        for i in range(1, len(sorted_periods)):
            p_date = dt.datetime.strptime(sorted_periods[i], "%Y-%m-%d")
            if p_date.year < curr_date.year and p_date.isocalendar()[1] == curr_week_num:
                if i + 1 < len(sorted_periods):
                    h_curr = period_max[sorted_periods[i]]
                    h_prev = period_max[sorted_periods[i+1]]
                    historical_wows.append(h_curr - h_prev)
                if len(historical_wows) == 5: break 
                
        if historical_wows:
            wow_5yr_avg = sum(historical_wows) / len(historical_wows)
    except: pass
    
    return StorageInfo(week=curr_period, total_bcf=curr_val, wow_bcf=wow, wow_5yr_avg=wow_5yr_avg, bias="DRAW" if wow < 0 else "BUILD", note="ok")

# =========================
# PRICE & HACKER BACK-ADJUSTED KELTNER CHANNELS
# =========================
@dataclass
class PriceInfo:
    source: str
    symbol: str
    close: Optional[float] = None
    ma20: Optional[float] = None
    ema20: Optional[float] = None
    atr14: Optional[float] = None
    kc_upper: Optional[float] = None
    kc_lower: Optional[float] = None
    rsi14: Optional[float] = None
    vol10: Optional[float] = None
    short_float_pct: Optional[float] = None
    note: str = ""

def compute_rsi(series: np.ndarray, period: int = 14) -> float:
    delta = np.diff(series)
    up, down = np.clip(delta, 0, None), -np.clip(delta, None, 0)
    if len(up) < period: return float("nan")
    roll_up = pd.Series(up).rolling(period).mean().iloc[-1]
    roll_down = pd.Series(down).rolling(period).mean().iloc[-1]
    if roll_down == 0: return 100.0
    return float(100.0 - (100.0 / (1.0 + (roll_up / roll_down))))

def fetch_price_yfinance_df(symbol: str) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        df_ng = yf.Ticker(symbol).history(period="6mo")[['High', 'Low', 'Close']]
        df_ung = yf.Ticker("UNG").history(period="6mo")[['Close']].rename(columns={'Close': 'Close_UNG'})
        
        if df_ng.empty or df_ung.empty: 
            return None

        df_ng.index = df_ng.index.tz_localize(None)
        df_ung.index = df_ung.index.tz_localize(None)

        df = df_ng.join(df_ung, how='inner')
        if len(df) < 25:
            return None

        df['Ret_NG'] = df['Close'].pct_change()
        df['Ret_UNG'] = df['Close_UNG'].pct_change()

        threshold = 0.04
        anomaly_mask = (df['Ret_NG'] - df['Ret_UNG']).abs() > threshold

        df['Fixed_Ret'] = df['Ret_NG']
        df.loc[anomaly_mask, 'Fixed_Ret'] = df.loc[anomaly_mask, 'Ret_UNG']

        adj_close = np.zeros(len(df))
        adj_close[-1] = df['Close'].iloc[-1]

        for i in range(len(df)-2, -1, -1):
            adj_close[i] = adj_close[i+1] / (1 + df['Fixed_Ret'].iloc[i+1])

        df['Adj_Close'] = adj_close
        df['Adj_Ratio'] = df['Adj_Close'] / df['Close']
        df['Adj_High'] = df['High'] * df['Adj_Ratio']
        df['Adj_Low'] = df['Low'] * df['Adj_Ratio']

        out_df = pd.DataFrame({
            'High': df['Adj_High'],
            'Low': df['Adj_Low'],
            'Close': df['Adj_Close']
        }, index=df.index)
        
        return out_df.dropna()
    except Exception as e:
        print(f"Back-Adjust Error: {e}")
        return None

def build_price_info() -> PriceInfo:
    df = fetch_price_yfinance_df(PRICE_SYMBOL_PRIMARY)
    sym = PRICE_SYMBOL_PRIMARY
    if df is None or len(df) < 25:
        df = fetch_price_yfinance_df(PRICE_SYMBOL_FALLBACK)
        sym = PRICE_SYMBOL_FALLBACK

    if df is not None and len(df) >= 25:
        close_arr = df['Close'].values
        close_val = float(close_arr[-1])
        ma20_val = float(np.mean(close_arr[-20:]))
        
        ema20 = df['Close'].ewm(span=20, adjust=False).mean()
        ema20_val = float(ema20.iloc[-1])
        
        prev_close = df['Close'].shift(1)
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - prev_close).abs()
        tr3 = (df['Low'] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        atr_val = float(atr14.iloc[-1])
        
        kc_upper_val = ema20_val + (2.0 * atr_val)
        kc_lower_val = ema20_val - (2.0 * atr_val)
        
        rsi_val = compute_rsi(close_arr, 14)
        ret = pd.Series(close_arr).pct_change().dropna()
        vol_val = float(ret.tail(10).std() * np.sqrt(252))
        
        # 抓取 UNG (ETF) 空單佔比做為全市場籌碼參考 - 增強版
        short_float_pct = None
        try:
            import yfinance as yf
            info = yf.Ticker("UNG").info
            if info.get("shortPercentOfFloat") is not None:
                short_float_pct = float(info["shortPercentOfFloat"]) * 100.0
            elif info.get("sharesShort") and info.get("floatShares"):
                short_float_pct = (float(info["sharesShort"]) / float(info["floatShares"])) * 100.0
            elif info.get("sharesShort") and info.get("sharesOutstanding"):
                short_float_pct = (float(info["sharesShort"]) / float(info["sharesOutstanding"])) * 100.0
        except:
            pass
        
        return PriceInfo(
            source="YF_Hacked", symbol=sym, close=close_val, 
            ma20=ma20_val, ema20=ema20_val, atr14=atr_val,
            kc_upper=kc_upper_val, kc_lower=kc_lower_val,
            rsi14=rsi_val, vol10=vol_val, short_float_pct=short_float_pct, note="ok"
        )
    return PriceInfo(source="NA", symbol="NA", note="price fail")

# =========================
# 🛡️ STRICT MACRO GEOPOLITICAL RISK MODULE
# =========================
@dataclass
class MacroRiskInfo:
    oil_change_pct: float = 0.0
    vix_change_pct: float = 0.0
    is_war_risk_high: bool = False
    note: str = ""

def check_macro_risk() -> MacroRiskInfo:
    try:
        import yfinance as yf
        tickers = yf.Tickers("CL=F ^VIX")
        hist_oil = tickers.tickers['CL=F'].history(period="5d")['Close']
        hist_vix = tickers.tickers['^VIX'].history(period="5d")['Close']
        
        if len(hist_oil) >= 2 and len(hist_vix) >= 2:
            oil_change = (hist_oil.iloc[-1] / hist_oil.iloc[-2]) - 1.0
            vix_change = (hist_vix.iloc[-1] / hist_vix.iloc[-2]) - 1.0
            vix_current = float(hist_vix.iloc[-1])
            
            cond_double = (oil_change > 0.035) and (vix_change > 0.10)
            cond_oil_extreme = (oil_change > 0.055)
            cond_vix_extreme = (vix_change > 0.20) and (vix_current > 20.0)
            
            is_risk_high = cond_double or cond_oil_extreme or cond_vix_extreme
            
            return MacroRiskInfo(
                oil_change_pct=float(oil_change),
                vix_change_pct=float(vix_change),
                is_war_risk_high=bool(is_risk_high),
                note="ok"
            )
    except Exception as e:
        return MacroRiskInfo(note=str(e))
    return MacroRiskInfo()

# =========================
# COT (CFTC Official API - 完全免費免 Key)
# =========================
@dataclass
class COTInfo:
    net_managed_money: Optional[float] = None
    mm_long: Optional[float] = None
    mm_short: Optional[float] = None
    ls_ratio: Optional[float] = None
    net_wow_change: Optional[float] = None
    note: str = "disabled"

def fetch_cot_cftc() -> COTInfo:
    """直接從美國 CFTC 官方 API 抓取大額交易人報告 (Disaggregated Futures Only)"""
    url = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
    
    # 023651 是 NYMEX Henry Hub 天然氣合約代碼
    params = {
        "cftc_contract_market_code": "023651",
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": 2
    }
    
    try:
        r = retry_get(url, params=params)
        if isinstance(r, str): return COTInfo(note=f"CFTC Error: {r[:30]}")
        
        data = r.json()
        if not data or len(data) == 0:
            return COTInfo(note="Empty CFTC data")
            
        # 本週數據 (Managed Money = m_money)
        curr = data[0]
        mm_long_curr = float(curr.get("m_money_positions_long_all", 0))
        mm_short_curr = float(curr.get("m_money_positions_short_all", 0))
        net_curr = mm_long_curr - mm_short_curr
        
        # 上週數據以計算 WoW 變化
        if len(data) > 1:
            prev = data[1]
            mm_long_prev = float(prev.get("m_money_positions_long_all", 0))
            mm_short_prev = float(prev.get("m_money_positions_short_all", 0))
            net_prev = mm_long_prev - mm_short_prev
        else:
            net_prev = net_curr
            
        ls_ratio = (mm_long_curr / mm_short_curr) if mm_short_curr > 0 else None
        
        return COTInfo(
            net_managed_money=net_curr,
            mm_long=mm_long_curr,
            mm_short=mm_short_curr,
            ls_ratio=ls_ratio,
            net_wow_change=net_curr - net_prev,
            note="ok"
        )
    except Exception as e:
        return COTInfo(note=f"Exception: {str(e)[:30]}")

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
# SIGNAL (整合籌碼多空/軋空因子)
# =========================
def score_system(d_hdd_fut7, storage, price, cot, macro: MacroRiskInfo) -> Tuple[int, int, int, int, int, int, str]:
    # 1. 🌡️ 天氣
    if d_hdd_fut7 >= 10.0: w = 2
    elif d_hdd_fut7 >= 3.0: w = 1
    elif d_hdd_fut7 <= -10.0: w = -2
    elif d_hdd_fut7 <= -3.0: w = -1
    else: w = 0  
    
    # 2. 🧱 庫存
    s = 0
    if storage.wow_bcf is not None and storage.wow_5yr_avg is not None:
        diff = storage.wow_bcf - storage.wow_5yr_avg
        if diff <= -15: s = 2
        elif diff <= -5: s = 1
        elif diff >= 15: s = -2
        elif diff >= 5: s = -1
    elif storage.wow_bcf is not None:
        s = 2 if storage.wow_bcf < -10 else -2 if storage.wow_bcf > 10 else 0

    # 3. 📈 價格
    p = 0
    if price.close is not None and price.ma20 is not None and price.rsi14 is not None:
        if price.close > price.ma20:
            p = -1 if price.rsi14 > 75 else 2
        else:
            p = 1 if price.rsi14 < 25 else -2

    # 4. 🏛️ 籌碼與空單 (升級版)
    c = 0
    if cot.net_managed_money is not None:
        c += 1 if cot.net_managed_money > 0 else -1
        
    if cot.net_wow_change is not None:
        c += 1 if cot.net_wow_change > 0 else -1
        
    # 若空單極高且技術面/基本面不差，加入軋空加分 (Short Squeeze Potential)
    if price.short_float_pct is not None and price.short_float_pct >= 15.0:
        if (w + s + p) >= 0:
            c += 1
    
    # 5. 💣 宏觀戰爭風險
    macro_score = 4 if macro.is_war_risk_high else 0 
    
    total = w + s + p + c + macro_score
    
    # 決定最終訊號
    if macro.is_war_risk_high:
        sig = "BOIL LONG (WAR RISK OVERRIDE)"
    elif total >= 3:
        sig = "BOIL LONG (2–5D)"
    elif total <= -3:
        sig = "KOLD LONG (2–5D)"
    else:
        sig = "WAIT"
        
    return w, s, p, c, macro_score, total, sig

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
    cot = fetch_cot_cftc()  # 永遠執行 CFTC 抓取
    macro = check_macro_risk()
    
    w_score, s_score, p_score, c_score, m_score, total_score, signal = score_system(d_hdd_fut7, storage, price, cot, macro)

    show_hdd = (m['hdd_30d'] + f['hdd_fut7']) > 0.5
    show_cdd = (m['cdd_30d'] + f['cdd_fut7']) > 0.5

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
        "price_ma20": price.ma20 if price.ma20 is not None else "", 
        "price_kc_upper": price.kc_upper if price.kc_upper is not None else "",
        "price_kc_lower": price.kc_lower if price.kc_lower is not None else "",
        "price_atr": price.atr14 if price.atr14 is not None else "",
        "price_rsi14": price.rsi14 if price.rsi14 is not None else "",
        "price_vol10": price.vol10 if price.vol10 is not None else "", 
        "price_short_float": price.short_float_pct if price.short_float_pct is not None else "",
        "cot_net_managed_money": cot.net_managed_money if cot.net_managed_money is not None else "",
        "cot_mm_long": cot.mm_long if cot.mm_long is not None else "",
        "cot_mm_short": cot.mm_short if cot.mm_short is not None else "",
        "cot_ls_ratio": cot.ls_ratio if cot.ls_ratio is not None else "",
        "cot_wow_change": cot.net_wow_change if cot.net_wow_change is not None else "",
        "score_weather": w_score, "score_storage": s_score, "score_price": p_score, "score_cot": c_score,
        "score_macro": m_score, "score_total": total_score, "signal": signal, "notes": f"storage={storage.note}; price={price.note}"
    }
    append_row(CSV_PATH, row)
    make_chart(wdf, f"{run_date} · {run_tag}", CHART_PATH)

    # =========================
    # 🚨 動態警示系統 (ALERTS)
    # =========================
    alerts = []
    
    if macro.is_war_risk_high:
        alerts.append(f"☢️ <b>地緣政治核彈警報</b>：偵測到避險資產異常飆升！原油變化 {macro.oil_change_pct*100:+.2f}%，VIX 變化 {macro.vix_change_pct*100:+.2f}%。")

    if price.close is not None and price.kc_upper is not None and price.kc_lower is not None:
        if price.close > price.kc_upper:
            alerts.append(f"⚠️ <b>肯特納突破 (強勢)</b>：收盤突破上軌 ({price.kc_upper:.2f})！短線動能極強，注意分批停利。")
        elif price.close < price.kc_lower:
            alerts.append(f"⚠️ <b>肯特納跌破 (極端超賣)</b>：收盤跌破下軌 ({price.kc_lower:.2f})！恐慌性殺跌，準備抓死貓反彈。")

    if price.rsi14 is not None:
        if price.rsi14 < 25:
            alerts.append(f"⚠️ <b>極度超賣 (RSI={price.rsi14:.1f})</b>：空頭動能衰竭，慎防報復性反彈！")
        elif price.rsi14 > 75:
            alerts.append(f"⚠️ <b>極度超買 (RSI={price.rsi14:.1f})</b>：多頭過熱，慎防高檔暴跌！")
            
    if price.short_float_pct is not None and price.short_float_pct >= 15.0:
        alerts.append(f"🔥 <b>軋空預警 (Short Float {price.short_float_pct:.1f}%)</b>：市場空單偏高，一旦反轉可能觸發軋空大暴漲！")

    if abs(d_hdd_fut7) >= 10.0:
        dir_str = "轉冷" if d_hdd_fut7 > 0 else "轉暖"
        alerts.append(f"⚠️ <b>氣象突變</b>：預報大幅{dir_str} (變化 {d_hdd_fut7:+.1f} HDD)！")

    if now.weekday() == 4:
        alerts.append("⚠️ <b>週末跳空風險</b>：今日為週五，切忌滿倉過週末！")

    # =========================
    # 組合 Telegram 訊息
    # =========================
    p_close_str = f"{price.close:.3f}" if price.close is not None else "NA"
    p_ema20_str = f"{price.ema20:.3f}" if price.ema20 is not None else "NA"
    p_kc_up_str = f"{price.kc_upper:.3f}" if price.kc_upper is not None else "NA"
    p_kc_dn_str = f"{price.kc_lower:.3f}" if price.kc_lower is not None else "NA"
    p_atr_str = f"{price.atr14:.3f}" if price.atr14 is not None else "NA"
    p_rsi_str = f"{price.rsi14:.1f}" if price.rsi14 is not None and not np.isnan(price.rsi14) else "NA"
    p_vol_str = f"{price.vol10*100:.1f}%" if price.vol10 is not None and not np.isnan(price.vol10) else "NA"
    p_short_str = f"{price.short_float_pct:.1f}%" if price.short_float_pct is not None else "NA"

    lines = [
        f"📌 <b>NG Composite Update ({run_date})</b>",
        f"• Run: <b>{run_tag}</b>",
        ""
    ]
    
    if alerts:
        lines.append("🚨 <b>系統特別警示 (ALERTS)</b>")
        lines.extend(alerts)
        lines.append("")

    lines.append(f"🌡️ <b>Weather Demand</b> (base {BASE_F:.0f}F)")
    if show_hdd:
        lines.append(f"• HDD 15D/30D: <b>{m['hdd_15d']:.1f}</b> / {m['hdd_30d']:.1f}")
    if show_cdd:
        lines.append(f"• CDD 15D/30D: <b>{m['cdd_15d']:.1f}</b> / {m['cdd_30d']:.1f}")
    lines.append("")

    lines.append("🧊/🔥 <b>Forecast Revision (7D)</b>")
    if show_hdd:
        lines.append(f"• HDD: <b>{f['hdd_fut7']:.1f}</b> ({fmt_arrow(d_hdd_fut7)} {d_hdd_fut7:+.1f})")
    if show_cdd:
        lines.append(f"• CDD: <b>{f['cdd_fut7']:.1f}</b> ({fmt_arrow(d_cdd_fut7)} {d_cdd_fut7:+.1f})")
    lines.append("")

    if storage.week and storage.total_bcf is not None:
        wow_str = f"{storage.wow_bcf:+.0f}" if storage.wow_bcf is not None else "NA"
        lines.extend([
            "🧱 <b>Storage (EIA Total)</b>",
            f"• Week: {storage.week} | Total: {storage.total_bcf:.0f} bcf",
        ])
        if storage.wow_5yr_avg is not None:
            diff = storage.wow_bcf - storage.wow_5yr_avg
            lines.append(f"• WoW: <b>{wow_str} bcf</b> (vs 5Yr Avg: {storage.wow_5yr_avg:+.0f} bcf)")
            lines.append(f"• Miss/Beat: <b>{diff:+.0f} bcf</b>")
        else:
            lines.append(f"• WoW: {wow_str} bcf | Bias: <b>{storage.bias}</b>")
        lines.append("")
    else:
        lines.extend(["🧱 <b>Storage</b>: NA", f"• Note: {storage.note}", ""])

    lines.extend([
        "🛡️ <b>Macro Risk (Oil / VIX)</b>",
        f"• WTI: <b>{macro.oil_change_pct*100:+.1f}%</b> | VIX: <b>{macro.vix_change_pct*100:+.1f}%</b>",
        f"• War Risk Triggered: <b>{'YES ☢️' if macro.is_war_risk_high else 'NO 🟢'}</b>",
        "",
        f"📈 <b>Price & Sentiment</b> ({price.symbol})",
        f"• Close: <b>{p_close_str}</b> | EMA20: {p_ema20_str}",
        f"• KC 軌道: <b>{p_kc_up_str}</b> / <b>{p_kc_dn_str}</b> (ATR: {p_atr_str})",
        f"• RSI14: {p_rsi_str} | Vol10: {p_vol_str}",
        f"• Short Float (空單佔比): <b>{p_short_str}</b>",
        ""
    ])

    if cot.net_managed_money is not None:
        cot_wow_str = f"{cot.net_wow_change:+.0f}" if cot.net_wow_change is not None else "NA"
        ls_ratio_str = f"{cot.ls_ratio:.2f}" if cot.ls_ratio is not None else "NA"
        lines.extend([
            "🏛️ <b>COT (CFTC Managed Money)</b>",
            f"• Net Pos: <b>{cot.net_managed_money:.0f}</b> (WoW: {cot_wow_str})",
            f"• L/S Ratio: <b>{ls_ratio_str}</b> (L: {cot.mm_long:.0f} / S: {cot.mm_short:.0f})",
            ""
        ])
    else:
        lines.extend(["🏛️ <b>COT Positioning</b>: NA (Fetch Error)", f"• Note: {cot.note}", ""])

    lines.extend([
        "🧮 <b>Score</b> (W / S / P / C / M)",
        f"• {w_score} / {s_score} / {p_score} / {c_score} / {m_score}  → Total: <b>{total_score}</b>",
        "",
        f"🎯 <b>Signal</b>: <b>{signal}</b>",
        f"🕒 Updated: {run_ts}"
    ])

    msg = "\n".join(lines)
    tg_send_message(TG_BOT_TOKEN, TG_CHAT_ID, msg)
    tg_send_photo(TG_BOT_TOKEN, TG_CHAT_ID, CHART_PATH, caption=f"📈 NG Composite Chart · {run_date}")
    print("[OK] Done.")

if __name__ == "__main__":
    run()
