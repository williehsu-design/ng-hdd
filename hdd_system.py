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
    note: str = ""

@dataclass
class COTInfo:
    net_managed_money: Optional[float] = None
    note: str = "disabled"

def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def fmt_arrow(delta: float) -> str:
    if delta > 0.1: return "⬆️"
    if delta < -0.1: return "⬇️"
    return "➖"

def retry_get(url: str, params: dict = None, headers: dict = None, tries: int = 3, timeout: int = 25) -> requests.Response:
    for i in range(tries):
        try:
            r = requests.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
            if r.status_code == 200: return r
        except: time.sleep(2)
    return None

def tg_send_message(token, chat_id, text):
    if not token or not chat_id: return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=25)

def tg_send_photo(token, chat_id, photo_path, caption):
    if not token or not chat_id or not os.path.exists(photo_path): return
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    with open(photo_path, "rb") as f:
        requests.post(url, data={"chat_id": chat_id, "caption": caption}, files={"photo": f}, timeout=40)

# =========================
# DATA FETCHING
# =========================
def fetch_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "daily": "temperature_2m_mean", "temperature_unit": "fahrenheit", "timezone": "UTC", "past_days": 14, "forecast_days": 16}
    r = retry_get(url, params=params)
    if not r: return pd.DataFrame()
    d = r.json().get("daily", {})
    df = pd.DataFrame({"date": pd.to_datetime(d.get("time", [])), "temp": d.get("temperature_2m_mean", [])})
    df["hdd"] = np.maximum(0, BASE_F - df["temp"])
    df["cdd"] = np.maximum(0, df["temp"] - BASE_F)
    df["date"] = df["date"].dt.tz_localize(None)
    return df

def fetch_storage_eia_v2(api_key):
    url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
    params = {"api_key": api_key, "frequency": "weekly", "data[0]": "value", "facets[series][]": "NW_NW_SWO_NG_R48_BCF", "sort[0][column]": "period", "sort[0][direction]": "desc", "length": 2}
    r = retry_get(url, params=params)
    if not r: return StorageInfo(note="EIA Fail")
    data = r.json().get("response", {}).get("data", [])
    if len(data) < 2: return StorageInfo(note="No data")
    curr, prev = float(data[0]["value"]), float(data[1]["value"])
    return StorageInfo(week=data[0]["period"], total_bcf=curr, wow_bcf=curr - prev, bias="DRAW" if curr < prev else "BUILD")

def fetch_price():
    try:
        import yfinance as yf
        df = yf.download(PRICE_SYMBOL_PRIMARY, period="1mo", interval="1d", progress=False)
        if df.empty: return PriceInfo("NA", "NA")
        close = df["Close"].iloc[-1]
        ma20 = df["Close"].tail(20).mean()
        return PriceInfo("YF", PRICE_SYMBOL_PRIMARY, float(close), float(ma20))
    except: return PriceInfo("NA", "NA")

# =========================
# MAIN RUN
# =========================
def run():
    now = utc_now()
    run_date = now.strftime("%Y-%m-%d")
    wdf = fetch_weather(LAT, LON)
    if wdf.empty: return
    
    today = pd.Timestamp(now.date())
    fut7 = wdf[wdf["date"] >= today].head(7)
    hdd_f7, cdd_f7 = float(fut7["hdd"].sum()), float(fut7["cdd"].sum())
    
    storage = fetch_storage_eia_v2(EIA_API_KEY)
    price = fetch_price()
    
    # 計算分數
    w_score = 2 if hdd_f7 > 40 else -2 if hdd_f7 < 15 else 0
    s_score = 2 if (storage.wow_bcf or 0) < -100 else -2 if (storage.wow_bcf or 0) > 0 else 0
    p_score = 1 if (price.close or 0) > (price.ma20 or 0) else -1
    total = w_score + s_score + p_score
    sig = "BOIL (多)" if total >= 2 else "KOLD (空)" if total <= -2 else "觀望"

    # 繪圖
    plt.figure(figsize=(10, 4))
    plt.plot(wdf["date"].tail(20), wdf["hdd"].tail(20), label="HDD")
    plt.plot(wdf["date"].tail(20), wdf["cdd"].tail(20), label="CDD")
    plt.legend(); plt.title(f"NG Weather Trend {run_date}"); plt.savefig(CHART_PATH); plt.close()

    # Telegram 訊息
    msg = (
        f"📌 <b>天然氣監控報告 ({run_date})</b>\n\n"
        f"🌡️ <b>氣溫預報 (未來7天)</b>\n"
        f"• HDD 加總: <b>{hdd_f7:.1f}</b>\n"
        f"• CDD 加總: <b>{cdd_f7:.1f}</b>\n\n"
        f"🧱 <b>庫存數據 (EIA 全美)</b>\n"
        f"• 總量: <b>{storage.total_bcf if storage.total_bcf else 'N/A'} Bcf</b>\n"
        f"• 週變動: <b>{storage.wow_bcf if storage.wow_bcf else 'N/A'} Bcf</b>\n\n"
        f"📈 <b>價格資訊</b>\n"
        f"• 目前價格: <b>{price.close:.3f if price.close else 'N/A'}</b>\n"
        f"• 20日均線: <b>{price.ma20:.3f if price.ma20 else 'N/A'}</b>\n\n"
        f"🧮 <b>評分系統 (各項得分)</b>\n"
        f"• 天氣: {w_score} | 庫存: {s_score} | 價格: {p_score}\n"
        f"• 總分: <b>{total}</b>\n\n"
        f"🎯 <b>建議操作: {sig}</b>"
    )
    
    tg_send_message(TG_BOT_TOKEN, TG_CHAT_ID, msg)
    tg_send_photo(TG_BOT_TOKEN, TG_CHAT_ID, CHART_PATH, f"NG Chart {run_date}")

if __name__ == "__main__":
    run()
