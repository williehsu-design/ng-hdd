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
# 用單點（例：NYC）或你想改成多城市 composite 也行；先穩定版保持單點。
LAT = 40.7128
LON = -74.0060
BASE_F = 65.0

CSV_PATH = "ng_composite_data.csv"
CHART_PATH = "ng_composite_chart.png"

PAST_DAYS = 14
FORECAST_DAYS = 16  # past+forecast+today 約31天

PRICE_SYMBOL_PRIMARY = "NG=F"
PRICE_SYMBOL_FALLBACK = "UNG"
FRED_SERIES_FALLBACK = "DHHNGSP"  # Henry Hub Spot (FRED)

# =========================
# ENV (GitHub Secrets)
# =========================
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "").strip()

EIA_API_KEY = os.getenv("EIA_API_KEY", "").strip()

# Nasdaq Data Link (Quandl) — 商用穩定版預設不抓
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
    if delta > 0.001:
        return "⬆️"
    if delta < -0.001:
        return "⬇️"
    return "➖"

def retry_get(url: str, params: dict = None, headers: dict = None, tries: int = 3, timeout: int = 25) -> requests.Response:
    last_err = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
            if r.status_code >= 400:
                raise requests.HTTPError(f"{r.status_code} {r.reason}: {r.text[:200]}", response=r)
            return r
        except Exception as e:
            last_err = e
            sleep_s = 1.0 + i * 0.8
            print(f"[WARN] HTTP attempt {i+1}/{tries} failed: {e}. sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"HTTP request failed after {tries} tries: {last_err}")

def safe_float_list(xs: List) -> List[float]:
    # Open-Meteo 偶爾會丟 None，這裡轉成 NaN 再補值，避免你之前那個 TypeError
    out = []
    for x in xs:
        if x is None:
            out.append(float("nan"))
        else:
            try:
                out.append(float(x))
            except Exception:
                out.append(float("nan"))
    return out

def tg_send_message(token: str, chat_id: str, text: str) -> None:
    if not token or not chat_id:
        print("[INFO] Telegram secrets missing; skip message.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    r = requests.post(url, data=payload, timeout=25)
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
        r = requests.post(url, data=data, files=files, timeout=40)
    if r.status_code >= 400:
        raise RuntimeError(f"Telegram sendPhoto failed: {r.status_code} {r.text[:200]}")

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

# =========================
# WEATHER / HDD / CDD
# =========================
def fetch_daily_mean_f(lat: float, lon: float, past_days: int, forecast_days: int) -> Tuple[List[str], List[float]]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_mean",
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
        "past_days": past_days,
        "forecast_days": forecast_days,
    }
    r = retry_get(url, params=params, tries=3, timeout=25)
    data = r.json()
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    temps = daily.get("temperature_2m_mean", [])
    if not dates or not temps or len(dates) != len(temps):
        raise RuntimeError(f"Open-Meteo returned invalid payload: {json.dumps(daily)[:300]}")

    temps_f = safe_float_list(list(temps))
    return list(dates), temps_f

def compute_hdd_cdd_series(dates: List[str], temps_f: List[float], base_f: float) -> pd.DataFrame:
    df = pd.DataFrame({"date": pd.to_datetime(dates), "tmean_f": temps_f}).sort_values("date").reset_index(drop=True)

    # 補值：前後補 + 線性插值，避免 NaN 造成後續不穩
    df["tmean_f"] = df["tmean_f"].astype(float)
    df["tmean_f"] = df["tmean_f"].interpolate(limit_direction="both").ffill().bfill()

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
        raise RuntimeError(f"Not enough days to compute 30D metrics (need 30, got {len(last30)})")

    return {
        "hdd_15d": weighted_avg_recent(last15["hdd"].to_numpy()),
        "hdd_30d": weighted_avg_recent(last30["hdd"].to_numpy()),
        "cdd_15d": weighted_avg_recent(last15["cdd"].to_numpy()),
        "cdd_30d": weighted_avg_recent(last30["cdd"].to_numpy()),
    }

def fut_sums(df: pd.DataFrame) -> Dict[str, float]:
    # 1. 取得「現在」的 UTC 日期，並轉成 Timestamp (去除時分秒)
    # 這樣才能跟 DataFrame 裡的 date 欄位 (格式通常是 YYYY-MM-DD 00:00:00) 對齊
    now_utc = dt.datetime.now(dt.timezone.utc)
    current_date = pd.Timestamp(now_utc.date())

    # 2. 篩選：只保留「今天」以後（含今天）的資料
    # 原本錯誤寫法：df["date"] > df["date"].max() (這樣會找不到任何資料)
    fut = df[df["date"] >= current_date].copy()
    
    # 重新排序確保順序正確
    fut = fut.sort_values("date").reset_index(drop=True)

    # 3. 如果篩選後沒資料（例如日期格式對不上），回傳 0.0 避免報錯，但在 Log 警告
    if fut.empty:
        print(f"[WARN] fut_sums found no future data! Check date formats. Current: {current_date}, DF Head: {df['date'].head(1)}")
        return {
            "hdd_fut7": 0.0, "hdd_fut15": 0.0,
            "cdd_fut7": 0.0, "cdd_fut15": 0.0,
        }

    # 4. 取未來 7 天與 15 天
    fut7 = fut.head(7)
    fut15 = fut.head(15)

    return {
        "hdd_fut7": float(fut7["hdd"].sum()),
        "hdd_fut15": float(fut15["hdd"].sum()),
        "cdd_fut7": float(fut7["cdd"].sum()),
        "cdd_fut15": float(fut15["cdd"].sum()),
    }

# =========================
# STORAGE (EIA v2)
# =========================
@dataclass
class StorageInfo:
    week: Optional[str] = None
    total_bcf: Optional[float] = None
    wow_bcf: Optional[float] = None
    bias: str = "NA"
    note: str = ""

def fetch_storage_eia_v2(api_key: str) -> StorageInfo:
    if not api_key:
        return StorageInfo(note="EIA_API_KEY not set (storage skipped)")

    url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
    # 取最近兩筆，算 WoW
    params = {
        "api_key": api_key,
        "frequency": "weekly",
        "data[0]": "value",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 2,
    }

    try:
        r = retry_get(url, params=params, tries=3, timeout=25)
        j = r.json()
        rows = j.get("response", {}).get("data", []) or []
        if len(rows) < 1:
            return StorageInfo(note="EIA v2 response empty")

        week = str(rows[0].get("period", "")) or None
        v0 = rows[0].get("value", None)
        total = float(v0) if v0 is not None else None

        wow = None
        if len(rows) >= 2 and rows[1].get("value", None) is not None and total is not None:
            wow = float(total - float(rows[1]["value"]))

        bias = "NA"
        if wow is not None:
            if wow < 0:
                bias = "DRAW"
            elif wow > 0:
                bias = "BUILD"
            else:
                bias = "FLAT"

        return StorageInfo(week=week, total_bcf=total, wow_bcf=wow, bias=bias, note="ok")
    except Exception as e:
        return StorageInfo(note=f"EIA v2 failed: {e}")

# =========================
# PRICE (yfinance -> FRED fallback)
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

def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.rolling(period).mean()
    loss = down.rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

def compute_vol(series: pd.Series, window: int = 10) -> float:
    ret = series.pct_change().dropna()
    if len(ret) < window + 1:
        return float("nan")
    return float(ret.tail(window).std() * np.sqrt(252))

def fetch_price_yfinance(symbol: str) -> Optional[pd.Series]:
    try:
        import yfinance as yf
        df = yf.download(symbol, period="3mo", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        close = df["Close"].dropna()
        if close.empty:
            return None
        return close
    except Exception:
        return None

def fetch_price_fred(series_id: str) -> Optional[pd.Series]:
    try:
        from pandas_datareader import data as pdr
        df = pdr.DataReader(series_id, "fred", start=dt.date.today() - dt.timedelta(days=120))
        if df is None or df.empty:
            return None
        s = df[series_id].dropna()
        if s.empty:
            return None
        return s
    except Exception:
        return None

def build_price_info() -> PriceInfo:
    close = fetch_price_yfinance(PRICE_SYMBOL_PRIMARY)
    if close is None:
        close = fetch_price_yfinance(PRICE_SYMBOL_FALLBACK)

    if close is not None and len(close) >= 25:
        ma20 = float(close.tail(20).mean())
        rsi = compute_rsi(close, 14) if len(close) >= 15 else float("nan")
        vol = compute_vol(close, 10)
        return PriceInfo(
            source="YF",
            symbol=PRICE_SYMBOL_PRIMARY,
            close=float(close.iloc[-1]),
            ma20=ma20,
            rsi14=rsi if np.isfinite(rsi) else None,
            vol10=vol if np.isfinite(vol) else None,
            note="ok",
        )

    # fallback: FRED
    fred = fetch_price_fred(FRED_SERIES_FALLBACK)
    if fred is not None and len(fred) >= 25:
        ma20 = float(fred.tail(20).mean())
        rsi = compute_rsi(fred, 14) if len(fred) >= 15 else float("nan")
        vol = compute_vol(fred, 10)
        return PriceInfo(
            source="FRED",
            symbol=f"FRED:{FRED_SERIES_FALLBACK}",
            close=float(fred.iloc[-1]),
            ma20=ma20,
            rsi14=rsi if np.isfinite(rsi) else None,
            vol10=vol if np.isfinite(vol) else None,
            note="ok",
        )

    return PriceInfo(source="NA", symbol="NA", note="price unavailable (yfinance + FRED failed)")

# =========================
# COT (OPTIONAL / SAFE)
# =========================
@dataclass
class COTInfo:
    net_managed_money: Optional[float] = None
    note: str = "disabled"

def fetch_cot_quandl(dataset_code: str, api_key: str) -> COTInfo:
    # 商用版：就算失敗也只能回 NA，不准讓整個 job 掛
    if not api_key:
        return COTInfo(note="QUANDL_API_KEY not set")
    
    url = f"https://data.nasdaq.com/api/v3/datasets/{dataset_code}.json"
    params = {"api_key": api_key, "rows": 5}
    
    # [修正1] 加入 User-Agent 避免被擋 (403 Forbidden)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        # [修正2] 這裡傳入 headers
        r = retry_get(url, params=params, headers=headers, tries=3, timeout=25)
        j = r.json()
        ds = j.get("dataset", {})
        cols = ds.get("column_names", [])
        data = ds.get("data", [])
        if not cols or not data:
            return COTInfo(note="Nasdaq COT empty")

        df = pd.DataFrame(data, columns=cols)
        # 嘗試用欄位名稱猜 Managed Money net（不同 dataset 欄位可能不一樣）
        cand_cols = [c for c in df.columns if "Managed" in c and "Net" in c]
        if cand_cols:
            net = float(df.iloc[0][cand_cols[0]])
            return COTInfo(net_managed_money=net, note="ok (nasdaq)")
        return COTInfo(note="Nasdaq COT: cannot find managed money net col")
    except Exception as e:
        # [修正3] 淨化錯誤訊息，避免 Telegram 崩潰
        err_msg = str(e)
        if "<html" in err_msg.lower() or "403" in err_msg:
            # 如果是網頁代碼，縮短訊息
            return COTInfo(note="Nasdaq access denied (403)")
        
        # 只取前 50 個字，避免訊息太長
        return COTInfo(note=f"Nasdaq COT failed: {err_msg[:50]}")
# =========================
# CHART
# =========================
def make_chart(weather_df: pd.DataFrame, run_tag: str, out_path: str) -> None:
    last30 = weather_df.tail(30).copy()
    if last30.empty:
        return

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
# SIGNAL (SIMPLE & STABLE)
# =========================
def regime_from_month(m: int) -> str:
    return "WINTER" if m in [11, 12, 1, 2, 3] else "SUMMER" if m in [6, 7, 8, 9] else "SHOULDER"

def score_system(
    d_hdd_fut7: float,
    storage: StorageInfo,
    price: PriceInfo,
    cot: COTInfo,
) -> Tuple[int, int, int, int, int, str]:
    # Weather score: forecast revision (fut7) only, avoid 15/30 disagreement noise
    w = 0
    if d_hdd_fut7 > 0.1:
        w = +2
    elif d_hdd_fut7 < -0.1:
        w = -2

    # Storage score: WoW draw/build
    s = 0
    if storage.wow_bcf is not None:
        if storage.wow_bcf < -10:
            s = +2
        elif storage.wow_bcf > 10:
            s = -2

    # Price score: close vs MA20
    p = 0
    if price.close is not None and price.ma20 is not None:
        p = +2 if price.close > price.ma20 else -2

    # COT score (optional)
    c = 0
    if cot.net_managed_money is not None:
        c = +1 if cot.net_managed_money > 0 else -1

    total = w + s + p + c

    if total >= 4:
        sig = "BOIL LONG (2–5D)"
    elif total <= -4:
        sig = "KOLD LONG (2–5D)"
    else:
        sig = "WAIT"

    return w, s, p, c, total, sig

# =========================
# MAIN
# =========================
def run():
    now = utc_now()
    run_date = now.strftime("%Y-%m-%d")
    run_ts = fmt_utc(now)

    run_tag = "AM (UTC)" if now.hour < 12 else "PM (UTC)"
    regime = regime_from_month(now.month)

    print(f"[INFO] ENV present: TG_BOT_TOKEN={'yes' if TG_BOT_TOKEN else 'no'} TG_CHAT_ID={'yes' if TG_CHAT_ID else 'no'} EIA_API_KEY={'yes' if EIA_API_KEY else 'no'} QUANDL_API_KEY={'yes' if QUANDL_API_KEY else 'no'} ENABLE_COT={'yes' if ENABLE_COT else 'no'}")

    # Weather
    dates, temps = fetch_daily_mean_f(LAT, LON, PAST_DAYS, FORECAST_DAYS)
    wdf = compute_hdd_cdd_series(dates, temps, BASE_F)

    m = compute_15_30(wdf)
    f = fut_sums(wdf)

    # history (prior same run_tag)
    hist = load_csv(CSV_PATH)
    prev = None
    if not hist.empty and "run_tag" in hist.columns:
        same = hist[hist["run_tag"] == run_tag]
        if not same.empty:
            prev = same.iloc[-1].to_dict()

    def prevv(key: str, fallback: float) -> float:
        if prev and key in prev and pd.notna(prev[key]):
            try:
                return float(prev[key])
            except Exception:
                return fallback
        return fallback

    d_hdd_fut7 = f["hdd_fut7"] - prevv("hdd_fut7", f["hdd_fut7"])
    d_hdd_fut15 = f["hdd_fut15"] - prevv("hdd_fut15", f["hdd_fut15"])
    d_cdd_fut7 = f["cdd_fut7"] - prevv("cdd_fut7", f["cdd_fut7"])
    d_cdd_fut15 = f["cdd_fut15"] - prevv("cdd_fut15", f["cdd_fut15"])

    # Storage
    storage = fetch_storage_eia_v2(EIA_API_KEY)
    print(f"[INFO] Storage note: {storage.note}")

    # Price
    price = build_price_info()
    print(f"[INFO] Price source: {price.symbol} ({price.note})")

    # COT (optional, safe)
    cot = COTInfo(note="disabled")
    if ENABLE_COT:
        cot = fetch_cot_quandl(COT_DATASET_CODE, QUANDL_API_KEY)

    # Score & Signal
    w_score, s_score, p_score, c_score, total_score, signal = score_system(d_hdd_fut7, storage, price, cot)

    # Volatility label
    vol_warn = ""
    if price.vol10 is not None and price.vol10 >= 0.30:
        vol_warn = f"⚠️ Volatility high (10D σ={price.vol10*100:.2f}%)"

    # Save CSV
    row = {
        "run_utc": run_ts,
        "date_utc": run_date,
        "run_tag": run_tag,
        "regime": regime,
        "hdd_15d": round(m["hdd_15d"], 2),
        "hdd_30d": round(m["hdd_30d"], 2),
        "cdd_15d": round(m["cdd_15d"], 2),
        "cdd_30d": round(m["cdd_30d"], 2),
        "hdd_fut7": round(f["hdd_fut7"], 1),
        "hdd_fut15": round(f["hdd_fut15"], 1),
        "cdd_fut7": round(f["cdd_fut7"], 1),
        "cdd_fut15": round(f["cdd_fut15"], 1),
        "delta_hdd_fut7": round(d_hdd_fut7, 2),
        "delta_hdd_fut15": round(d_hdd_fut15, 2),
        "delta_cdd_fut7": round(d_cdd_fut7, 2),
        "delta_cdd_fut15": round(d_cdd_fut15, 2),
        "storage_week": storage.week or "",
        "storage_total_bcf": storage.total_bcf if storage.total_bcf is not None else "",
        "storage_wow_bcf": storage.wow_bcf if storage.wow_bcf is not None else "",
        "storage_bias": storage.bias,
        "price_symbol": price.symbol,
        "price_close": price.close if price.close is not None else "",
        "price_ma20": price.ma20 if price.ma20 is not None else "",
        "price_rsi14": price.rsi14 if price.rsi14 is not None else "",
        "price_vol10": price.vol10 if price.vol10 is not None else "",
        "cot_net_managed_money": cot.net_managed_money if cot.net_managed_money is not None else "",
        "score_weather": w_score,
        "score_storage": s_score,
        "score_price": p_score,
        "score_cot": c_score,
        "score_total": total_score,
        "signal": signal,
        "notes": f"storage_note={storage.note}; price_note={price.note}; cot_note={cot.note}",
    }
    append_row(CSV_PATH, row)

    # Chart
    make_chart(wdf, f"{run_date} · {run_tag}", CHART_PATH)

    # Telegram message
    lines = []
    lines.append(f"📌 <b>NG Composite Update ({run_date})</b>")
    lines.append(f"• Run: <b>{run_tag}</b>  | Regime: <b>{regime}</b>")
    if vol_warn:
        lines.append(vol_warn)
    lines.append("")
    lines.append(f"🌡️ <b>Composite HDD/CDD</b> (base {BASE_F:.0f}F)")
    lines.append(f"• HDD 15D: <b>{m['hdd_15d']:.2f}</b> | 30D: <b>{m['hdd_30d']:.2f}</b>")
    lines.append(f"• CDD 15D: <b>{m['cdd_15d']:.2f}</b> | 30D: <b>{m['cdd_30d']:.2f}</b>")
    lines.append("")
    lines.append("🧊/🔥 <b>Forecast Revision</b> (vs prior same run-tag)")
    lines.append(f"• HDD Fut7: <b>{f['hdd_fut7']:.1f}</b> ({fmt_arrow(d_hdd_fut7)} {d_hdd_fut7:+.2f}) | Fut15: <b>{f['hdd_fut15']:.1f}</b> ({fmt_arrow(d_hdd_fut15)} {d_hdd_fut15:+.2f})")
    lines.append(f"• CDD Fut7: <b>{f['cdd_fut7']:.1f}</b> ({fmt_arrow(d_cdd_fut7)} {d_cdd_fut7:+.2f}) | Fut15: <b>{f['cdd_fut15']:.1f}</b> ({fmt_arrow(d_cdd_fut15)} {d_cdd_fut15:+.2f})")
    lines.append("")

    if storage.week and storage.total_bcf is not None:
        lines.append("🧱 <b>Storage</b> (EIA · Lower 48 Total)")
        wow_str = "NA" if storage.wow_bcf is None else f"{storage.wow_bcf:+.0f} bcf"
        lines.append(f"• Week: {storage.week} | Total: {storage.total_bcf:.0f} bcf")
        lines.append(f"• WoW: {wow_str} | Bias: <b>{storage.bias}</b>")
        lines.append("")
    else:
        lines.append("🧱 <b>Storage</b>: NA")
        lines.append(f"• Note: {storage.note}")
        lines.append("")

    if price.close is not None and price.ma20 is not None:
        lines.append(f"📈 <b>Price</b> ({price.symbol})")
        lines.append(f"• Close: {price.close:.3f} | MA20: {price.ma20:.3f}")
        if price.rsi14 is not None:
            lines.append(f"• RSI14: {price.rsi14:.1f}")
        if price.vol10 is not None:
            lines.append(f"• Vol10: {price.vol10*100:.2f}%")
        lines.append("")
    else:
        lines.append(f"📈 <b>Price</b>: NA ({price.note})")
        lines.append("")

    lines.append("📊 <b>COT</b> (Managed Money)")
    if ENABLE_COT:
        if cot.net_managed_money is not None:
            lines.append(f"• Net: {cot.net_managed_money:+.0f} (contracts)")
        else:
            lines.append(f"• NA ({cot.note})")
    else:
        lines.append("• Disabled (stable mode)")
    lines.append("")

    lines.append("🧮 <b>Score</b> (Weather / Storage / Price / COT)")
    lines.append(f"• {w_score} / {s_score} / {p_score} / {c_score}  → Total: <b>{total_score}</b>")
    lines.append("")
    lines.append(f"🎯 <b>Signal</b>: {signal}")
    lines.append(f"🕒 Updated: {run_ts}")

    msg = "\n".join(lines)

    tg_send_message(TG_BOT_TOKEN, TG_CHAT_ID, msg)
    tg_send_photo(TG_BOT_TOKEN, TG_CHAT_ID, CHART_PATH, caption=f"📈 NG Composite Chart · {run_date} · {run_tag}")

    print("[OK] Done.")

if __name__ == "__main__":
    run()
