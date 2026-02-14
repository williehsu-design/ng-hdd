import os
import json
import time
import re
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from io import StringIO

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


# =========================
# CONFIG
# =========================
BASE_F = float(os.getenv("BASE_F", "65.0"))

CSV_PATH = os.getenv("CSV_PATH", "ng_hdd_data.csv")
CHART_PATH = os.getenv("CHART_PATH", "hdd_cdd_chart.png")

PAST_DAYS = int(os.getenv("PAST_DAYS", "14"))
FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", "16"))  # ~31 points total

EPS = float(os.getenv("EPS", "0.01"))  # treat tiny deltas as 0
VOL10_WARN = float(os.getenv("VOL10_WARN", "0.06"))  # 6% daily-return stdev

# Composite city basket (weights sum ~=1, we normalize anyway)
CITY_BASKET = [
    ("Chicago", 41.8781, -87.6298, 0.22),
    ("New York", 40.7128, -74.0060, 0.18),
    ("Boston", 42.3601, -71.0589, 0.12),
    ("Atlanta", 33.7490, -84.3880, 0.10),
    ("Dallas", 32.7767, -96.7970, 0.12),
    ("Denver", 39.7392, -104.9903, 0.12),
    ("Los Angeles", 34.0522, -118.2437, 0.14),
]

PRICE_FRED_SERIES = os.getenv("PRICE_FRED_SERIES", "DHHNGSP")  # Henry Hub spot (FRED)
COT_Z_EXTREME = float(os.getenv("COT_Z_EXTREME", "1.5"))

# Nasdaq Data Link COT dataset code (user can override)
# ⚠️ 不同帳號/方案可看到的 dataset 代碼可能不同，所以做成可配置。
COT_DATASET_CODE = os.getenv("COT_DATASET_CODE", "CFTC/023651_F_L_ALL")

# =========================
# ENV (GitHub Secrets)
# =========================
TG_BOT_TOKEN = (os.getenv("TG_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or "").strip()
TG_CHAT_ID = (os.getenv("TG_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID") or "").strip()
EIA_API_KEY = (os.getenv("EIA_API_KEY") or "").strip()
QUANDL_API_KEY = (os.getenv("QUANDL_API_KEY") or "").strip()


# =========================
# HELPERS
# =========================
def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def fmt_utc(ts: dt.datetime) -> str:
    return ts.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def _redact(s: str) -> str:
    s = re.sub(r"(api_key=)[^&\s]+", r"\1***", s)
    s = re.sub(r"(Authorization:\s*Bearer\s+)[^\s]+", r"\1***", s, flags=re.IGNORECASE)
    return s

def retry_get(url: str, params: dict, tries: int = 3, timeout: int = 25) -> requests.Response:
    last_err = None
    headers = {
        "User-Agent": "Mozilla/5.0 (GitHubActions; NGCompositeMonitor/4.0)",
        "Accept": "*/*",
    }
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=headers)
            if r.status_code >= 400:
                raise requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
            return r
        except Exception as e:
            last_err = e
            sleep_s = 1.0 + i * 0.8
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

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def run_tag_from_utc(now: dt.datetime) -> str:
    # 00:20 UTC => AM, 12:20 UTC => PM
    return "AM" if now.hour < 12 else "PM"


# =========================
# WEATHER (Open-Meteo) -> Composite HDD/CDD
# =========================
def fetch_daily_temps_f(lat: float, lon: float, past_days: int, forecast_days: int) -> Tuple[List[str], List[float]]:
    """
    Open-Meteo daily temps (F). Robust against nulls:
    mean -> (max+min)/2 -> forward/backward fill.
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
    r = retry_get(url, params=params, tries=3, timeout=25)
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

    # ffill then bfill
    last = None
    for i in range(n):
        if temps[i] is None and last is not None:
            temps[i] = last
        elif temps[i] is not None:
            last = temps[i]

    nxt = None
    for i in range(n - 1, -1, -1):
        if temps[i] is not None:
            nxt = temps[i]
        elif nxt is not None:
            temps[i] = nxt

    if any(v is None for v in temps):
        raise RuntimeError("Open-Meteo temps still contain nulls after fill")

    return dates, [float(x) for x in temps]

def compute_degree_days(dates: List[str], temps_f: List[float], base_f: float) -> pd.DataFrame:
    df = pd.DataFrame({"date": pd.to_datetime(dates), "tmean_f": temps_f})
    df = df.sort_values("date").reset_index(drop=True)
    df["hdd"] = np.maximum(0.0, base_f - df["tmean_f"])
    df["cdd"] = np.maximum(0.0, df["tmean_f"] - base_f)
    return df[["date", "hdd", "cdd"]]

def build_composite_weather() -> pd.DataFrame:
    weights = np.array([w for (_, _, _, w) in CITY_BASKET], dtype=float)
    weights = weights / weights.sum()

    per_city: List[pd.DataFrame] = []
    for (name, lat, lon, _) in CITY_BASKET:
        dates, temps = fetch_daily_temps_f(lat, lon, PAST_DAYS, FORECAST_DAYS)
        df = compute_degree_days(dates, temps, BASE_F)
        df.rename(columns={"hdd": f"hdd_{name}", "cdd": f"cdd_{name}"}, inplace=True)
        per_city.append(df)

    merged = per_city[0]
    for df in per_city[1:]:
        merged = merged.merge(df, on="date", how="inner")

    hdd_cols = [f"hdd_{name}" for (name, _, _, _) in CITY_BASKET]
    cdd_cols = [f"cdd_{name}" for (name, _, _, _) in CITY_BASKET]
    hdd_mat = merged[hdd_cols].to_numpy(dtype=float)
    cdd_mat = merged[cdd_cols].to_numpy(dtype=float)

    comp = pd.DataFrame({
        "date": merged["date"],
        "hdd": (hdd_mat * weights.reshape(1, -1)).sum(axis=1),
        "cdd": (cdd_mat * weights.reshape(1, -1)).sum(axis=1),
    }).sort_values("date").reset_index(drop=True)

    return comp

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

def future_window_sum(df: pd.DataFrame, today_utc_date: dt.date, days: int, col: str) -> float:
    dff = df.copy()
    dff["d"] = dff["date"].dt.date
    future = dff[dff["d"] > today_utc_date].sort_values("date")
    win = future.head(days)
    return float(win[col].sum())


# =========================
# STORAGE (EIA v2 seriesid) - Lower 48 Total Working Gas
# =========================
@dataclass
class StorageInfo:
    week: Optional[str] = None
    total_bcf: Optional[float] = None
    wow_bcf: Optional[float] = None
    bias: str = "NA"       # DRAW / BUILD / FLAT / NA
    vs5y_bcf: Optional[float] = None
    note: str = ""

def fetch_storage_series_v2(api_key: str, seriesid: str) -> List[dict]:
    url = f"https://api.eia.gov/v2/seriesid/{seriesid}"
    params = {"api_key": api_key}
    r = retry_get(url, params=params, tries=3, timeout=30)
    j = r.json()
    return j.get("response", {}).get("data", []) or []

def compute_5y_avg_same_week(data: List[dict], latest_week: str) -> Optional[float]:
    try:
        latest_date = dt.date.fromisoformat(latest_week)
    except Exception:
        return None

    target_week = latest_date.isocalendar().week
    target_year = latest_date.isocalendar().year

    rows = []
    for x in data:
        p = str(x.get("period", "")).strip()
        v = x.get("value", None)
        if not p or v is None:
            continue
        try:
            d = dt.date.fromisoformat(p)
        except Exception:
            continue
        iso = d.isocalendar()
        if iso.week != target_week:
            continue
        if iso.year == target_year:
            continue
        rows.append((d, float(v)))

    rows.sort(key=lambda t: t[0], reverse=True)
    vals = [v for (_, v) in rows[:5]]
    if len(vals) < 3:
        return None
    return float(np.mean(vals))

def fetch_storage_lower48_total(api_key: str) -> StorageInfo:
    if not api_key:
        return StorageInfo(note="EIA_API_KEY not set (storage skipped)")

    seriesid = "NG.NW2_EPG0_SWO_R48_BCF.W"
    try:
        data = fetch_storage_series_v2(api_key, seriesid)
        if not data:
            raise RuntimeError("v2/seriesid empty data")

        data_sorted = sorted(data, key=lambda x: str(x.get("period", "")), reverse=True)
        latest = data_sorted[0]
        prev = data_sorted[1] if len(data_sorted) > 1 else None

        week = str(latest.get("period", "")).strip() or None
        total = float(latest["value"]) if latest.get("value", None) is not None else None

        wow = None
        bias = "NA"
        if prev is not None and total is not None and prev.get("value", None) is not None:
            prev_total = float(prev["value"])
            wow = total - prev_total
            if abs(wow) < 0.5:
                bias = "FLAT"
            elif wow < 0:
                bias = "DRAW"
            else:
                bias = "BUILD"

        avg5 = compute_5y_avg_same_week(data_sorted, week) if week else None
        vs5 = (total - avg5) if (total is not None and avg5 is not None) else None

        return StorageInfo(week=week, total_bcf=total, wow_bcf=wow, bias=bias, vs5y_bcf=vs5, note="ok")
    except Exception as e:
        return StorageInfo(note=f"storage fetch failed: {_redact(str(e))}")


# =========================
# PRICE (FRED) + indicators
# =========================
@dataclass
class PriceInfo:
    symbol: str
    price: Optional[float] = None
    ma20: Optional[float] = None
    rsi14: Optional[float] = None
    vol10: Optional[float] = None
    note: str = ""

def calc_rsi(close: pd.Series, period: int = 14) -> Optional[float]:
    close = pd.to_numeric(close, errors="coerce").dropna()
    if len(close) < period + 2:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

def fetch_price_fred(series_id: str) -> PriceInfo:
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
        params = {"id": series_id}
        r = retry_get(url, params=params, tries=3, timeout=25)

        df = pd.read_csv(StringIO(r.text))
        if df is None or df.empty or df.shape[1] < 2:
            return PriceInfo(symbol=f"FRED:{series_id}", note="fred: empty")

        col = df.columns[1]
        close = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(close) < 30:
            return PriceInfo(symbol=f"FRED:{series_id}", note="fred: not enough history")

        last = float(close.iloc[-1])
        ma20 = float(close.tail(20).mean())
        rsi14 = calc_rsi(close, 14)

        ret = close.pct_change().dropna()
        vol10 = float(ret.tail(10).std()) if len(ret) >= 10 else None

        return PriceInfo(symbol=f"FRED:{series_id}", price=last, ma20=ma20, rsi14=rsi14, vol10=vol10, note="ok")
    except Exception as e:
        return PriceInfo(symbol=f"FRED:{series_id}", note=f"fred failed: {e}")


# =========================
# COT (Managed Money) - STABLE: Nasdaq Data Link -> CFTC fallback
# =========================
@dataclass
class COTInfo:
    net_position: Optional[float] = None
    zscore: Optional[float] = None
    score: int = 0
    note: str = ""
    source: str = "NA"

def cot_score_from_series(net_series: pd.Series) -> COTInfo:
    net_series = pd.to_numeric(net_series, errors="coerce").dropna()
    if len(net_series) < 60:
        return COTInfo(note="COT not enough history", source="parsed")

    latest = float(net_series.iloc[-1])
    window = net_series.tail(156)  # ~3y
    mean = float(window.mean())
    std = float(window.std()) if float(window.std()) > 0 else 1.0
    z = (latest - mean) / std

    score = 0
    if z < -COT_Z_EXTREME:
        score = +1
    elif z > COT_Z_EXTREME:
        score = -1

    return COTInfo(net_position=latest, zscore=float(z), score=score, note="ok", source="parsed")

def fetch_cot_nasdaq_datalink(dataset_code: str, api_key: str) -> COTInfo:
    """
    Nasdaq Data Link dataset endpoint:
      https://data.nasdaq.com/api/v3/datasets/{CODE}.json?api_key=...
    We attempt to detect columns containing Money Manager (Managed Money) Long/Short.
    """
    if not api_key:
        return COTInfo(note="QUANDL_API_KEY not set", source="nasdaq")

    # dataset_code like "CFTC/023651_F_L_ALL"
    url = f"https://data.nasdaq.com/api/v3/datasets/{dataset_code}.json"
    params = {"api_key": api_key}
    try:
        r = retry_get(url, params=params, tries=3, timeout=35)
        j = r.json()
        ds = j.get("dataset", {})
        data = ds.get("data", []) or []
        cols = ds.get("column_names", []) or []
        if not data or not cols:
            return COTInfo(note="Nasdaq dataset empty", source="nasdaq")

        df = pd.DataFrame(data, columns=cols)

        # find date column
        date_col = None
        for c in df.columns:
            if str(c).lower() in ("date", "report_date", "report_date_as_yyyy-mm-dd"):
                date_col = c
                break
        if date_col is None:
            date_col = df.columns[0]

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)

        # detect Managed Money / Money Manager long/short columns (most common patterns)
        def find_col(patterns: List[str]) -> Optional[str]:
            for c in df.columns:
                cl = str(c).lower()
                ok = True
                for p in patterns:
                    if p not in cl:
                        ok = False
                        break
                if ok:
                    return c
            return None

        long_col = (
            find_col(["money", "manager", "long"]) or
            find_col(["managed", "money", "long"]) or
            find_col(["money", "manager", "positions", "long"])
        )
        short_col = (
            find_col(["money", "manager", "short"]) or
            find_col(["managed", "money", "short"]) or
            find_col(["money", "manager", "positions", "short"])
        )

        if long_col is None or short_col is None:
            # dataset exists but schema different -> tell user which columns we saw
            sample_cols = ", ".join(map(str, list(df.columns)[:18]))
            return COTInfo(note=f"Nasdaq COT columns not found. sample_cols={sample_cols}", source="nasdaq")

        net = pd.to_numeric(df[long_col], errors="coerce") - pd.to_numeric(df[short_col], errors="coerce")
        info = cot_score_from_series(net)
        info.source = "nasdaq"
        return info

    except Exception as e:
        return COTInfo(note=f"Nasdaq COT failed: {_redact(str(e))}", source="nasdaq")

def fetch_cot_cftc_f_disagg() -> COTInfo:
    """
    Fallback: CFTC file (may be blocked on GitHub Actions sometimes).
    https://www.cftc.gov/dea/newcot/f_disagg.txt
    We parse NATURAL GAS and compute managed money net.
    """
    url = "https://www.cftc.gov/dea/newcot/f_disagg.txt"
    try:
        r = retry_get(url, params={}, tries=3, timeout=35)
        df = pd.read_csv(StringIO(r.text))

        # find market column
        market_col = None
        for c in df.columns:
            if "market" in str(c).lower() and "exchange" in str(c).lower():
                market_col = c
                break
        if market_col is None:
            return COTInfo(note="CFTC schema changed (no market column)", source="cftc")

        sub = df[df[market_col].astype(str).str.contains("NATURAL GAS", case=False, na=False)].copy()
        if sub.empty:
            return COTInfo(note="CFTC NG not found", source="cftc")

        # date col
        date_col = None
        for c in sub.columns:
            cl = str(c).lower()
            if "report_date" in cl and "as" in cl:
                date_col = c
                break
        if date_col is None:
            # try a generic report date column
            for c in sub.columns:
                if "report_date" in str(c).lower():
                    date_col = c
                    break
        if date_col is None:
            return COTInfo(note="CFTC schema changed (no date column)", source="cftc")

        # long/short cols for managed money
        def pick(patterns: List[str]) -> Optional[str]:
            for c in sub.columns:
                cl = str(c).lower()
                ok = True
                for p in patterns:
                    if p not in cl:
                        ok = False
                        break
                if ok:
                    return c
            return None

        long_col = pick(["managed", "money", "long"])
        short_col = pick(["managed", "money", "short"])
        if long_col is None or short_col is None:
            return COTInfo(note="CFTC schema changed (no managed money long/short)", source="cftc")

        sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
        sub[long_col] = pd.to_numeric(sub[long_col], errors="coerce")
        sub[short_col] = pd.to_numeric(sub[short_col], errors="coerce")
        sub = sub.dropna(subset=[date_col, long_col, short_col]).sort_values(date_col)

        net = sub[long_col] - sub[short_col]
        info = cot_score_from_series(net)
        info.source = "cftc"
        return info

    except Exception as e:
        return COTInfo(note=f"CFTC COT failed: {_redact(str(e))}", source="cftc")

def fetch_cot_managed_money() -> COTInfo:
    # 1) Nasdaq Data Link (stable) if key present
    if QUANDL_API_KEY:
        info = fetch_cot_nasdaq_datalink(COT_DATASET_CODE, QUANDL_API_KEY)
        if info.note == "ok":
            return info
        # If Nasdaq exists but cannot parse columns, still fallback to CFTC
        print(f"[WARN] Nasdaq COT not ok: {info.note}")

    # 2) CFTC fallback
    info2 = fetch_cot_cftc_f_disagg()
    if info2.note == "ok":
        return info2

    # 3) NA
    combined = f"nasdaq=({info.note if QUANDL_API_KEY else 'no_key'}); cftc=({info2.note})"
    return COTInfo(note=f"COT failed: {combined}", source="NA")


# =========================
# REGIME / EVENT WINDOW
# =========================
def season_regime(today: dt.date) -> Tuple[str, float]:
    m = today.month
    if m in (11, 12, 1, 2, 3):
        return ("WINTER", 1.0)
    if m in (6, 7, 8, 9):
        return ("SUMMER", 1.0)
    return ("SHOULDER", 0.6)

def is_eia_event_window(now_utc: dt.datetime) -> bool:
    # Thu 10:30 ET +/- ~1h
    if ZoneInfo is None:
        return False
    try:
        et = now_utc.astimezone(ZoneInfo("America/New_York"))
        if et.weekday() != 3:
            return False
        t = et.time()
        return dt.time(9, 45) <= t <= dt.time(11, 45)
    except Exception:
        return False


# =========================
# SCORING
# =========================
def score_weather(regime: str, rev7_hdd: float, rev15_hdd: float, rev7_cdd: float, rev15_cdd: float) -> int:
    w7 = _norm_delta(rev7_hdd); w15 = _norm_delta(rev15_hdd)
    c7 = _norm_delta(rev7_cdd); c15 = _norm_delta(rev15_cdd)

    if regime == "WINTER":
        if w7 > 0 and w15 > 0:
            return 2 if w7 > w15 else 1
        if w7 < 0 and w15 < 0:
            return -2 if w7 < w15 else -1
        return 0

    if regime == "SUMMER":
        if c7 > 0 and c15 > 0:
            return 2 if c7 > c15 else 1
        if c7 < 0 and c15 < 0:
            return -2 if c7 < c15 else -1
        return 0

    # shoulder
    if (w7 > 0 and w15 > 0) and (c7 <= 0 and c15 <= 0):
        return 1
    if (w7 < 0 and w15 < 0) and (c7 >= 0 and c15 >= 0):
        return -1
    return 0

def score_storage(s: StorageInfo) -> int:
    score = 0
    if s.wow_bcf is not None:
        if s.wow_bcf < -0.5:
            score += 1
        elif s.wow_bcf > 0.5:
            score -= 1
    if s.vs5y_bcf is not None:
        if s.vs5y_bcf < -1.0:
            score += 1
        elif s.vs5y_bcf > 1.0:
            score -= 1
    return score

def score_price(p: PriceInfo) -> int:
    if p.price is None or p.ma20 is None:
        return 0
    score = 0
    score += 1 if p.price > p.ma20 else -1
    if p.rsi14 is not None:
        if p.rsi14 > 55:
            score += 1
        elif p.rsi14 < 45:
            score -= 1
    return score

def score_cot(c: COTInfo) -> int:
    return int(c.score or 0)

def decide_signal(regime: str, total_score: int, event_risk: bool, vol_high: bool) -> Tuple[str, str]:
    strong = 3 if regime != "SHOULDER" else 4
    if event_risk and abs(total_score) < strong:
        return ("WAIT", "EIA event-risk window")
    if vol_high and abs(total_score) < strong:
        return ("WAIT", "high volatility filter")

    if total_score >= strong:
        return ("BOIL LONG (2–5D)", f"score {total_score}")
    if total_score <= -strong:
        return ("KOLD LONG (2–5D)", f"score {total_score}")

    return ("WAIT", f"score {total_score}")


# =========================
# IO / CHART / TELEGRAM
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

def make_chart(comp_df: pd.DataFrame, run_label: str, out_path: str) -> None:
    last30 = comp_df.tail(30).copy()
    if last30.empty:
        return
    fig = plt.figure(figsize=(10, 4.3))
    ax = plt.gca()
    ax.plot(last30["date"], last30["hdd"], label=f"Composite HDD (base {BASE_F:.0f}F)")
    ax.plot(last30["date"], last30["cdd"], label=f"Composite CDD (base {BASE_F:.0f}F)")
    ax.set_title(f"Composite HDD/CDD Trend · {run_label}")
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
        r = requests.post(url, data=data, files=files, timeout=35)
    if r.status_code >= 400:
        raise RuntimeError(f"Telegram sendPhoto failed: {r.status_code} {r.text[:200]}")


# =========================
# MAIN
# =========================
def find_prev_same_tag(hist: pd.DataFrame, tag: str) -> Optional[pd.Series]:
    if hist.empty or "run_tag" not in hist.columns:
        return None
    sub = hist[hist["run_tag"].astype(str) == str(tag)].copy()
    if sub.empty:
        return None
    return sub.iloc[-1]

def run():
    now = utc_now()
    run_date = now.strftime("%Y-%m-%d")
    run_ts = fmt_utc(now)
    tag = run_tag_from_utc(now)
    run_label = f"{run_date} {tag} UTC"
    today_utc = now.date()

    # Weather
    comp = build_composite_weather()
    m = compute_15_30(comp)

    # Future windows & revisions vs prior same tag
    fut7_hdd = future_window_sum(comp, today_utc, 7, "hdd")
    fut15_hdd = future_window_sum(comp, today_utc, 15, "hdd")
    fut7_cdd = future_window_sum(comp, today_utc, 7, "cdd")
    fut15_cdd = future_window_sum(comp, today_utc, 15, "cdd")

    hist = load_csv(CSV_PATH)
    prev_same = find_prev_same_tag(hist, tag)

    if prev_same is not None:
        prev_fut7_hdd = float(prev_same.get("fut7_hdd", fut7_hdd))
        prev_fut15_hdd = float(prev_same.get("fut15_hdd", fut15_hdd))
        prev_fut7_cdd = float(prev_same.get("fut7_cdd", fut7_cdd))
        prev_fut15_cdd = float(prev_same.get("fut15_cdd", fut15_cdd))
    else:
        prev_fut7_hdd, prev_fut15_hdd, prev_fut7_cdd, prev_fut15_cdd = fut7_hdd, fut15_hdd, fut7_cdd, fut15_cdd

    rev7_hdd = _norm_delta(fut7_hdd - prev_fut7_hdd)
    rev15_hdd = _norm_delta(fut15_hdd - prev_fut15_hdd)
    rev7_cdd = _norm_delta(fut7_cdd - prev_fut7_cdd)
    rev15_cdd = _norm_delta(fut15_cdd - prev_fut15_cdd)

    # Storage
    storage = fetch_storage_lower48_total(EIA_API_KEY)

    # Price
    price = fetch_price_fred(PRICE_FRED_SERIES)
    vol_high = (price.vol10 is not None and price.vol10 > VOL10_WARN)

    # COT
    cot = fetch_cot_managed_money()

    # Regime / event risk
    regime, w_mult = season_regime(today_utc)
    event_risk = is_eia_event_window(now)

    # Scores
    w_raw = score_weather(regime, rev7_hdd, rev15_hdd, rev7_cdd, rev15_cdd)
    w_score = int(round(w_raw * w_mult))
    s_score = score_storage(storage)
    p_score = score_price(price)
    c_score = score_cot(cot)
    total = w_score + s_score + p_score + c_score

    # Confidence (1..10)
    conf = 5.0 + 1.2 * abs(total)
    if event_risk:
        conf -= 1.5
    if vol_high:
        conf -= 1.0
    if regime == "SHOULDER":
        conf -= 0.5
    conf = clamp(conf, 1.0, 10.0)

    signal, sig_note = decide_signal(regime, total, event_risk, vol_high)

    # Save row
    row = {
        "run_utc": run_ts,
        "run_tag": tag,
        "date_utc": run_date,
        "regime": regime,
        "hdd_15d": round(m["hdd_15d"], 2),
        "hdd_30d": round(m["hdd_30d"], 2),
        "cdd_15d": round(m["cdd_15d"], 2),
        "cdd_30d": round(m["cdd_30d"], 2),
        "fut7_hdd": round(fut7_hdd, 2),
        "fut15_hdd": round(fut15_hdd, 2),
        "fut7_cdd": round(fut7_cdd, 2),
        "fut15_cdd": round(fut15_cdd, 2),
        "rev7_hdd": round(rev7_hdd, 2),
        "rev15_hdd": round(rev15_hdd, 2),
        "rev7_cdd": round(rev7_cdd, 2),
        "rev15_cdd": round(rev15_cdd, 2),
        "storage_week": storage.week or "",
        "storage_total_bcf": storage.total_bcf if storage.total_bcf is not None else "",
        "storage_wow_bcf": storage.wow_bcf if storage.wow_bcf is not None else "",
        "storage_vs5y_bcf": storage.vs5y_bcf if storage.vs5y_bcf is not None else "",
        "storage_bias": storage.bias,
        "price_symbol": price.symbol,
        "price": price.price if price.price is not None else "",
        "ma20": price.ma20 if price.ma20 is not None else "",
        "rsi14": price.rsi14 if price.rsi14 is not None else "",
        "vol10": price.vol10 if price.vol10 is not None else "",
        "cot_source": cot.source,
        "cot_net": cot.net_position if cot.net_position is not None else "",
        "cot_z": cot.zscore if cot.zscore is not None else "",
        "weather_score": w_score,
        "storage_score": s_score,
        "price_score": p_score,
        "cot_score": c_score,
        "total_score": total,
        "signal": signal,
        "confidence": f"{conf:.1f}/10",
        "notes": f"sig_note={sig_note}; storage_note={storage.note}; price_note={price.note}; cot_note={cot.note}; "
                 f"cot_dataset={COT_DATASET_CODE}",
    }
    append_row(CSV_PATH, row)

    # Chart
    make_chart(comp, run_label, CHART_PATH)

    # Telegram message
    lines = []
    lines.append(f"📌 <b>NG Composite Update ({run_date})</b>")
    lines.append(f"• Run: <b>{tag}</b> (UTC)  | Regime: <b>{regime}</b>")
    if vol_high and price.vol10 is not None:
        lines.append(f"⚠️ <b>Volatility high</b> (10D σ={price.vol10:.2%})")
    if event_risk:
        lines.append("⚠️ <b>EIA event-risk window</b>")
    lines.append("")

    lines.append(f"🌡️ <b>Composite HDD/CDD</b> (base {BASE_F:.0f}F)")
    lines.append(f"• HDD 15D: <b>{m['hdd_15d']:.2f}</b> | 30D: <b>{m['hdd_30d']:.2f}</b>")
    lines.append(f"• CDD 15D: <b>{m['cdd_15d']:.2f}</b> | 30D: <b>{m['cdd_30d']:.2f}</b>")
    lines.append("")

    lines.append("🧊/🔥 <b>Forecast Revision</b> (vs prior same run-tag)")
    lines.append(f"• HDD Fut7: <b>{fut7_hdd:.1f}</b> ({fmt_arrow(rev7_hdd)} {rev7_hdd:+.2f}) | "
                 f"Fut15: <b>{fut15_hdd:.1f}</b> ({fmt_arrow(rev15_hdd)} {rev15_hdd:+.2f})")
    lines.append(f"• CDD Fut7: <b>{fut7_cdd:.1f}</b> ({fmt_arrow(rev7_cdd)} {rev7_cdd:+.2f}) | "
                 f"Fut15: <b>{fut15_cdd:.1f}</b> ({fmt_arrow(rev15_cdd)} {rev15_cdd:+.2f})")
    lines.append("")

    if storage.week and storage.total_bcf is not None:
        lines.append("🧱 <b>Storage</b> (EIA · Lower 48 Total)")
        lines.append(f"• Week: {storage.week} | Total: <b>{storage.total_bcf:.0f}</b> bcf")
        if storage.wow_bcf is not None:
            lines.append(f"• WoW: {storage.wow_bcf:+.0f} bcf | Bias: <b>{storage.bias}</b>")
        if storage.vs5y_bcf is not None:
            lines.append(f"• vs 5Y(avg same wk): {storage.vs5y_bcf:+.0f} bcf")
        lines.append("")
    else:
        lines.append("🧱 <b>Storage</b>: NA")
        lines.append(f"• Note: {storage.note}")
        lines.append("")

    if price.price is not None and price.ma20 is not None:
        lines.append(f"📈 <b>Price</b> ({price.symbol})")
        lines.append(f"• Close: {price.price:.3f} | MA20: {price.ma20:.3f}")
        if price.rsi14 is not None:
            lines.append(f"• RSI14: {price.rsi14:.1f}")
        if price.vol10 is not None:
            lines.append(f"• Vol10: {price.vol10:.2%}")
        lines.append("")
    else:
        lines.append(f"📈 <b>Price</b>: NA ({price.note})")
        lines.append("")

    lines.append("📊 <b>COT (Managed Money)</b>")
    if cot.zscore is not None and cot.net_position is not None:
        lines.append(f"• Source: {cot.source} | Net: {cot.net_position:,.0f} | Z: {cot.zscore:.2f} | Score: {c_score:+d}")
    else:
        lines.append(f"• NA ({cot.note})")
    lines.append("")

    lines.append("🧮 <b>Score</b> (Weather / Storage / Price / COT)")
    lines.append(f"• {w_score} / {s_score} / {p_score} / {c_score}  → Total: <b>{total}</b>")
    lines.append("")
    lines.append(f"🎯 <b>Signal</b>: {signal}")
    lines.append(f"🧠 Confidence: {conf:.1f}/10")
    lines.append(f"🕒 Updated: {run_ts}")

    msg = "\n".join(lines)
    tg_send_message(TG_BOT_TOKEN, TG_CHAT_ID, msg)
    tg_send_photo(TG_BOT_TOKEN, TG_CHAT_ID, CHART_PATH, caption=f"📈 Composite HDD/CDD · {run_label}")

    print("[INFO] ENV present:",
          f"TG_BOT_TOKEN={'yes' if TG_BOT_TOKEN else 'no'}",
          f"TG_CHAT_ID={'yes' if TG_CHAT_ID else 'no'}",
          f"EIA_API_KEY={'yes' if EIA_API_KEY else 'no'}",
          f"QUANDL_API_KEY={'yes' if QUANDL_API_KEY else 'no'}")
    print(f"[INFO] COT dataset={COT_DATASET_CODE} source={cot.source} note={cot.note}")
    print("[OK] Done.")

if __name__ == "__main__":
    run()
