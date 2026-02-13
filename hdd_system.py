import os
import json
import time
import math
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import pytz
import requests
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Config
# =========================
LAT = 40.7128
LON = -74.0060

BASE_F = 65.0

# Open-Meteo limit: forecast_days max 16 (常見錯誤就是超過16)
FORECAST_DAYS = 16
PAST_DAYS = 30  # 拉過去30天 + 未來16天，足夠算15D/30D & 做趨勢圖

CSV_PATH = "ng_hdd_data.csv"
CHART_PATH = "hdd_cdd_chart.png"

# Storage (EIA WNGSR)
EIA_WNGSR_JSON = "https://ir.eia.gov/ngs/wngsr.json"

# Telegram
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "").strip()


# =========================
# Helpers
# =========================
def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_utc(ts: Optional[dt.datetime] = None) -> str:
    if ts is None:
        ts = utc_now()
    return ts.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def retry_get(url: str, params: Dict[str, Any], tries: int = 3, timeout: int = 25) -> requests.Response:
    last_err = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code >= 400:
                raise requests.HTTPError(f"{r.status_code} {r.reason}: {r.text[:200]}", response=r)
            return r
        except Exception as e:
            last_err = e
            sleep_s = 1.0 + (i * 0.7)
            print(f"[WARN] HTTP attempt {i+1}/{tries} failed: {e}. sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"HTTP request failed after {tries} tries: {last_err}")


def weighted_average(values: List[float]) -> float:
    """
    線性加權：越近的權重越高
    values[0] 是最舊，values[-1] 是最新
    """
    n = len(values)
    if n == 0:
        return 0.0
    weights = list(range(1, n + 1))
    s = sum(v * w for v, w in zip(values, weights))
    return s / sum(weights)


def compute_hdd_cdd_series(temps_f: List[float], base_f: float) -> Tuple[List[float], List[float]]:
    hdd = [max(0.0, base_f - t) for t in temps_f]
    cdd = [max(0.0, t - base_f) for t in temps_f]
    return hdd, cdd


def fmt_num(x: Optional[float], nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


# =========================
# Open-Meteo weather fetch
# =========================
def fetch_daily_mean_f(lat: float, lon: float, past_days: int, forecast_days: int) -> Tuple[List[str], List[float]]:
    """
    取 daily temperature_2m_mean，以 UTC 日期序列回傳
    past_days + forecast_days 的總長度足夠做 30D / 15D
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_mean",
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
        "past_days": int(past_days),
        "forecast_days": int(forecast_days),
    }
    r = retry_get(url, params=params, tries=3, timeout=20)
    j = r.json()

    daily = j.get("daily", {})
    dates = daily.get("time", [])
    temps = daily.get("temperature_2m_mean", [])

    if not dates or not temps or len(dates) != len(temps):
        raise RuntimeError("Weather API returned invalid daily series.")

    return dates, [float(x) for x in temps]


# =========================
# Storage (EIA WNGSR)
# =========================
@dataclass
class StorageData:
    week_ending: str
    total_bcf: Optional[float]
    net_change_bcf: Optional[float]
    fivey_avg_bcf: Optional[float]
    vs_5y_bcf: Optional[float]
    vs_5y_pct: Optional[float]
    is_new_release: bool
    note: str


def parse_wngsr_json(text: str) -> Dict[str, Any]:
    # Fix UTF-8 BOM issue
    cleaned = text.encode("utf-8", errors="ignore").decode("utf-8-sig", errors="ignore")
    return json.loads(cleaned)


def fetch_storage_latest(csv_df: Optional[pd.DataFrame]) -> StorageData:
    """
    取最新一筆 WNGSR，並判斷是否 NEW（跟 CSV 最後存過的 storage_week 比）
    """
    try:
        r = retry_get(EIA_WNGSR_JSON, params={}, tries=3, timeout=25)
        j = parse_wngsr_json(r.content.decode("utf-8", errors="ignore"))
    except Exception as e:
        return StorageData(
            week_ending="NA",
            total_bcf=None,
            net_change_bcf=None,
            fivey_avg_bcf=None,
            vs_5y_bcf=None,
            vs_5y_pct=None,
            is_new_release=False,
            note=f"Storage fetch failed: {e}",
        )

    # wngsr.json 結構通常包含 series / data
    # 常見 keys: "series" -> [{"data":[[date, value]...], ...}]
    # 也有版本是 "data":[{"period":"YYYY-MM-DD", ...}]
    week_ending = "NA"
    total_bcf = None
    net_change = None
    fivey = None

    try:
        if "series" in j and j["series"]:
            s0 = j["series"][0]
            data = s0.get("data", [])
            # data: [[ "2026-02-07", 2345 ], ...] 最新通常在最前
            if data:
                week_ending = str(data[0][0])
                total_bcf = float(data[0][1])
        elif "data" in j and isinstance(j["data"], list) and j["data"]:
            # 可能是 list of dict
            # 嘗試抓 period + workingGasInStorage (名稱不保證)
            row0 = j["data"][0]
            week_ending = str(row0.get("period") or row0.get("weekEnding") or "NA")
            for k in ["workingGasInStorage", "value", "total", "working_gas"]:
                if k in row0:
                    total_bcf = float(row0[k])
                    break
    except Exception:
        pass

    # net_change / fivey 可能需要看其他欄位；wngsr.json 有時也有 "change" "fiveYearAverage"
    # 這裡做「最大相容」抓法：從字典中找看起來像的 key
    try:
        # 深度搜尋（簡單版）
        def find_number_like(obj, keys):
            if isinstance(obj, dict):
                for kk, vv in obj.items():
                    if kk in keys:
                        try:
                            return float(vv)
                        except Exception:
                            pass
                    res = find_number_like(vv, keys)
                    if res is not None:
                        return res
            elif isinstance(obj, list):
                for it in obj:
                    res = find_number_like(it, keys)
                    if res is not None:
                        return res
            return None

        net_change = find_number_like(j, {"netChange", "net_change", "change", "weeklyChange"})
        fivey = find_number_like(j, {"fiveYearAverage", "five_year_average", "fiveYAvg", "five_year_avg"})
    except Exception:
        pass

    vs_5y = None
    vs_5y_pct = None
    if total_bcf is not None and fivey is not None and fivey != 0:
        vs_5y = total_bcf - fivey
        vs_5y_pct = (vs_5y / fivey) * 100.0

    # 判斷 NEW：看 CSV 最後一列是否有 storage_week
    is_new = False
    if csv_df is not None and not csv_df.empty and "storage_week" in csv_df.columns:
        last_week = str(csv_df.iloc[-1].get("storage_week", "")).strip()
        if last_week and week_ending != "NA" and week_ending != last_week:
            is_new = True

    note = "OK"
    if week_ending == "NA" and total_bcf is None:
        note = "Storage parse incomplete"

    return StorageData(
        week_ending=week_ending,
        total_bcf=total_bcf,
        net_change_bcf=net_change,
        fivey_avg_bcf=fivey,
        vs_5y_bcf=vs_5y,
        vs_5y_pct=vs_5y_pct,
        is_new_release=is_new,
        note=note,
    )


# =========================
# Telegram
# =========================
def tg_api_url(method: str) -> str:
    return f"https://api.telegram.org/bot{TG_BOT_TOKEN}/{method}"


def tg_send_message(text: str) -> None:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("[INFO] Telegram not configured; skip sendMessage.")
        return
    payload = {"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    r = requests.post(tg_api_url("sendMessage"), data=payload, timeout=25)
    if r.status_code >= 400:
        raise RuntimeError(f"Telegram sendMessage failed: {r.status_code} {r.text[:200]}")


def tg_send_photo(caption: str, photo_path: str) -> None:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("[INFO] Telegram not configured; skip sendPhoto.")
        return
    with open(photo_path, "rb") as f:
        files = {"photo": f}
        data = {"chat_id": TG_CHAT_ID, "caption": caption}
        r = requests.post(tg_api_url("sendPhoto"), data=data, files=files, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"Telegram sendPhoto failed: {r.status_code} {r.text[:200]}")


# =========================
# Signal logic
# =========================
def signal_from_delta(delta_hdd15: float, delta_hdd30: float) -> Tuple[str, str]:
    """
    用 HDD revision 做大方向訊號（你可再依偏好調整閾值）
    """
    # 以 15D 為主，30D 為輔
    if delta_hdd15 >= 10:
        return "🔥 Bullish (colder revision)", "bull"
    if delta_hdd15 <= -10:
        return "🧊 Bearish (warmer revision)", "bear"
    # 中間區
    return "😐 Neutral", "neutral"


# =========================
# Chart
# =========================
def make_chart(dates: List[str], hdd: List[float], cdd: List[float], title: str, out_path: str) -> None:
    # dates: YYYY-MM-DD strings
    x = list(range(len(dates)))

    plt.figure(figsize=(12, 5))
    plt.plot(x, hdd, label=f"Daily HDD (base {BASE_F:.0f}F)")
    plt.plot(x, cdd, label=f"Daily CDD (base {BASE_F:.0f}F)")
    plt.xticks(x[::2], dates[::2], rotation=45, ha="right")
    plt.xlabel("Day (UTC)")
    plt.ylabel("Degree Days")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# =========================
# Main
# =========================
def read_csv() -> Optional[pd.DataFrame]:
    if not os.path.exists(CSV_PATH):
        return None
    try:
        return pd.read_csv(CSV_PATH)
    except Exception:
        return None


def get_yoy_from_csv(df: pd.DataFrame, today_utc: str, col: str) -> Optional[float]:
    """
    YoY：用 CSV 裡同月同日（-365天附近）抓最接近的一筆（容許±2天）
    """
    if df is None or df.empty or "run_date_utc" not in df.columns or col not in df.columns:
        return None
    try:
        target = dt.datetime.strptime(today_utc, "%Y-%m-%d").date() - dt.timedelta(days=365)
        df2 = df.copy()
        df2["run_date_utc"] = pd.to_datetime(df2["run_date_utc"], errors="coerce").dt.date
        df2 = df2.dropna(subset=["run_date_utc"])
        # 找最接近 target 的
        df2["diff"] = df2["run_date_utc"].apply(lambda d: abs((d - target).days))
        cand = df2[df2["diff"] <= 2].sort_values("diff")
        if cand.empty:
            return None
        v = cand.iloc[0][col]
        return float(v) if pd.notna(v) else None
    except Exception:
        return None


def run():
    now = utc_now()
    run_date = now.strftime("%Y-%m-%d")  # 以 UTC date 做key

    df_old = read_csv()

    # 1) Weather
    dates, temps = fetch_daily_mean_f(LAT, LON, PAST_DAYS, FORECAST_DAYS)
    hdd_series, cdd_series = compute_hdd_cdd_series(temps, BASE_F)

    # 2) 15D/30D weighted (取 series 最後 N 天)
    hdd15 = weighted_average(hdd_series[-15:])
    hdd30 = weighted_average(hdd_series[-30:])
    cdd15 = weighted_average(cdd_series[-15:])
    cdd30 = weighted_average(cdd_series[-30:])

    # 3) Delta vs previous run (同 CSV 最後一筆)
    prev_hdd15 = prev_hdd30 = prev_cdd15 = prev_cdd30 = None
    if df_old is not None and not df_old.empty:
        prev_hdd15 = float(df_old.iloc[-1].get("hdd_15d", float("nan")))
        prev_hdd30 = float(df_old.iloc[-1].get("hdd_30d", float("nan")))
        prev_cdd15 = float(df_old.iloc[-1].get("cdd_15d", float("nan")))
        prev_cdd30 = float(df_old.iloc[-1].get("cdd_30d", float("nan")))

    delta_hdd15 = (hdd15 - prev_hdd15) if prev_hdd15 is not None and not math.isnan(prev_hdd15) else 0.0
    delta_hdd30 = (hdd30 - prev_hdd30) if prev_hdd30 is not None and not math.isnan(prev_hdd30) else 0.0
    delta_cdd15 = (cdd15 - prev_cdd15) if prev_cdd15 is not None and not math.isnan(prev_cdd15) else 0.0
    delta_cdd30 = (cdd30 - prev_cdd30) if prev_cdd30 is not None and not math.isnan(prev_cdd30) else 0.0

    sig_text, _sig_key = signal_from_delta(delta_hdd15, delta_hdd30)

    # 4) Storage (EIA WNGSR) + BOM fix + NEW detect
    storage = fetch_storage_latest(df_old)

    # 5) YoY from CSV (如果 CSV 有歷史才會出)
    yoy_hdd15 = get_yoy_from_csv(df_old, run_date, "hdd_15d") if df_old is not None else None
    yoy_hdd30 = get_yoy_from_csv(df_old, run_date, "hdd_30d") if df_old is not None else None
    yoy_cdd15 = get_yoy_from_csv(df_old, run_date, "cdd_15d") if df_old is not None else None
    yoy_cdd30 = get_yoy_from_csv(df_old, run_date, "cdd_30d") if df_old is not None else None

    # 6) Save chart
    title = f"HDD/CDD Trend · {run_date} UTC"
    make_chart(dates, hdd_series, cdd_series, title, CHART_PATH)

    # 7) Append CSV row
    row = {
        "run_date_utc": run_date,
        "updated_utc": iso_utc(now),
        "base_f": BASE_F,
        "hdd_15d": round(hdd15, 2),
        "hdd_30d": round(hdd30, 2),
        "cdd_15d": round(cdd15, 2),
        "cdd_30d": round(cdd30, 2),
        "delta_hdd_15d": round(delta_hdd15, 2),
        "delta_hdd_30d": round(delta_hdd30, 2),
        "delta_cdd_15d": round(delta_cdd15, 2),
        "delta_cdd_30d": round(delta_cdd30, 2),
        "signal": sig_text,
        "storage_week": storage.week_ending,
        "storage_total_bcf": storage.total_bcf,
        "storage_net_change_bcf": storage.net_change_bcf,
        "storage_5y_avg_bcf": storage.fivey_avg_bcf,
        "storage_vs_5y_bcf": storage.vs_5y_bcf,
        "storage_vs_5y_pct": storage.vs_5y_pct,
        "storage_note": storage.note,
    }

    df_new = pd.DataFrame([row]) if df_old is None else pd.concat([df_old, pd.DataFrame([row])], ignore_index=True)
    df_new.to_csv(CSV_PATH, index=False)

    # 8) Compose Telegram text (更好讀)
    def arrow(x: float) -> str:
        if x >= 0.01:
            return "⬆️"
        if x <= -0.01:
            return "⬇️"
        return "➖"

    storage_tag = "🟢 NEW" if storage.is_new_release else "🟡 Latest"
    storage_block = (
        f"🧱 <b>Storage (EIA WNGSR)</b> {storage_tag}\n"
        f"• Week: {storage.week_ending}\n"
        f"• Total: {fmt_num(storage.total_bcf,0)} bcf\n"
        f"• Net chg: {fmt_num(storage.net_change_bcf,0)} bcf\n"
        f"• 5Y avg: {fmt_num(storage.fivey_avg_bcf,0)} bcf\n"
        f"• vs 5Y: {fmt_num(storage.vs_5y_bcf,0)} bcf ({fmt_num(storage.vs_5y_pct,2)}%)\n"
        if storage.total_bcf is not None else
        f"🧱 <b>Storage</b>: NA\n• {storage.note}\n"
    )

    yoy_block = (
        f"📆 <b>YoY</b>\n"
        f"• HDD15: {fmt_num((hdd15 - yoy_hdd15),2) if yoy_hdd15 is not None else 'NA'}\n"
        f"• HDD30: {fmt_num((hdd30 - yoy_hdd30),2) if yoy_hdd30 is not None else 'NA'}\n"
        f"• CDD15: {fmt_num((cdd15 - yoy_cdd15),2) if yoy_cdd15 is not None else 'NA'}\n"
        f"• CDD30: {fmt_num((cdd30 - yoy_cdd30),2) if yoy_cdd30 is not None else 'NA'}\n"
    )

    text = (
        f"📌 <b>HDD/CDD Update ({run_date})</b>\n\n"
        f"🔥 <b>HDD (base {BASE_F:.0f}F)</b>\n"
        f"• 15D Wtd: {fmt_num(hdd15)}  ({arrow(delta_hdd15)} Δ {fmt_num(delta_hdd15)})\n"
        f"• 30D Wtd: {fmt_num(hdd30)}  ({arrow(delta_hdd30)} Δ {fmt_num(delta_hdd30)})\n\n"
        f"☀️ <b>CDD (base {BASE_F:.0f}F)</b>\n"
        f"• 15D Wtd: {fmt_num(cdd15)}  ({arrow(delta_cdd15)} Δ {fmt_num(delta_cdd15)})\n"
        f"• 30D Wtd: {fmt_num(cdd30)}  ({arrow(delta_cdd30)} Δ {fmt_num(delta_cdd30)})\n\n"
        f"{storage_block}\n"
        f"{yoy_block}\n"
        f"📍 <b>Signal</b>: {sig_text}\n"
        f"⏱ <b>Updated</b>: {iso_utc(now)}"
    )

    # 9) Send Telegram text + chart
    tg_send_message(text)
    tg_send_photo(caption=f"📈 Trend (HDD/CDD) · {run_date} UTC", photo_path=CHART_PATH)


if __name__ == "__main__":
    run()
