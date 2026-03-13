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
    # 6dca-aqww 是 CFTC 的持倉分類報告資料集 ID
    url = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
    
    # 023651 是 NYMEX Henry Hub 天然氣的專屬代碼
    params = {
        "cftc_contract_market_code": "023651",
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": 2  # 抓取最新的兩週數據以計算 WoW
    }
    
    try:
        r = retry_get(url, params=params)
        if isinstance(r, str): return COTInfo(note=f"CFTC Error: {r[:30]}")
        
        data = r.json()
        if not data or len(data) == 0:
            return COTInfo(note="Empty CFTC data")
            
        # 提取最新一期 (本週)
        curr = data[0]
        # m_money 代表 Managed Money (基金經理人)
        mm_long_curr = float(curr.get("m_money_positions_long_all", 0))
        mm_short_curr = float(curr.get("m_money_positions_short_all", 0))
        net_curr = mm_long_curr - mm_short_curr
        
        # 提取上一期 (上週) 以計算部位變化
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
