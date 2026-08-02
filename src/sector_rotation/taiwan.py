"""Taiwan-listed company master data and curated research universes."""

from __future__ import annotations

import json
from urllib.request import Request, urlopen

import pandas as pd

from .universe import AssetInfo


TWSE_COMPANY_API = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
TPEX_COMPANY_API = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"
TWSE_FUND_API = "https://openapi.twse.com.tw/v1/opendata/t187ap47_L"
TWSE_DAILY_QUOTES_API = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TPEX_QUOTES_API = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"

TW_BENCHMARK = "0050.TW"
TW_DEFENSIVE_ASSET = "00679B.TWO"

INDUSTRY_NAMES = {
    "01": "水泥工業",
    "02": "食品工業",
    "03": "塑膠工業",
    "04": "紡織纖維",
    "05": "電機機械",
    "06": "電器電纜",
    "08": "玻璃陶瓷",
    "09": "造紙工業",
    "10": "鋼鐵工業",
    "11": "橡膠工業",
    "12": "汽車工業",
    "14": "建材營造",
    "15": "航運業",
    "16": "觀光餐旅",
    "17": "金融保險",
    "18": "貿易百貨",
    "20": "其他",
    "21": "化學工業",
    "22": "生技醫療",
    "23": "油電燃氣",
    "24": "半導體業",
    "25": "電腦及週邊設備",
    "26": "光電業",
    "27": "通信網路業",
    "28": "電子零組件",
    "29": "電子通路業",
    "30": "資訊服務業",
    "31": "其他電子業",
    "32": "文化創意業",
    "33": "農業科技",
    "34": "電子商務",
    "35": "綠能環保",
    "36": "數位雲端",
    "37": "運動休閒",
    "38": "居家生活",
}

TW_ETF_GROUPS: dict[str, dict[str, str]] = {
    "大盤與市值型": {
        "0050.TW": "元大台灣50",
        "006208.TW": "富邦台50",
        "00922.TW": "國泰台灣領袖50",
        "00923.TW": "群益台ESG低碳50",
    },
    "科技與半導體": {
        "0052.TW": "富邦科技",
        "00881.TW": "國泰台灣科技龍頭",
        "00935.TW": "野村臺灣新科技50",
        "00927.TW": "群益半導體收益",
        "00904.TW": "新光臺灣半導體30",
    },
    "高股息與低波動": {
        "0056.TW": "元大高股息",
        "00878.TW": "國泰永續高股息",
        "00713.TW": "元大台灣高息低波",
        "00919.TW": "群益台灣精選高息",
        "00918.TW": "大華優利高填息30",
        "00915.TW": "凱基優選高股息30",
    },
    "中小型與成長": {
        "0051.TW": "元大中型100",
        "00850.TW": "元大臺灣ESG永續",
        "00692.TW": "富邦公司治理",
    },
    "債券與防禦": {
        "00679B.TWO": "元大美債20年",
        "00687B.TWO": "國泰20年美債",
        "00772B.TWO": "中信高評級公司債",
    },
}

TW_THEME_CODES: dict[str, list[str]] = {
    "晶圓製造": ["2330", "2303", "5347", "6770"],
    "IC設計": ["2454", "3034", "2379", "3661", "3443", "5274", "4966", "6415", "6531"],
    "封裝測試": ["3711", "2449", "6239", "8150", "6147", "3264"],
    "半導體設備與材料": ["3131", "3583", "6223", "6196", "6187", "5443", "6488", "1560", "3413", "3653", "6640", "4768"],
    "AI伺服器與ODM": ["2317", "2382", "3231", "6669", "2356", "2376", "2324", "4938"],
    "散熱與電源": ["3017", "3324", "3653", "2421", "2308", "6412"],
    "PCB與銅箔基板": ["3037", "2368", "2383", "8046", "6274", "3189", "2313", "4958", "3715"],
    "網通與高速傳輸": ["2345", "6285", "3596", "5388", "4979", "2412", "4904"],
    "光學與光通訊": ["3008", "3406", "3363", "4977", "3019", "3081"],
    "記憶體與儲存": ["2408", "2344", "2337", "8299", "3260"],
    "金融控股": ["2881", "2882", "2891", "2886", "2884", "2885", "2892", "5880", "2880", "5876"],
    "航運與航空": ["2603", "2609", "2615", "2606", "2618", "2610", "2637"],
    "生技製藥": ["6446", "1795", "4743", "6472", "4142", "4128", "6547"],
    "電動車零組件": ["2308", "1536", "2231", "2250", "1319", "3665", "6279", "9951"],
    "重電與綠能": ["1513", "1519", "1503", "1609", "6806", "6869", "3708", "3576", "6244"],
    "軍工與無人機": ["2634", "6753", "4541", "4571", "8033", "8222", "2208"],
    "營建資產": ["2542", "2548", "5534", "5522", "2504", "2515"],
    "消費與零售": ["2912", "5903", "8454", "2727", "2707", "9910", "1216", "1231"],
    "鋼鐵塑化與原物料": ["2002", "2014", "2027", "2031", "1301", "1303", "1326"],
}


def _get_json(url: str) -> list[dict[str, str]]:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 rotation-research/0.4"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def fetch_taiwan_company_master() -> pd.DataFrame:
    """Download and normalize official TWSE and TPEx company master data."""
    twse = pd.DataFrame(_get_json(TWSE_COMPANY_API)).rename(
        columns={
            "公司代號": "Code",
            "公司簡稱": "Name",
            "公司名稱": "Full name",
            "產業別": "Industry code",
            "已發行普通股數或TDR原股發行股數": "Issued shares",
        }
    )
    twse["Market"] = "上市"
    twse["Yahoo ticker"] = twse["Code"].astype(str).str.strip() + ".TW"

    tpex = pd.DataFrame(_get_json(TPEX_COMPANY_API)).rename(
        columns={
            "SecuritiesCompanyCode": "Code",
            "CompanyAbbreviation": "Name",
            "CompanyName": "Full name",
            "SecuritiesIndustryCode": "Industry code",
            "IssueShares": "Issued shares",
        }
    )
    tpex["Market"] = "上櫃"
    tpex["Yahoo ticker"] = tpex["Code"].astype(str).str.strip() + ".TWO"

    columns = [
        "Code",
        "Yahoo ticker",
        "Name",
        "Full name",
        "Industry code",
        "Issued shares",
        "Market",
    ]
    master = pd.concat([twse[columns], tpex[columns]], ignore_index=True)
    master["Code"] = master["Code"].astype(str).str.strip()
    master["Name"] = master["Name"].astype(str).str.strip()
    master["Industry code"] = master["Industry code"].astype(str).str.strip().str.zfill(2)
    master["Industry"] = master["Industry code"].map(INDUSTRY_NAMES).fillna("其他／未分類")
    master["Issued shares"] = pd.to_numeric(master["Issued shares"], errors="coerce")
    master["Asset type"] = "股票"
    return master.drop_duplicates("Yahoo ticker").sort_values(["Industry", "Market", "Code"])


def _classify_etf(name: str, fund_type: str = "") -> str:
    """Assign a practical research group from official ETF descriptions."""
    text = f"{name} {fund_type}"
    if "債" in text:
        return "債券 ETF"
    if any(token in text for token in ["原油", "黃金", "白銀", "期貨", "商品"]):
        return "商品／期貨 ETF"
    if any(token in text for token in ["槓桿", "反向", "正2", "正二", "反1", "反一"]):
        return "槓桿／反向 ETF"
    if any(token in text for token in ["主動式", "主動"]):
        return "主動式 ETF"
    if any(token in text for token in ["國外", "海外", "美國", "日本", "中國", "印度"]):
        return "海外股票 ETF"
    return "台灣股票 ETF"


def fetch_taiwan_etf_master() -> pd.DataFrame:
    """Download all exchange-listed ETFs from TWSE and TPEx official data.

    TWSE publishes a dedicated fund master. TPEx's official closing-quote feed
    contains both stocks and ETFs; Taiwan ETF security codes begin with ``00``,
    while exchange-traded notes use a different ``02`` prefix and are excluded.
    """
    twse_raw = pd.DataFrame(_get_json(TWSE_FUND_API))
    current_twse_codes = {
        str(row.get("Code", "")).strip()
        for row in _get_json(TWSE_DAILY_QUOTES_API)
    }
    twse_raw = twse_raw[
        twse_raw["基金代號"].astype(str).str.strip().isin(current_twse_codes)
    ].copy()
    twse = pd.DataFrame(
        {
            "Code": twse_raw["基金代號"].astype(str).str.strip(),
            "Name": twse_raw["基金簡稱"].astype(str).str.strip(),
            "Full name": twse_raw["基金中文名稱"].astype(str).str.strip(),
            "Fund type": twse_raw["基金類型"].astype(str).str.strip(),
            "Issued shares": pd.to_numeric(
                twse_raw["發行單位數/轉換數"], errors="coerce"
            ),
        }
    )
    twse["Market"] = "上市"
    twse["Yahoo ticker"] = twse["Code"] + ".TW"

    tpex_raw = pd.DataFrame(_get_json(TPEX_QUOTES_API))
    tpex_raw["Code"] = tpex_raw["SecuritiesCompanyCode"].astype(str).str.strip()
    tpex_raw = tpex_raw[tpex_raw["Code"].str.startswith("00")].copy()
    tpex = pd.DataFrame(
        {
            "Code": tpex_raw["Code"],
            "Name": tpex_raw["CompanyName"].astype(str).str.strip(),
            "Full name": tpex_raw["CompanyName"].astype(str).str.strip(),
            "Fund type": "",
            "Issued shares": pd.to_numeric(tpex_raw["Capitals"], errors="coerce"),
        }
    )
    tpex["Market"] = "上櫃"
    tpex["Yahoo ticker"] = tpex["Code"] + ".TWO"

    master = pd.concat([twse, tpex], ignore_index=True)
    master["Industry code"] = "ETF"
    master["Industry"] = [
        _classify_etf(name, fund_type)
        for name, fund_type in zip(master["Name"], master["Fund type"], strict=False)
    ]
    master["Asset type"] = "ETF"
    return (
        master[
            [
                "Code",
                "Yahoo ticker",
                "Name",
                "Full name",
                "Industry code",
                "Issued shares",
                "Market",
                "Industry",
                "Asset type",
                "Fund type",
            ]
        ]
        .drop_duplicates("Yahoo ticker")
        .sort_values(["Market", "Industry", "Code"])
        .reset_index(drop=True)
    )


def fetch_taiwan_security_master() -> pd.DataFrame:
    """Return one complete master for listed/OTC companies and ETFs."""
    companies = fetch_taiwan_company_master()
    companies["Fund type"] = ""
    etfs = fetch_taiwan_etf_master()
    columns = list(etfs.columns)
    return (
        pd.concat([companies.reindex(columns=columns), etfs], ignore_index=True)
        .drop_duplicates("Yahoo ticker")
        .sort_values(["Asset type", "Market", "Industry", "Code"])
        .reset_index(drop=True)
    )


def official_industry_groups(master: pd.DataFrame) -> list[str]:
    return sorted(master["Industry"].dropna().unique())


def assets_from_official_industries(
    master: pd.DataFrame,
    industries: list[str],
    max_per_industry: int | None = None,
) -> dict[str, AssetInfo]:
    """Build complete industry universes, optionally capped by issued shares."""
    selected = master[
        master["Industry"].isin(industries)
        & master.get("Asset type", pd.Series("股票", index=master.index)).eq("股票")
    ].copy()
    selected = selected.sort_values(
        ["Industry", "Issued shares"], ascending=[True, False]
    )
    if max_per_industry is not None:
        selected = selected.groupby("Industry", group_keys=False).head(max_per_industry)
    return {
        row["Yahoo ticker"]: AssetInfo(
            ticker=row["Yahoo ticker"],
            name=f"{row['Name']} ({row['Market']})",
            group=row["Industry"],
        )
        for _, row in selected.iterrows()
    }


def assets_from_taiwan_themes(
    master: pd.DataFrame,
    themes: list[str],
) -> dict[str, AssetInfo]:
    by_code = master.set_index("Code")
    assets: dict[str, AssetInfo] = {}
    for theme in themes:
        for code in TW_THEME_CODES[theme]:
            if code not in by_code.index:
                continue
            row = by_code.loc[code]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            ticker = str(row["Yahoo ticker"])
            assets.setdefault(
                ticker,
                AssetInfo(
                    ticker=ticker,
                    name=f"{row['Name']} ({row['Market']})",
                    group=theme,
                ),
            )
    return assets


def assets_from_taiwan_etfs(groups: list[str]) -> dict[str, AssetInfo]:
    assets: dict[str, AssetInfo] = {}
    for group in groups:
        for ticker, name in TW_ETF_GROUPS[group].items():
            assets.setdefault(ticker, AssetInfo(ticker=ticker, name=name, group=group))
    return assets


def assets_from_taiwan_security_master(
    master: pd.DataFrame,
    *,
    markets: list[str] | None = None,
    asset_types: list[str] | None = None,
    groups: list[str] | None = None,
) -> dict[str, AssetInfo]:
    """Build an uncapped universe from the complete Taiwan security master."""
    selected = master.copy()
    if markets:
        selected = selected[selected["Market"].isin(markets)]
    if asset_types:
        selected = selected[selected["Asset type"].isin(asset_types)]
    if groups:
        selected = selected[selected["Industry"].isin(groups)]
    return {
        row["Yahoo ticker"]: AssetInfo(
            ticker=row["Yahoo ticker"],
            name=f"{row['Name']} ({row['Market']}／{row['Asset type']})",
            group=row["Industry"],
        )
        for _, row in selected.iterrows()
    }


def custom_taiwan_assets(
    raw_codes: list[str],
    master: pd.DataFrame,
) -> dict[str, AssetInfo]:
    by_code = master.set_index("Code")
    assets: dict[str, AssetInfo] = {}
    for raw in raw_codes:
        code = raw.upper().split(".", maxsplit=1)[0]
        if code in by_code.index:
            row = by_code.loc[code]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            ticker = str(row["Yahoo ticker"])
            assets[ticker] = AssetInfo(
                ticker=ticker,
                name=f"{row['Name']} ({row['Market']})",
                group="自訂台股",
            )
    return assets


def all_taiwan_research_assets(master: pd.DataFrame) -> dict[str, AssetInfo]:
    """Return every listed/OTC stock and ETF in the supplied master."""
    return assets_from_taiwan_security_master(master)
