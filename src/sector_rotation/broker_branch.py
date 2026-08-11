"""Taiwan broker-branch trade ingestion and explainable research signals.

Broker branches are transaction channels, not investor identities.  A branch's
buy/sell imbalance is therefore used as a concentration and persistence signal,
then confirmed with institutional flow and price data before a stock is placed
on the research-candidate list.
"""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


FINMIND_BRANCH_URL = (
    "https://api.finmindtrade.com/api/v4/"
    "taiwan_stock_trading_daily_report"
)

BRANCH_COLUMNS = [
    "Date",
    "Ticker",
    "Broker ID",
    "Broker",
    "Price",
    "Buy shares",
    "Sell shares",
]


def normalize_broker_branch_trades(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize FinMind-style or uploaded broker-branch rows."""
    if frame.empty:
        return pd.DataFrame(columns=BRANCH_COLUMNS)
    aliases = {
        "date": "Date",
        "stock_id": "Ticker",
        "securities_trader_id": "Broker ID",
        "securities_trader": "Broker",
        "price": "Price",
        "buy": "Buy shares",
        "sell": "Sell shares",
        "buy_volume": "Buy shares",
        "sell_volume": "Sell shares",
    }
    normalized = frame.rename(columns={key: value for key, value in aliases.items() if key in frame})
    missing = [column for column in BRANCH_COLUMNS if column not in normalized]
    if missing:
        raise ValueError(f"券商分點資料缺少欄位：{', '.join(missing)}")
    normalized = normalized[BRANCH_COLUMNS].copy()
    normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce").dt.normalize()
    normalized["Ticker"] = (
        normalized["Ticker"].astype(str).str.strip().str.replace(r"\.(TW|TWO)$", "", regex=True)
    )
    normalized["Broker ID"] = normalized["Broker ID"].astype(str).str.strip()
    normalized["Broker"] = normalized["Broker"].astype(str).str.strip()
    for column in ["Price", "Buy shares", "Sell shares"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0.0)
    return normalized.dropna(subset=["Date"]).reset_index(drop=True)


def fetch_finmind_broker_branch(
    ticker: str,
    session: date,
    token: str,
) -> pd.DataFrame:
    """Fetch one stock/session from FinMind's sponsor broker-branch endpoint."""
    if not token:
        raise ValueError("需要 FINMIND_TOKEN 才能下載券商分點資料。")
    query = urlencode({"data_id": ticker.split(".")[0], "date": session.isoformat()})
    request = Request(
        f"{FINMIND_BRANCH_URL}?{query}",
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "Mozilla/5.0 taiwan-fund-flow-lab/1.0",
        },
    )
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if payload.get("status") != 200:
        raise RuntimeError(payload.get("msg", "券商分點資料下載失敗。"))
    return normalize_broker_branch_trades(pd.DataFrame(payload.get("data", [])))


def load_broker_branch_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=BRANCH_COLUMNS)
    if path.suffix.lower() == ".csv":
        return normalize_broker_branch_trades(pd.read_csv(path))
    return normalize_broker_branch_trades(pd.read_parquet(path))


def aggregate_branch_activity(
    trades: pd.DataFrame,
    *,
    window: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return branch/stock activity and stock-level concentration signals."""
    trades = normalize_broker_branch_trades(trades)
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame()
    dates = sorted(trades["Date"].unique())[-max(1, window) :]
    selected = trades[trades["Date"].isin(dates)].copy()
    selected["Net shares"] = selected["Buy shares"] - selected["Sell shares"]
    selected["Net value"] = selected["Net shares"] * selected["Price"]
    selected["Gross shares"] = selected["Buy shares"] + selected["Sell shares"]

    branch_stock = (
        selected.groupby(["Ticker", "Broker ID", "Broker"], as_index=False)
        .agg(
            **{
                "Buy shares": ("Buy shares", "sum"),
                "Sell shares": ("Sell shares", "sum"),
                "Net shares": ("Net shares", "sum"),
                "Net value": ("Net value", "sum"),
                "Gross shares": ("Gross shares", "sum"),
                "Active days": ("Date", "nunique"),
            }
        )
    )
    branch_stock["Direction"] = np.select(
        [branch_stock["Net shares"] > 0, branch_stock["Net shares"] < 0],
        ["買超", "賣超"],
        default="中性",
    )

    def summarize(group: pd.DataFrame) -> pd.Series:
        buyers = group[group["Net shares"] > 0].sort_values("Net value", ascending=False)
        sellers = group[group["Net shares"] < 0].sort_values("Net value")
        positive_value = buyers["Net value"].clip(lower=0).sum()
        negative_value = -sellers["Net value"].clip(upper=0).sum()
        active_count = int((group["Net shares"] != 0).sum())
        return pd.Series(
            {
                "Buying branches": int(len(buyers)),
                "Selling branches": int(len(sellers)),
                "Positive branch ratio": len(buyers) / active_count if active_count else 0.0,
                "Positive branch value": positive_value,
                "Negative branch value": negative_value,
                "Top 3 buying concentration": (
                    buyers["Net value"].head(3).sum() / positive_value if positive_value > 0 else 0.0
                ),
                "Top buyer": buyers.iloc[0]["Broker"] if not buyers.empty else "—",
                "Top buyer value": buyers.iloc[0]["Net value"] if not buyers.empty else 0.0,
                "Top seller": sellers.iloc[0]["Broker"] if not sellers.empty else "—",
                "Top seller value": sellers.iloc[0]["Net value"] if not sellers.empty else 0.0,
                "Observed days": int(selected[selected["Ticker"] == group.name]["Date"].nunique()),
            }
        )

    stocks = branch_stock.groupby("Ticker").apply(summarize, include_groups=False).reset_index()
    stocks["Branch concentration score"] = (
        stocks["Top buyer value"].rank(pct=True).fillna(0.5) * 45
        + stocks["Positive branch ratio"].rank(pct=True).fillna(0.5) * 25
        + stocks["Top 3 buying concentration"].rank(pct=True).fillna(0.5) * 30
    )
    stocks["Signal date"] = pd.Timestamp(max(dates))
    return branch_stock.sort_values("Net value", ascending=False), stocks.sort_values(
        "Branch concentration score", ascending=False
    )


def build_broker_research_candidates(
    branch_stocks: pd.DataFrame,
    institutional: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:
    """Combine branch concentration, institutional flow, and price confirmation."""
    if branch_stocks.empty:
        return pd.DataFrame()
    candidates = branch_stocks.copy()
    ticker_map = master.drop_duplicates("Code").set_index("Code")
    candidates["Yahoo ticker"] = candidates["Ticker"].map(
        ticker_map["Yahoo ticker"].to_dict() if "Yahoo ticker" in ticker_map else {}
    )
    for column in ["Name", "Industry", "Detailed industry", "Investment theme"]:
        if column in ticker_map:
            candidates[column] = candidates["Ticker"].map(ticker_map[column].to_dict())
    institutional_columns = [
        "Ticker",
        "Selected net value",
        "Selected return",
        "Flow score",
    ]
    available = [column for column in institutional_columns if column in institutional]
    if available:
        candidates = candidates.merge(
            institutional[available].rename(columns={"Ticker": "Yahoo ticker"}),
            on="Yahoo ticker",
            how="left",
        )
    for column in ["Selected net value", "Selected return", "Flow score"]:
        if column not in candidates:
            candidates[column] = 0.0
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce").fillna(0.0)
    candidates["Research score"] = (
        candidates["Branch concentration score"] * 0.50
        + candidates["Flow score"].rank(pct=True).fillna(0.5) * 100 * 0.30
        + candidates["Selected return"].rank(pct=True).fillna(0.5) * 100 * 0.20
    )
    candidates["Recommendation"] = np.select(
        [
            (candidates["Selected net value"] > 0) & (candidates["Selected return"] > 0),
            candidates["Selected net value"] > 0,
        ],
        ["優先研究", "觀察價格確認"],
        default="分點集中、法人未確認",
    )
    candidates["Reason"] = candidates.apply(
        lambda row: (
            f"{row['Top buyer']}為最大買超分點；買超分點占比"
            f"{row['Positive branch ratio']:.0%}；前三大買盤集中度"
            f"{row['Top 3 buying concentration']:.0%}；法人"
            f"{row['Selected net value'] / 1e8:+.1f}億；價格"
            f"{row['Selected return']:+.1%}。"
        ),
        axis=1,
    )
    return candidates.sort_values("Research score", ascending=False).reset_index(drop=True)
