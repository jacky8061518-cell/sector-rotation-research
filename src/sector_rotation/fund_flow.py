"""Taiwan institutional-flow ingestion and explainable flow signals."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


TWSE_INSTITUTIONAL_URL = "https://www.twse.com.tw/rwd/zh/fund/T86"
TPEX_INSTITUTIONAL_URL = (
    "https://www.tpex.org.tw/web/stock/3insti/daily_trade/"
    "3itrade_hedge_result.php"
)

FLOW_COLUMNS = [
    "Date",
    "Ticker",
    "Name",
    "Market",
    "Foreign net shares",
    "Trust net shares",
    "Dealer net shares",
    "Total net shares",
]


def _get_json(url: str, params: dict[str, str]) -> dict:
    query = urlencode(params)
    request = Request(
        f"{url}?{query}",
        headers={"User-Agent": "Mozilla/5.0 rotation-research/0.6"},
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def _number(value: object) -> float:
    if value is None:
        return 0.0
    text = str(value).replace(",", "").strip()
    if text in {"", "--", "---"}:
        return 0.0
    return float(text)


def fetch_twse_institutional_flows(session: date) -> pd.DataFrame:
    """Fetch one TWSE session of stock-level institutional net shares."""
    payload = _get_json(
        TWSE_INSTITUTIONAL_URL,
        {
            "date": session.strftime("%Y%m%d"),
            "selectType": "ALLBUT0999",
            "response": "json",
        },
    )
    if payload.get("stat") != "OK":
        return pd.DataFrame(columns=FLOW_COLUMNS)
    rows = []
    for row in payload.get("data", []):
        if len(row) < 19:
            continue
        code = str(row[0]).strip()
        rows.append(
            {
                "Date": pd.Timestamp(session),
                "Ticker": f"{code}.TW",
                "Name": str(row[1]).strip(),
                "Market": "上市",
                "Foreign net shares": _number(row[4]),
                "Trust net shares": _number(row[10]),
                "Dealer net shares": _number(row[11]),
                "Total net shares": _number(row[18]),
            }
        )
    return pd.DataFrame(rows, columns=FLOW_COLUMNS)


def fetch_tpex_institutional_flows(session: date) -> pd.DataFrame:
    """Fetch one TPEx session of stock-level institutional net shares."""
    roc_date = f"{session.year - 1911:03d}/{session:%m/%d}"
    payload = _get_json(
        TPEX_INSTITUTIONAL_URL,
        {
            "l": "zh-tw",
            "o": "json",
            "se": "EW",
            "t": "D",
            "d": roc_date,
        },
    )
    if payload.get("stat") != "ok" or not payload.get("tables"):
        return pd.DataFrame(columns=FLOW_COLUMNS)
    tables = [table for table in payload["tables"] if table.get("data")]
    if not tables:
        return pd.DataFrame(columns=FLOW_COLUMNS)
    rows = []
    for row in tables[0]["data"]:
        if len(row) < 24:
            continue
        code = str(row[0]).strip()
        rows.append(
            {
                "Date": pd.Timestamp(session),
                "Ticker": f"{code}.TWO",
                "Name": str(row[1]).strip(),
                "Market": "上櫃",
                "Foreign net shares": _number(row[4]),
                "Trust net shares": _number(row[13]),
                "Dealer net shares": _number(row[22]),
                "Total net shares": _number(row[23]),
            }
        )
    return pd.DataFrame(rows, columns=FLOW_COLUMNS)


def fetch_taiwan_institutional_flows(session: date) -> pd.DataFrame:
    """Fetch TWSE and TPEx institutional flows for one session."""
    frames = []
    for fetcher in (fetch_twse_institutional_flows, fetch_tpex_institutional_flows):
        try:
            frame = fetcher(session)
        except (OSError, TimeoutError, ValueError, json.JSONDecodeError):
            frame = pd.DataFrame(columns=FLOW_COLUMNS)
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FLOW_COLUMNS)


def update_institutional_flow_cache(
    cache_path: Path,
    sessions: Iterable[pd.Timestamp | date],
) -> pd.DataFrame:
    """Incrementally save official institutional-flow observations."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = pd.DataFrame(columns=FLOW_COLUMNS)
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached["Date"] = pd.to_datetime(cached["Date"])
    existing_dates = set(cached["Date"].dt.normalize()) if not cached.empty else set()
    fresh = []
    for raw_session in sessions:
        session = pd.Timestamp(raw_session).date()
        if pd.Timestamp(session) in existing_dates:
            continue
        frame = fetch_taiwan_institutional_flows(session)
        if not frame.empty:
            fresh.append(frame)
    frames = ([cached] if not cached.empty else []) + fresh
    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=FLOW_COLUMNS)
    )
    if not combined.empty:
        combined["Date"] = pd.to_datetime(combined["Date"])
        combined = (
            combined.drop_duplicates(["Date", "Ticker"], keep="last")
            .sort_values(["Date", "Ticker"])
            .reset_index(drop=True)
        )
    combined.to_parquet(cache_path, index=False)
    return combined


def _rank_score(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    total_weight = sum(max(0.0, weight) for weight in weights.values())
    if total_weight == 0:
        return score
    for column, weight in weights.items():
        if weight <= 0:
            continue
        score += (
            frame[column].replace([np.inf, -np.inf], np.nan).rank(pct=True).fillna(0.5)
            * weight
        )
    return 100 * score / total_weight


def _flow_stage(row: pd.Series) -> str:
    short_flow = row["5D flow intensity"]
    long_flow = row["20D flow intensity"]
    price_return = row["20D return"]
    if short_flow > 0 and long_flow <= 0:
        return "早期轉入"
    if short_flow > 0 and long_flow > 0 and price_return > 0:
        return "資金累積＋價格確認"
    if short_flow > 0 and long_flow > 0 and price_return <= 0:
        return "資金累積、價格未確認"
    if short_flow <= 0 and long_flow > 0 and price_return > 0:
        return "漲勢仍在、資金減速"
    if short_flow <= 0 and long_flow > 0 and price_return <= 0:
        return "中期流入、短期轉弱"
    if short_flow < 0 and long_flow < 0:
        return "資金撤出"
    return "觀察"


def _research_action(stage: str) -> str:
    return {
        "早期轉入": "列入觀察，等待中期流量或價格確認",
        "資金累積＋價格確認": "優先研究領先股，分批而非追價",
        "資金累積、價格未確認": "等待價格轉強與流入廣度擴大",
        "漲勢仍在、資金減速": "不追價，提高停利或降低權重",
        "中期流入、短期轉弱": "保留觀察，等待短期資金回流",
        "資金撤出": "排除逆勢加碼，檢查既有部位風險",
        "觀察": "訊號不一致，暫不採取方向性動作",
    }.get(stage, "觀察")


def _flow_reason(row: pd.Series) -> str:
    direction = "淨流入" if row["Dominant flow value"] >= 0 else "淨流出"
    pace = "加速" if row["Flow acceleration"] > 0 else "減速"
    confirmation = (
        f"價格已確認（20日 {row['20D return']:+.1%}）"
        if row["20D return"] > 0
        else f"價格尚未確認（20日 {row['20D return']:+.1%}）"
    )
    return (
        f"{row['Dominant investor']}主導{direction}"
        f"（20日 {row['Dominant flow value'] / 1e8:+.1f}億）；"
        f"近5日相較20日均速{pace}；"
        f"流入廣度 {row['Positive flow breadth']:.0%}；"
        f"前三大帶動標的占絕對流量 {row['Top 3 concentration']:.0%}；"
        f"{confirmation}。主要標的：{row['Leading stocks']}。"
    )


def calculate_fund_flow_signals(
    prices: pd.DataFrame,
    flows: pd.DataFrame,
    master: pd.DataFrame,
    *,
    short_window: int = 5,
    long_window: int = 20,
    weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate security- and industry-level institutional-flow signals.

    Net shares are converted to estimated TWD using each session's closing
    price. This is a transparent proxy, not an exact transaction cash flow.
    """
    if flows.empty or prices.empty:
        return pd.DataFrame(), pd.DataFrame()
    weights = weights or {
        "20D flow intensity": 40.0,
        "5D flow intensity": 30.0,
        "Trust 20D intensity": 20.0,
        "20D return": 10.0,
    }
    flow_dates = pd.to_datetime(flows["Date"]).dt.normalize()
    cutoff = pd.Timestamp(prices.index.max()).normalize()
    flows = flows[flow_dates <= cutoff].copy()
    if flows.empty:
        return pd.DataFrame(), pd.DataFrame()
    available_dates = sorted(pd.to_datetime(flows["Date"]).dt.normalize().unique())
    selected_dates = available_dates[-long_window:]
    short_dates = set(selected_dates[-short_window:])
    daily_dates = {selected_dates[-1]}
    selected = flows[pd.to_datetime(flows["Date"]).dt.normalize().isin(selected_dates)].copy()
    selected = selected[selected["Ticker"].isin(prices.columns)]
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()

    close_long = (
        prices.reindex(pd.to_datetime(selected_dates))
        .ffill()
        .stack(future_stack=True)
        .rename("Close")
        .rename_axis(["Date", "Ticker"])
        .reset_index()
    )
    selected["Date"] = pd.to_datetime(selected["Date"]).dt.normalize()
    selected = selected.merge(close_long, on=["Date", "Ticker"], how="left")
    for investor in ["Foreign", "Trust", "Dealer", "Total"]:
        selected[f"{investor} net value"] = (
            selected[f"{investor} net shares"] * selected["Close"]
        )

    long_agg = selected.groupby("Ticker").agg(
        Name=("Name", "last"),
        Market=("Market", "last"),
        **{
            "Foreign 20D value": ("Foreign net value", "sum"),
            "Trust 20D value": ("Trust net value", "sum"),
            "Dealer 20D value": ("Dealer net value", "sum"),
            "20D net value": ("Total net value", "sum"),
            "20D net shares": ("Total net shares", "sum"),
            "Trust 20D shares": ("Trust net shares", "sum"),
        },
    )
    short_agg = (
        selected[selected["Date"].isin(short_dates)]
        .groupby("Ticker")
        .agg(
            **{
                "Foreign 5D value": ("Foreign net value", "sum"),
                "Trust 5D value": ("Trust net value", "sum"),
                "Dealer 5D value": ("Dealer net value", "sum"),
                "5D net value": ("Total net value", "sum"),
                "5D net shares": ("Total net shares", "sum"),
                "Trust 5D shares": ("Trust net shares", "sum"),
            }
        )
    )
    daily_agg = (
        selected[selected["Date"].isin(daily_dates)]
        .groupby("Ticker")
        .agg(
            **{
                "Foreign 1D value": ("Foreign net value", "sum"),
                "Trust 1D value": ("Trust net value", "sum"),
                "Dealer 1D value": ("Dealer net value", "sum"),
                "1D net value": ("Total net value", "sum"),
                "1D net shares": ("Total net shares", "sum"),
                "Trust 1D shares": ("Trust net shares", "sum"),
            }
        )
    )
    securities = long_agg.join(short_agg, how="left").join(
        daily_agg,
        how="left",
    ).fillna(0.0)

    metadata = master.drop_duplicates("Yahoo ticker").set_index("Yahoo ticker")
    securities = securities.join(
        metadata[["Industry", "Asset type", "Issued shares"]],
        how="left",
    )
    latest_close = prices.ffill().iloc[-1]
    securities["Close"] = securities.index.map(latest_close.to_dict())
    securities["Market cap proxy"] = (
        pd.to_numeric(securities["Issued shares"], errors="coerce")
        * securities["Close"]
    )
    shares = pd.to_numeric(securities["Issued shares"], errors="coerce").replace(0, np.nan)
    securities["1D flow intensity"] = securities["1D net shares"] / shares
    securities["5D flow intensity"] = securities["5D net shares"] / shares
    securities["20D flow intensity"] = securities["20D net shares"] / shares
    securities["Trust 1D intensity"] = securities["Trust 1D shares"] / shares
    securities["Trust 5D intensity"] = securities["Trust 5D shares"] / shares
    securities["Trust 20D intensity"] = securities["Trust 20D shares"] / shares
    securities["1D return"] = prices.pct_change(1).iloc[-1].reindex(securities.index)
    securities["5D return"] = prices.pct_change(short_window).iloc[-1].reindex(
        securities.index
    )
    securities["20D return"] = prices.pct_change(long_window).iloc[-1].reindex(
        securities.index
    )
    securities["Flow score"] = _rank_score(securities, weights)
    securities["Stage"] = securities.apply(_flow_stage, axis=1)
    securities["Signal date"] = pd.Timestamp(selected_dates[-1])
    securities.index.name = "Ticker"
    securities = securities.reset_index().sort_values("Flow score", ascending=False)

    valid_groups = securities.dropna(subset=["Industry"]).copy()
    groups = valid_groups.groupby("Industry").agg(
        Constituents=("Ticker", "nunique"),
        **{
            "1D net value": ("1D net value", "sum"),
            "5D net value": ("5D net value", "sum"),
            "20D net value": ("20D net value", "sum"),
            "Foreign 1D value": ("Foreign 1D value", "sum"),
            "Trust 1D value": ("Trust 1D value", "sum"),
            "Dealer 1D value": ("Dealer 1D value", "sum"),
            "Foreign 5D value": ("Foreign 5D value", "sum"),
            "Trust 5D value": ("Trust 5D value", "sum"),
            "Dealer 5D value": ("Dealer 5D value", "sum"),
            "Foreign 20D value": ("Foreign 20D value", "sum"),
            "Trust 20D value": ("Trust 20D value", "sum"),
            "Dealer 20D value": ("Dealer 20D value", "sum"),
            "Market cap proxy": ("Market cap proxy", "sum"),
            "1D return": ("1D return", "mean"),
            "5D return": ("5D return", "mean"),
            "20D return": ("20D return", "mean"),
            "1D positive breadth": ("1D net value", lambda values: (values > 0).mean()),
            "5D positive breadth": ("5D net value", lambda values: (values > 0).mean()),
            "20D positive breadth": ("20D net value", lambda values: (values > 0).mean()),
        },
    )
    groups["1D flow intensity"] = groups["1D net value"] / groups["Market cap proxy"]
    groups["5D flow intensity"] = groups["5D net value"] / groups["Market cap proxy"]
    groups["20D flow intensity"] = groups["20D net value"] / groups["Market cap proxy"]
    groups["Trust 1D intensity"] = groups["Trust 1D value"] / groups["Market cap proxy"]
    groups["Trust 5D intensity"] = groups["Trust 5D value"] / groups["Market cap proxy"]
    groups["Trust 20D intensity"] = groups["Trust 20D value"] / groups["Market cap proxy"]
    groups["Positive flow breadth"] = groups["20D positive breadth"]
    group_weights = {
        "20D flow intensity": weights.get("20D flow intensity", 0),
        "5D flow intensity": weights.get("5D flow intensity", 0),
        "Trust 20D intensity": weights.get("Trust 20D intensity", 0),
        "20D return": weights.get("20D return", 0),
    }
    groups["Flow score"] = _rank_score(groups, group_weights)
    groups["Stage"] = groups.apply(_flow_stage, axis=1)
    leaders = (
        securities.sort_values(["Industry", "Flow score"], ascending=[True, False])
        .groupby("Industry")["Ticker"]
        .apply(lambda values: "、".join(values.head(3)))
    )
    groups["Leading stocks"] = leaders
    investor_columns = {
        "外資": "Foreign 20D value",
        "投信": "Trust 20D value",
        "自營商": "Dealer 20D value",
    }
    investor_values = groups[list(investor_columns.values())]
    dominant_columns = investor_values.abs().idxmax(axis=1)
    reverse_investor_columns = {column: name for name, column in investor_columns.items()}
    groups["Dominant investor"] = dominant_columns.map(reverse_investor_columns)
    groups["Dominant flow value"] = [
        groups.loc[index, column]
        for index, column in dominant_columns.items()
    ]
    groups["Flow acceleration"] = groups["5D net value"] - groups["20D net value"] / 4
    concentration = (
        valid_groups.assign(Absolute_flow=valid_groups["20D net value"].abs())
        .groupby("Industry")["Absolute_flow"]
        .apply(
            lambda values: (
                values.nlargest(3).sum() / values.sum()
                if values.sum() > 0
                else 0.0
            )
        )
    )
    groups["Top 3 concentration"] = groups.index.map(concentration.to_dict())
    groups["Research action"] = groups["Stage"].map(_research_action)
    groups["Flow reason"] = groups.apply(_flow_reason, axis=1)
    groups["Signal date"] = pd.Timestamp(selected_dates[-1])
    groups = groups.reset_index().sort_values("Flow score", ascending=False)
    return securities, groups


def calculate_daily_group_flows(
    prices: pd.DataFrame,
    flows: pd.DataFrame,
    master: pd.DataFrame,
    *,
    periods: int = 20,
) -> pd.DataFrame:
    """Return daily investor net-value estimates aggregated by industry."""
    if prices.empty or flows.empty:
        return pd.DataFrame()
    selected = flows.copy()
    selected["Date"] = pd.to_datetime(selected["Date"]).dt.normalize()
    selected = selected[selected["Ticker"].isin(prices.columns)]
    selected_dates = sorted(selected["Date"].unique())[-periods:]
    selected = selected[selected["Date"].isin(selected_dates)]
    closes = (
        prices.reindex(pd.to_datetime(selected_dates))
        .ffill()
        .stack(future_stack=True)
        .rename("Close")
        .rename_axis(["Date", "Ticker"])
        .reset_index()
    )
    selected = selected.merge(closes, on=["Date", "Ticker"], how="left")
    metadata = (
        master.drop_duplicates("Yahoo ticker")
        .set_index("Yahoo ticker")[["Industry"]]
    )
    selected["Industry"] = selected["Ticker"].map(metadata["Industry"].to_dict())
    for investor in ["Foreign", "Trust", "Dealer", "Total"]:
        selected[f"{investor} net value"] = (
            selected[f"{investor} net shares"] * selected["Close"]
        )
    return (
        selected.dropna(subset=["Industry"])
        .groupby(["Date", "Industry"], as_index=False)[
            [
                "Foreign net value",
                "Trust net value",
                "Dealer net value",
                "Total net value",
            ]
        ]
        .sum()
        .sort_values(["Date", "Industry"])
    )
