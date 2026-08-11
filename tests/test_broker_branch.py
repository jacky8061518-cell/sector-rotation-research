import pandas as pd

from sector_rotation.broker_branch import (
    aggregate_branch_activity,
    build_broker_research_candidates,
    normalize_broker_branch_trades,
    parse_histock_branch_page,
    parse_histock_weekly_branch_page,
)


def sample_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-08-07"] * 6,
            "stock_id": ["2330", "2330", "2330", "2408", "2408", "2408"],
            "securities_trader_id": ["A", "B", "C", "A", "B", "C"],
            "securities_trader": ["甲", "乙", "丙", "甲", "乙", "丙"],
            "price": [100, 100, 100, 50, 50, 50],
            "buy": [1000, 200, 0, 3000, 100, 0],
            "sell": [0, 100, 1100, 0, 0, 3100],
        }
    )


def test_normalize_finmind_schema():
    normalized = normalize_broker_branch_trades(sample_trades())
    assert list(normalized.columns) == [
        "Date", "Ticker", "Broker ID", "Broker", "Price", "Buy shares", "Sell shares",
        "Horizon",
    ]
    assert normalized["Ticker"].tolist()[0] == "2330"
    assert normalized["Horizon"].eq("Daily").all()


def test_branch_activity_ranks_concentrated_buying():
    branches, stocks = aggregate_branch_activity(sample_trades())
    assert not branches.empty
    assert set(stocks["Ticker"]) == {"2330", "2408"}
    assert stocks.iloc[0]["Top buyer"] == "甲"
    assert stocks.iloc[0]["Positive branch ratio"] > 0
    assert stocks.iloc[0]["Buyer seller strength"] > 0


def test_candidates_require_confirmation_for_priority():
    _, stocks = aggregate_branch_activity(sample_trades())
    institutional = pd.DataFrame(
        {
            "Ticker": ["2330.TW", "2408.TW"],
            "Selected net value": [1e9, -1e8],
            "Selected return": [0.05, -0.02],
            "Flow score": [90, 40],
        }
    )
    master = pd.DataFrame(
        {
            "Code": ["2330", "2408"],
            "Yahoo ticker": ["2330.TW", "2408.TW"],
            "Name": ["台積電", "南亞科"],
            "Industry": ["半導體業", "半導體業"],
        }
    )
    candidates = build_broker_research_candidates(stocks, institutional, master)
    tsmc = candidates[candidates["Ticker"] == "2330"].iloc[0]
    assert tsmc["Recommendation"] == "優先研究"
    assert "最大買超分點" in tsmc["Reason"]


def test_parses_histock_weekly_branch_table():
    html = """
    <div>2026/08/03 ~ 2026/08/10</div>
    <table><tr>
      <td>賣超分點</td><td>100</td><td>500</td><td>-400</td><td>52.5</td>
      <td>買超分點</td><td>700</td><td>100</td><td>600</td><td>51.8</td>
    </tr></table>
    """
    parsed = parse_histock_weekly_branch_page(html, "2408")
    assert set(parsed["Broker"]) == {"買超分點", "賣超分點"}
    buyer = parsed[parsed["Broker"] == "買超分點"].iloc[0]
    assert buyer["Buy shares"] == 700_000
    assert buyer["Date"] == pd.Timestamp("2026-08-10")
    assert buyer["Horizon"] == "Weekly"


def test_parser_preserves_requested_monthly_horizon():
    html = """
    <div>2026/07/11 ~ 2026/08/10</div>
    <table><tr>
      <td>賣超分點</td><td>10</td><td>20</td><td>-10</td><td>52.5</td>
      <td>買超分點</td><td>30</td><td>10</td><td>20</td><td>51.8</td>
    </tr></table>
    """
    parsed = parse_histock_branch_page(html, "2408", "Monthly")
    assert parsed["Horizon"].eq("Monthly").all()
