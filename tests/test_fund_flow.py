import numpy as np
import pandas as pd

from sector_rotation.fund_flow import (
    calculate_daily_group_flows,
    calculate_fund_flow_signals,
    detect_new_institutional_buyers,
)


def test_detects_first_institutional_buy_after_quiet_window():
    dates = pd.bdate_range("2026-01-01", periods=6)
    rows = []
    for session in dates:
        rows.extend(
            [
                {
                    "Date": session, "Ticker": "1111.TW", "Name": "首次買",
                    "Market": "上市", "Foreign net shares": 0,
                    "Trust net shares": 0, "Dealer net shares": 0,
                },
                {
                    "Date": session, "Ticker": "2222.TWO", "Name": "之前買過",
                    "Market": "上櫃", "Foreign net shares": 0,
                    "Trust net shares": 0, "Dealer net shares": 0,
                },
            ]
        )
    flows = pd.DataFrame(rows)
    flows.loc[(flows["Ticker"] == "1111.TW") & (flows["Date"] == dates[-1]), "Trust net shares"] = 5000
    flows.loc[(flows["Ticker"] == "2222.TWO") & (flows["Date"] == dates[-2]), "Foreign net shares"] = 10
    flows.loc[(flows["Ticker"] == "2222.TWO") & (flows["Date"] == dates[-1]), "Foreign net shares"] = 5000

    result = detect_new_institutional_buyers(
        flows,
        lookback_sessions=5,
        minimum_latest_net_shares=1000,
    )

    assert result["Ticker"].tolist() == ["1111.TW"]
    assert result.iloc[0]["Triggered investors"] == "投信"
    assert result.iloc[0]["Triggered latest net shares"] == 5000


def test_relaxed_first_buy_uses_nonpositive_prior_cumulative_flow():
    dates = pd.bdate_range("2026-01-01", periods=4)
    flows = pd.DataFrame(
        {
            "Date": dates,
            "Ticker": "1111.TW",
            "Name": "由賣轉買",
            "Market": "上市",
            "Foreign net shares": [-100, 20, -50, 200],
            "Trust net shares": 0,
            "Dealer net shares": 0,
        }
    )
    strict = detect_new_institutional_buyers(flows, lookback_sessions=3)
    relaxed = detect_new_institutional_buyers(
        flows,
        lookback_sessions=3,
        strict_no_prior_buying=False,
    )
    assert strict.empty
    assert relaxed["Ticker"].tolist() == ["1111.TW"]


def test_calculates_security_and_group_flow_leadership():
    dates = pd.bdate_range("2026-01-01", periods=25)
    prices = pd.DataFrame(
        {
            "1111.TW": np.linspace(100, 125, len(dates)),
            "2222.TW": np.linspace(100, 90, len(dates)),
        },
        index=dates,
    )
    rows = []
    for session in dates:
        rows.extend(
            [
                {
                    "Date": session,
                    "Ticker": "1111.TW",
                    "Name": "流入股",
                    "Market": "上市",
                    "Foreign net shares": 1000,
                    "Trust net shares": 500,
                    "Dealer net shares": 0,
                    "Total net shares": 1500,
                },
                {
                    "Date": session,
                    "Ticker": "2222.TW",
                    "Name": "流出股",
                    "Market": "上市",
                    "Foreign net shares": -1000,
                    "Trust net shares": -500,
                    "Dealer net shares": 0,
                    "Total net shares": -1500,
                },
            ]
        )
    flows = pd.DataFrame(rows)
    master = pd.DataFrame(
        {
            "Yahoo ticker": ["1111.TW", "2222.TW"],
            "Industry": ["流入產業", "流出產業"],
            "Asset type": ["股票", "股票"],
            "Issued shares": [1_000_000, 1_000_000],
        }
    )

    securities, groups = calculate_fund_flow_signals(prices, flows, master)

    assert securities.iloc[0]["Ticker"] == "1111.TW"
    assert securities.iloc[0]["Stage"] == "資金累積＋價格確認"
    assert groups.iloc[0]["Industry"] == "流入產業"
    assert groups.iloc[0]["20D net value"] > 0
    assert groups.iloc[-1]["20D net value"] < 0
    assert groups.iloc[0]["Dominant investor"] in {"外資", "投信", "自營商"}
    assert "主要標的" in groups.iloc[0]["Flow reason"]
    assert groups.iloc[0]["Research action"]

    daily = calculate_daily_group_flows(prices, flows, master)
    assert set(daily["Industry"]) == {"流入產業", "流出產業"}
    assert daily["Date"].nunique() == 20
