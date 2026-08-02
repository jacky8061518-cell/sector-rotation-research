import numpy as np
import pandas as pd

from sector_rotation.fund_flow import (
    calculate_daily_group_flows,
    calculate_fund_flow_signals,
)


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
