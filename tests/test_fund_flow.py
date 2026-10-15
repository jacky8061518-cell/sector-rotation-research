import numpy as np
import pandas as pd

from sector_rotation.fund_flow import (
    backtest_institutional_first_buy_events,
    calculate_daily_group_flows,
    calculate_fund_flow_signals,
    detect_new_institutional_buyers,
)


def test_backtests_first_trust_buy_from_next_session_without_lookahead():
    dates = pd.bdate_range("2021-01-01", periods=320)
    prices = pd.DataFrame(
        {
            "1111.TW": np.linspace(100, 200, len(dates)),
            "0050.TW": np.linspace(100, 125, len(dates)),
        },
        index=dates,
    )
    trust = np.zeros(len(dates))
    trust[25] = 10_000
    flows = pd.DataFrame(
        {
            "Date": dates,
            "Ticker": "1111.TW",
            "Name": "測試股",
            "Market": "上市",
            "Foreign net shares": 0,
            "Trust net shares": trust,
            "Dealer net shares": 0,
        }
    )

    events, summary = backtest_institutional_first_buy_events(
        prices,
        flows,
        investor="投信",
        lookback_sessions=20,
        minimum_latest_net_shares=1000,
        years=5,
        eligible_tickers={"1111.TW"},
    )

    assert len(events) == 1
    assert events.iloc[0]["Signal date"] == dates[25]
    assert events.iloc[0]["Entry date"] == dates[26]
    expected = prices.loc[dates[47], "1111.TW"] / prices.loc[dates[26], "1111.TW"] - 1
    assert np.isclose(events.iloc[0]["1個月報酬"], expected)
    assert summary.loc[summary["持有期間"] == "1個月", "勝率"].iloc[0] == 1.0


def test_first_buy_backtest_excludes_repeat_positive_buy_days():
    dates = pd.bdate_range("2024-01-01", periods=80)
    prices = pd.DataFrame(
        {"1111.TW": np.linspace(100, 110, len(dates)), "0050.TW": 100.0},
        index=dates,
    )
    trust = np.zeros(len(dates))
    trust[25:28] = 1000
    flows = pd.DataFrame(
        {
            "Date": dates, "Ticker": "1111.TW", "Name": "測試股", "Market": "上市",
            "Foreign net shares": 0, "Trust net shares": trust, "Dealer net shares": 0,
        }
    )
    events, _ = backtest_institutional_first_buy_events(
        prices, flows, lookback_sessions=20, years=5
    )
    assert events["Signal date"].tolist() == [dates[25]]


def test_full_history_first_buy_only_accepts_first_positive_day_after_enough_history():
    dates = pd.bdate_range("2021-01-01", periods=320)
    prices = pd.DataFrame(
        {"1111.TW": np.linspace(100, 140, len(dates)), "0050.TW": 100.0},
        index=dates,
    )
    trust = np.zeros(len(dates))
    trust[260] = 5000
    trust[280] = 8000
    flows = pd.DataFrame(
        {
            "Date": dates, "Ticker": "1111.TW", "Name": "歷史首買", "Market": "上市",
            "Foreign net shares": 0, "Trust net shares": trust, "Dealer net shares": 0,
        }
    )

    events, _ = backtest_institutional_first_buy_events(
        prices,
        flows,
        investor="投信",
        full_history_no_prior_buying=True,
        minimum_history_sessions=252,
        minimum_latest_net_shares=1000,
        years=5,
    )

    assert events["Signal date"].tolist() == [dates[260]]


def test_full_history_screener_rejects_stock_with_any_earlier_positive_buy():
    dates = pd.bdate_range("2024-01-01", periods=65)
    rows = []
    for ticker in ("1111.TW", "2222.TW"):
        for session in dates:
            rows.append(
                {
                    "Date": session, "Ticker": ticker, "Name": ticker, "Market": "上市",
                    "Foreign net shares": 0, "Trust net shares": 0, "Dealer net shares": 0,
                }
            )
    flows = pd.DataFrame(rows)
    flows.loc[(flows["Ticker"] == "1111.TW") & (flows["Date"] == dates[-1]), "Trust net shares"] = 5000
    flows.loc[(flows["Ticker"] == "2222.TW") & (flows["Date"] == dates[20]), "Trust net shares"] = 10
    flows.loc[(flows["Ticker"] == "2222.TW") & (flows["Date"] == dates[-1]), "Trust net shares"] = 5000

    result = detect_new_institutional_buyers(
        flows,
        investors=("投信",),
        minimum_latest_net_shares=1000,
        full_history_no_prior_buying=True,
        minimum_history_sessions=60,
    )

    assert result["Ticker"].tolist() == ["1111.TW"]
    assert result.iloc[0]["History sessions"] == 64


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
