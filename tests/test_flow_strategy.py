import numpy as np
import pandas as pd

from sector_rotation.flow_strategy import FlowStrategyConfig, run_weekly_flow_strategy


def strategy_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2026-01-05", periods=35)
    tickers = [f"{number:04d}.TW" for number in range(1001, 1008)]
    prices = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for offset, ticker in enumerate(tickers):
        prices[ticker] = 50 + offset + np.arange(len(dates)) * (0.2 + offset * 0.01)
    # Force the first leader below its MA after it has been selected.
    prices.loc[dates[-3]:, tickers[0]] = [45.0, 44.0, 43.0]

    rows = []
    flow_strength = dict(zip(tickers, [700, 600, 500, 400, 300, 200, 100], strict=True))
    for session in dates:
        for ticker in tickers:
            rows.append(
                {
                    "Date": session,
                    "Ticker": ticker,
                    "Name": ticker,
                    "Market": "上市",
                    "Foreign net shares": flow_strength[ticker],
                    "Trust net shares": 0,
                    "Dealer net shares": 0,
                    "Total net shares": flow_strength[ticker],
                }
            )
    flows = pd.DataFrame(rows)
    master = pd.DataFrame(
        {
            "Yahoo ticker": tickers,
            "Name": [f"Company {number}" for number in range(len(tickers))],
            "Industry": "測試產業",
            "Detailed industry": "測試細分",
            "Investment theme": "測試主題",
            "Supply-chain role": "測試角色",
        }
    )
    return prices, flows, master


def test_weekly_flow_strategy_selects_top_five_and_shifts_returns():
    prices, flows, master = strategy_inputs()
    result = run_weekly_flow_strategy(
        prices,
        flows,
        master,
        FlowStrategyConfig(top_n=5, ma_window=10),
    )
    selected = result.latest_candidates[result.latest_candidates["Selected"]]
    assert len(selected) <= 5
    assert not result.weekly_rankings.empty
    assert result.deployed_weights.iloc[0].sum() == 0
    assert result.net_returns.index.equals(prices.index)
    assert (result.turnover >= 0).all()


def test_weekly_flow_strategy_records_moving_average_exit():
    prices, flows, master = strategy_inputs()
    result = run_weekly_flow_strategy(
        prices,
        flows,
        master,
        FlowStrategyConfig(top_n=5, ma_window=10),
    )
    assert result.trade_log["Action"].str.contains("跌破10日線").any()
    exit_rows = result.trade_log[result.trade_log["Action"].str.contains("跌破10日線")]
    assert (exit_rows["Execution date"] >= exit_rows["Signal date"]).all()


def test_new_high_mode_only_selects_new_high_candidates():
    prices, flows, master = strategy_inputs()
    result = run_weekly_flow_strategy(
        prices,
        flows,
        master,
        FlowStrategyConfig(
            top_n=5,
            ma_window=10,
            entry_mode="New high + inflow",
            new_high_window=20,
        ),
    )
    selected = result.weekly_rankings[result.weekly_rankings["Selected"]]
    assert not selected.empty
    assert selected["New high"].all()
