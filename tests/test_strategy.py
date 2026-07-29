import numpy as np
import pandas as pd

from sector_rotation.strategy import (
    BacktestConfig,
    compute_momentum_scores,
    run_backtest,
    to_monthly_prices,
)


def sample_prices(months: int = 30) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=months, freq="ME")
    return pd.DataFrame(
        {
            "AAA": 100 * np.cumprod(np.repeat(1.02, months)),
            "BBB": 100 * np.cumprod(np.repeat(0.99, months)),
            "SHY": 100 * np.cumprod(np.repeat(1.001, months)),
        },
        index=index,
    )


def test_monthly_conversion_uses_last_observation():
    daily = pd.DataFrame(
        {"AAA": [10.0, 11.0, 12.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-01"]),
    )
    monthly = to_monthly_prices(daily)
    assert monthly.loc[pd.Timestamp("2024-01-31"), "AAA"] == 11.0


def test_composite_score_identifies_stronger_asset():
    prices = sample_prices()
    scores = compute_momentum_scores(prices, ["AAA", "BBB"], {3: 0.5, 6: 0.5})
    latest = scores.dropna().iloc[-1]
    assert latest["AAA"] > 0
    assert latest["AAA"] > latest["BBB"]


def test_backtest_shifts_signal_to_next_month():
    prices = sample_prices()
    config = BacktestConfig({3: 1.0}, top_n=1, defensive_asset="SHY")
    result = run_backtest(prices, ["AAA", "BBB"], config)
    assert (result.deployed_weights["AAA"].iloc[1:] == 1.0).all()
    assert result.net_returns.iloc[1] > 0


def test_negative_momentum_moves_to_defensive_asset():
    prices = sample_prices()
    falling = prices.copy()
    falling["AAA"] = 100 * np.cumprod(np.repeat(0.98, len(falling)))
    config = BacktestConfig({3: 1.0}, top_n=1, defensive_asset="SHY")
    result = run_backtest(falling, ["AAA", "BBB"], config)
    assert result.target_weights["SHY"].iloc[-1] == 1.0


def test_weights_sum_to_one_after_signals_are_available():
    prices = sample_prices()
    config = BacktestConfig({3: 1.0}, top_n=2, defensive_asset="SHY")
    result = run_backtest(prices, ["AAA", "BBB"], config)
    sums = result.target_weights.sum(axis=1)
    assert np.allclose(sums, 1.0)
