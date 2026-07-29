import numpy as np
import pandas as pd
import pytest

from sector_rotation.data import download_adjusted_prices
from sector_rotation.strategy import (
    BacktestConfig,
    compute_momentum_scores,
    resample_prices,
    run_backtest,
    to_monthly_prices,
)
from sector_rotation.universe import UNIVERSE_GROUPS, assets_for


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


def test_market_data_rejects_empty_date_range_before_network_call():
    same_day = pd.Timestamp("2026-07-29").date()
    with pytest.raises(ValueError, match="end date must be later"):
        download_adjusted_prices(["SPY"], same_day, same_day)


def test_daily_frequency_keeps_every_observation():
    daily = pd.DataFrame(
        {"AAA": [10.0, 11.0, 12.0]},
        index=pd.bdate_range("2024-01-02", periods=3),
    )
    sampled = resample_prices(daily, "Daily")
    assert len(sampled) == 3


def test_weekly_frequency_uses_last_observation_of_week():
    daily = pd.DataFrame(
        {"AAA": np.arange(1.0, 11.0)},
        index=pd.bdate_range("2024-01-01", periods=10),
    )
    sampled = resample_prices(daily, "Weekly")
    assert sampled.iloc[0, 0] == 5.0
    assert sampled.iloc[1, 0] == 10.0
    assert sampled.index[-1] == daily.index[-1]


def test_monthly_frequency_labels_partial_month_with_actual_date():
    daily = pd.DataFrame(
        {"AAA": [10.0, 11.0]},
        index=pd.to_datetime(["2026-07-28", "2026-07-29"]),
    )
    sampled = resample_prices(daily, "Monthly")
    assert sampled.index[-1] == pd.Timestamp("2026-07-29")


def test_daily_backtest_uses_daily_annualization_and_shift():
    index = pd.bdate_range("2024-01-01", periods=80)
    prices = pd.DataFrame(
        {
            "AAA": 100 * np.cumprod(np.repeat(1.002, len(index))),
            "BBB": 100 * np.cumprod(np.repeat(0.999, len(index))),
            "SHY": 100 * np.cumprod(np.repeat(1.0001, len(index))),
        },
        index=index,
    )
    result = run_backtest(
        prices,
        ["AAA", "BBB"],
        BacktestConfig({20: 1.0}, frequency="Daily", top_n=1),
    )
    assert len(result.net_returns) > 40
    assert result.deployed_weights["AAA"].iloc[1] == 1.0


def test_detailed_universe_retains_group_metadata():
    groups = list(UNIVERSE_GROUPS["Detailed industries — ETFs"])
    assets = assets_for("Detailed industries — ETFs", groups)
    assert len(assets) >= 30
    assert assets["SOXX"].group == "Technology"
