from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.backtest import run_factor_backtest
from factors.costs import CostConfig, transaction_cost
from factors.portfolio import PortfolioConfig, assign_quantiles, build_weights


def test_quantiles_and_dollar_neutral_weights() -> None:
    scores = pd.Series(range(20), index=[f"S{i:02d}" for i in range(20)], dtype=float)
    groups = assign_quantiles(scores)
    weights = build_weights(
        scores,
        PortfolioConfig(minimum_holdings=4, maximum_stock_weight=0.30),
    )

    assert groups.value_counts().sort_index().tolist() == [4, 4, 4, 4, 4]
    assert weights.clip(lower=0).sum() == pytest.approx(1.0)
    assert (-weights.clip(upper=0)).sum() == pytest.approx(1.0)


def test_portfolio_caps_survive_redistribution() -> None:
    scores = pd.Series(range(50), index=[f"S{i:02d}" for i in range(50)], dtype=float)
    industry = pd.Series(
        [["A", "B", "C"][index % 3] for index in range(50)],
        index=scores.index,
    )
    weights = build_weights(
        scores,
        PortfolioConfig(
            long_short=False,
            minimum_holdings=10,
            maximum_stock_weight=0.15,
            maximum_industry_weight=0.60,
        ),
        industry=industry,
    )

    assert weights.sum() == pytest.approx(1.0)
    assert weights.max() <= 0.15 + 1e-10
    assert weights.groupby(industry.reindex(weights.index)).sum().max() <= 0.60 + 1e-10


def test_taiwan_cost_model_is_asymmetric() -> None:
    previous = pd.Series({"A": 1.0})
    target = pd.Series({"B": 1.0})
    total, buys, sells = transaction_cost(
        previous,
        target,
        "TW",
        CostConfig(tw_commission_discount=1.0, tw_slippage_rate=0.0),
    )

    assert buys == pytest.approx(1.0)
    assert sells == pytest.approx(1.0)
    assert total == pytest.approx(0.001425 * 2 + 0.003)


def test_backtest_executes_after_signal_and_zero_cost_is_identity() -> None:
    dates = pd.bdate_range("2025-01-01", periods=8)
    tickers = [f"S{i:02d}" for i in range(10)]
    values = np.full((8, 10), 100.0)
    values[2:, 5:] = 110.0
    prices = pd.DataFrame(values, index=dates, columns=tickers)
    prices.index.name = "date"
    scores = pd.DataFrame([range(10)], index=[dates[0]], columns=tickers, dtype=float)
    result = run_factor_backtest(
        scores,
        prices,
        "TW",
        PortfolioConfig(long_short=False, minimum_holdings=2, maximum_stock_weight=0.60),
        CostConfig(
            tw_commission_rate=0,
            tw_commission_discount=0,
            tw_sell_tax_rate=0,
            tw_slippage_rate=0,
        ),
    )

    assert result.holdings["date"].min() == dates[1]
    assert dates[0] not in set(result.holdings["date"])
    pd.testing.assert_series_equal(result.returns["gross"], result.returns["net"], check_names=False)
