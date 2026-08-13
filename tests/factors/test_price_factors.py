from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from factors.context import DataContext
from factors.registry import get_factor


@pytest.mark.parametrize("name", ["mom_12_1", "mom_6m", "vol_60d", "beta_252d", "max_ret_5d"])
def test_each_price_factor_returns_the_bound_universe(name: str, context: DataContext) -> None:
    values = get_factor(name).compute(context, context.asof.normalize())

    assert values.index.tolist() == context.universe().tolist()
    assert values.name == name


def test_momentum_known_answers(context: DataContext) -> None:
    prices = context.prices(("adj_close",))["adj_close"].unstack("ticker")
    twelve_one = get_factor("mom_12_1").compute(context, context.asof.normalize())
    six_month = get_factor("mom_6m").compute(context, context.asof.normalize())

    expected_twelve_one = prices["AAA.TW"].iloc[-22] / prices["AAA.TW"].iloc[-253] - 1
    expected_six_month = prices["AAA.TW"].iloc[-1] / prices["AAA.TW"].iloc[-127] - 1
    assert np.isclose(twelve_one["AAA.TW"], expected_twelve_one)
    assert np.isclose(six_month["AAA.TW"], expected_six_month)


def test_volatility_beta_and_max_return_known_answers(context: DataContext) -> None:
    prices = context.prices(("adj_close",))["adj_close"].unstack("ticker")
    returns = prices.pct_change(fill_method=None)

    volatility = get_factor("vol_60d").compute(context, context.asof.normalize())
    beta = get_factor("beta_252d").compute(context, context.asof.normalize())
    maximum = get_factor("max_ret_5d").compute(context, context.asof.normalize())

    benchmark = returns["0050.TW"].iloc[-252:]
    expected_beta = returns["AAA.TW"].iloc[-252:].cov(benchmark) / benchmark.var()
    assert np.isclose(volatility["AAA.TW"], returns["AAA.TW"].iloc[-60:].std() * np.sqrt(252))
    assert np.isclose(beta["AAA.TW"], expected_beta)
    assert np.isclose(maximum["BBB.TW"], returns["BBB.TW"].iloc[-5:].max())


def test_factor_rejects_mismatched_asof(context: DataContext) -> None:
    with pytest.raises(ValueError, match="must match"):
        get_factor("mom_6m").compute(context, context.asof - timedelta(days=5))
