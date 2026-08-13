"""Explicit, configurable transaction-cost models."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .spec import Market


@dataclass(frozen=True)
class CostConfig:
    tw_commission_rate: float = 0.001425
    tw_commission_discount: float = 0.60
    tw_sell_tax_rate: float = 0.003
    tw_slippage_rate: float = 0.001
    us_commission_rate: float = 0.0
    us_slippage_rate: float = 0.0005

    def __post_init__(self) -> None:
        values = vars(self).values()
        if any(value < 0 for value in values):
            raise ValueError("Cost rates cannot be negative.")
        if self.tw_commission_discount > 1:
            raise ValueError("Commission discount must be within [0, 1].")


DEFAULT_COST_CONFIG = CostConfig()


def transaction_cost(
    previous: pd.Series,
    target: pd.Series,
    market: Market,
    config: CostConfig = DEFAULT_COST_CONFIG,
) -> tuple[float, float, float]:
    """Return total, buy, and sell costs as fractions of portfolio value."""
    tickers = previous.index.union(target.index)
    delta = target.reindex(tickers, fill_value=0.0) - previous.reindex(tickers, fill_value=0.0)
    buy_turnover = float(delta.clip(lower=0).sum())
    sell_turnover = float((-delta.clip(upper=0)).sum())
    if market == "TW":
        commission = config.tw_commission_rate * config.tw_commission_discount
        buy_rate = commission + config.tw_slippage_rate
        sell_rate = commission + config.tw_sell_tax_rate + config.tw_slippage_rate
    else:
        buy_rate = config.us_commission_rate + config.us_slippage_rate
        sell_rate = buy_rate
    return (
        buy_turnover * buy_rate + sell_turnover * sell_rate,
        buy_turnover,
        sell_turnover,
    )
