"""Liquidity and size factors that activate when point-in-time fields exist."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..context import DataContext
from ..registry import register_factor
from ..spec import FactorSpec


@register_factor
class TurnoverTwentyDay:
    spec = FactorSpec(
        name="turnover_20d",
        label="20 日成交值週轉率",
        category="liquidity",
        direction=-1,
        lookback_days=20,
        requires=("prices.traded_value", "market_cap"),
        markets=("US", "TW"),
        description="二十日日均成交金額除以同期市值；低週轉分數較高。",
    )

    def compute(self, ctx: DataContext, asof: pd.Timestamp) -> pd.Series:
        try:
            traded = ctx.prices(("traded_value",), 20)["traded_value"].groupby(level="ticker").mean()
        except ValueError:
            return pd.Series(np.nan, index=ctx.universe(), name=self.spec.name)
        cap = ctx.market_cap(1)
        latest_cap = cap["market_cap"].groupby(level="ticker").last() if "market_cap" in cap else pd.Series(dtype=float)
        return traded.div(latest_cap).reindex(ctx.universe()).rename(self.spec.name)


@register_factor
class LogMarketCapitalization:
    spec = FactorSpec(
        name="size_ln_mcap",
        label="市值規模",
        category="liquidity",
        direction=-1,
        lookback_days=1,
        requires=("market_cap",),
        markets=("US", "TW"),
        description="流通市值自然對數；較小公司分數較高。",
    )

    def compute(self, ctx: DataContext, asof: pd.Timestamp) -> pd.Series:
        cap = ctx.market_cap(1)
        values = cap["market_cap"].groupby(level="ticker").last() if "market_cap" in cap else pd.Series(dtype=float)
        return np.log(values.where(values > 0)).reindex(ctx.universe()).rename(self.spec.name)
