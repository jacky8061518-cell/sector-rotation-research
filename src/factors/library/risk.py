"""Realized-risk and lottery-effect factors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..context import DataContext
from ..registry import register_factor
from ..spec import FactorSpec
from ._price import adjusted_close


@register_factor
class RealizedVolatilitySixtyDay:
    spec = FactorSpec(
        name="vol_60d",
        label="60 日實現波動",
        category="risk",
        direction=-1,
        lookback_days=61,
        requires=("prices",),
        markets=("US", "TW"),
        description="近六十個交易日報酬標準差年化，低波動分數較高。",
    )

    def compute(self, ctx: DataContext, asof: pd.Timestamp) -> pd.Series:
        prices = adjusted_close(ctx, asof, self.spec.lookback_days)
        result = prices.pct_change(fill_method=None).iloc[-60:].std(ddof=1) * np.sqrt(252)
        return result.reindex(ctx.universe()).rename(self.spec.name)


@register_factor
class MarketBetaTwoHundredFiftyTwoDay:
    spec = FactorSpec(
        name="beta_252d",
        label="252 日市場 Beta",
        category="risk",
        direction=-1,
        lookback_days=253,
        requires=("prices",),
        markets=("US", "TW"),
        description="相對市場基準的 252 日 beta，低 beta 分數較高。",
    )

    def compute(self, ctx: DataContext, asof: pd.Timestamp) -> pd.Series:
        prices = adjusted_close(ctx, asof, self.spec.lookback_days)
        if ctx.benchmark not in prices or len(prices) < 253:
            return pd.Series(np.nan, index=ctx.universe(), name=self.spec.name)
        returns = prices.pct_change(fill_method=None).iloc[-252:]
        benchmark = returns[ctx.benchmark]
        variance = float(benchmark.var(ddof=1))
        if not np.isfinite(variance) or variance == 0:
            return pd.Series(np.nan, index=ctx.universe(), name=self.spec.name)
        result = returns.cov()[ctx.benchmark].div(variance)
        return result.reindex(ctx.universe()).rename(self.spec.name)


@register_factor
class MaximumReturnFiveDay:
    spec = FactorSpec(
        name="max_ret_5d",
        label="近五日最大單日報酬",
        category="risk",
        direction=-1,
        lookback_days=6,
        requires=("prices",),
        markets=("US", "TW"),
        description="捕捉近期極端正報酬的彩券偏好，較低數值分數較高。",
        reference="Bali, Cakici, and Whitelaw (2011)",
    )

    def compute(self, ctx: DataContext, asof: pd.Timestamp) -> pd.Series:
        prices = adjusted_close(ctx, asof, self.spec.lookback_days)
        result = prices.pct_change(fill_method=None).iloc[-5:].max()
        return result.reindex(ctx.universe()).rename(self.spec.name)
