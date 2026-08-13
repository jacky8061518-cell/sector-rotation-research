"""Price momentum factors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..context import DataContext
from ..registry import register_factor
from ..spec import FactorSpec
from ._price import adjusted_close


@register_factor
class MomentumTwelveOne:
    spec = FactorSpec(
        name="mom_12_1",
        label="12–1 個月動能",
        category="momentum",
        direction=1,
        lookback_days=253,
        requires=("prices",),
        markets=("US", "TW"),
        description="比較約十二個月至一個月前的報酬，避開近期反轉。",
        reference="Jegadeesh and Titman (1993)",
    )

    def compute(self, ctx: DataContext, asof: pd.Timestamp) -> pd.Series:
        prices = adjusted_close(ctx, asof, self.spec.lookback_days)
        result = prices.iloc[-22].div(prices.iloc[-253]).sub(1) if len(prices) >= 253 else pd.Series(dtype=float)
        return result.reindex(ctx.universe()).rename(self.spec.name)


@register_factor
class MomentumSixMonth:
    spec = FactorSpec(
        name="mom_6m",
        label="六個月動能",
        category="momentum",
        direction=1,
        lookback_days=127,
        requires=("prices",),
        markets=("US", "TW"),
        description="衡量最近約六個交易月的累積調整後報酬。",
    )

    def compute(self, ctx: DataContext, asof: pd.Timestamp) -> pd.Series:
        prices = adjusted_close(ctx, asof, self.spec.lookback_days)
        result = prices.iloc[-1].div(prices.iloc[-127]).sub(1) if len(prices) >= 127 else pd.Series(dtype=float)
        return result.reindex(ctx.universe()).rename(self.spec.name)


@register_factor
class ResidualMomentumTwelveMonth:
    spec = FactorSpec(
        name="resid_mom_12m",
        label="市場／產業殘差動能",
        category="momentum",
        direction=1,
        lookback_days=253,
        requires=("prices", "industry"),
        markets=("US", "TW"),
        description="剔除市場 beta 與當期產業平均後的十二個月殘差報酬。",
    )

    def compute(self, ctx: DataContext, asof: pd.Timestamp) -> pd.Series:
        prices = adjusted_close(ctx, asof, self.spec.lookback_days)
        if len(prices) < 253 or ctx.benchmark not in prices:
            return pd.Series(np.nan, index=ctx.universe(), name=self.spec.name)
        returns = prices.pct_change(fill_method=None).iloc[1:]
        market = returns[ctx.benchmark]
        variance = market.var(ddof=1)
        if not np.isfinite(variance) or variance <= 0:
            return pd.Series(np.nan, index=ctx.universe(), name=self.spec.name)
        beta = returns.cov()[ctx.benchmark].div(variance)
        fitted = market.to_numpy()[:, None] * beta.to_numpy()[None, :]
        residual = returns.subtract(fitted)
        cumulative = residual.add(1).prod(min_count=200).sub(1).reindex(ctx.universe())
        industries = ctx.industry_map().reindex(cumulative.index).fillna("未分類")
        result = cumulative.sub(cumulative.groupby(industries).transform("mean"))
        return result.rename(self.spec.name)
