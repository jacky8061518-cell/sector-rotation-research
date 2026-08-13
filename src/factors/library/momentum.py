"""Price momentum factors."""

from __future__ import annotations

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
        result = (
            prices.iloc[-22].div(prices.iloc[-253]).sub(1)
            if len(prices) >= 253
            else pd.Series(dtype=float)
        )
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
        result = (
            prices.iloc[-1].div(prices.iloc[-127]).sub(1)
            if len(prices) >= 127
            else pd.Series(dtype=float)
        )
        return result.reindex(ctx.universe()).rename(self.spec.name)
