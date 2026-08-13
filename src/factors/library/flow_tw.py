"""Taiwan institutional-flow factors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..context import DataContext
from ..registry import register_factor
from ..spec import FactorSpec


@register_factor
class InstitutionalFlowTwentyDay:
    spec = FactorSpec(
        name="flow_inst_20d",
        label="法人 20 日資金流強度",
        category="flow",
        direction=1,
        lookback_days=20,
        requires=("inst_flow.total_net_value", "prices.traded_value"),
        markets=("TW",),
        description="三大法人二十日累計買超金額除以同期成交金額。",
    )

    def compute(self, ctx: DataContext, asof: pd.Timestamp) -> pd.Series:
        flow = ctx.inst_flow(20)
        try:
            traded = ctx.prices(("traded_value",), 20)
        except ValueError:
            traded = pd.DataFrame()
        if "total_net_value" not in flow or "traded_value" not in traded:
            return pd.Series(np.nan, index=ctx.universe(), name=self.spec.name)
        numerator = flow["total_net_value"].groupby(level="ticker").sum(min_count=1)
        denominator = traded["traded_value"].groupby(level="ticker").sum(min_count=1)
        return numerator.div(denominator.replace(0, np.nan)).reindex(ctx.universe()).rename(self.spec.name)


@register_factor
class ForeignFlowPersistence:
    spec = FactorSpec(
        name="flow_foreign_persist",
        label="外資連續買超",
        category="flow",
        direction=1,
        lookback_days=20,
        requires=("inst_flow.foreign_net_shares",),
        markets=("TW",),
        description="截至訊號日的外資連續買超交易日數，上限二十日。",
    )

    def compute(self, ctx: DataContext, asof: pd.Timestamp) -> pd.Series:
        flow = ctx.inst_flow(20)
        if "foreign_net_shares" not in flow:
            return pd.Series(np.nan, index=ctx.universe(), name=self.spec.name)

        def trailing_positive(values: pd.Series) -> int:
            positive = values.gt(0).to_numpy()[::-1]
            return int(np.argmax(~positive)) if (~positive).any() else len(positive)

        values = flow["foreign_net_shares"].groupby(level="ticker").apply(trailing_positive)
        return values.reindex(ctx.universe()).astype(float).rename(self.spec.name)
