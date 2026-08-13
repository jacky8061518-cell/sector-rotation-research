"""Publication-aware Taiwan monthly-revenue factors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..context import DataContext
from ..registry import register_factor
from ..spec import FactorSpec


def _latest_revenue(ctx: DataContext, window: int) -> pd.DataFrame:
    revenue = ctx.revenue(window)
    if revenue.empty:
        return revenue
    return revenue.sort_index().groupby(level="ticker").tail(window)


@register_factor
class RevenueYearOverYear:
    spec = FactorSpec(
        name="rev_yoy",
        label="月營收年增率",
        category="growth",
        direction=1,
        lookback_days=1,
        requires=("revenue.revenue_yoy", "revenue.published_at"),
        markets=("TW",),
        description="最近一筆已公告月營收的年增率，以實際擷取／公布時間控管可見性。",
    )

    def compute(self, ctx: DataContext, asof: pd.Timestamp) -> pd.Series:
        revenue = _latest_revenue(ctx, 1)
        values = (
            revenue["revenue_yoy"].groupby(level="ticker").last() if "revenue_yoy" in revenue else pd.Series(dtype=float)
        )
        return values.reindex(ctx.universe()).rename(self.spec.name)


@register_factor
class RevenueMomentumThreeMonth:
    spec = FactorSpec(
        name="rev_mom_3m",
        label="三個月營收動能",
        category="growth",
        direction=1,
        lookback_days=3,
        requires=("revenue.revenue", "revenue.published_at"),
        markets=("TW",),
        description="最近三個已公告月份營收相對去年同期合計的成長率。",
    )

    def compute(self, ctx: DataContext, asof: pd.Timestamp) -> pd.Series:
        revenue = _latest_revenue(ctx, 15)
        required = {"revenue", "period"}
        if not required.issubset(revenue.columns):
            return pd.Series(np.nan, index=ctx.universe(), name=self.spec.name)
        frame = revenue.reset_index().sort_values(["ticker", "period"])
        frame["period"] = pd.PeriodIndex(frame["period"], freq="M")
        lookup = frame.set_index(["ticker", "period"])["revenue"]
        results: dict[str, float] = {}
        for ticker, group in frame.groupby("ticker"):
            latest = group.tail(3)
            prior_periods = latest["period"].map(lambda period: period - 12)
            prior = lookup.reindex(pd.MultiIndex.from_arrays([[ticker] * len(prior_periods), prior_periods])).to_numpy()
            current = latest["revenue"].to_numpy(dtype=float)
            denominator = float(np.nansum(prior))
            results[str(ticker)] = (
                float(np.nansum(current) / denominator - 1) if denominator > 0 and np.isfinite(prior).all() else np.nan
            )
        return pd.Series(results, name=self.spec.name).reindex(ctx.universe())
