"""Publication-aware fundamental growth factors."""

from __future__ import annotations

import pandas as pd

from ..context import DataContext
from ..registry import register_factor
from ..spec import FactorSpec
from ._fundamental import latest_field


@register_factor
class EarningsGrowthYearOverYear:
    spec = FactorSpec(
        "eps_growth_yoy",
        "EPS 年增率",
        "growth",
        1,
        1,
        ("financials.eps_growth_yoy",),
        ("US", "TW"),
        "最近已公告 EPS 相對去年同期成長率。",
    )

    def compute(self, ctx: DataContext, asof: pd.Timestamp) -> pd.Series:
        return latest_field(ctx, "eps_growth_yoy").reindex(ctx.universe()).rename(self.spec.name)


@register_factor
class SalesGrowthYearOverYear:
    spec = FactorSpec(
        "sales_growth_yoy",
        "營收年增率（財報）",
        "growth",
        1,
        1,
        ("financials.sales_growth_yoy",),
        ("US", "TW"),
        "最近已公告季度營收相對去年同期成長率。",
    )

    def compute(self, ctx: DataContext, asof: pd.Timestamp) -> pd.Series:
        return latest_field(ctx, "sales_growth_yoy").reindex(ctx.universe()).rename(self.spec.name)
