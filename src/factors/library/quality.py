"""Publication-aware quality factors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..context import DataContext
from ..registry import register_factor
from ..spec import FactorSpec
from ._fundamental import latest_field


@register_factor
class ReturnOnEquity:
    spec = FactorSpec(
        "roe", "股東權益報酬率", "quality", 1, 1, ("financials.roe",), ("US", "TW"), "最近已公告的股東權益報酬率。"
    )

    def compute(self, ctx: DataContext, asof: pd.Timestamp) -> pd.Series:
        return latest_field(ctx, "roe").reindex(ctx.universe()).rename(self.spec.name)


@register_factor
class GrossProfitability:
    spec = FactorSpec(
        "gross_profitability",
        "毛利資產比",
        "quality",
        1,
        1,
        ("financials.gross_profit", "financials.total_assets"),
        ("US", "TW"),
        "最近已公告毛利除以總資產。",
    )

    def compute(self, ctx: DataContext, asof: pd.Timestamp) -> pd.Series:
        return (
            latest_field(ctx, "gross_profit")
            .div(latest_field(ctx, "total_assets").replace(0, np.nan))
            .reindex(ctx.universe())
            .rename(self.spec.name)
        )


@register_factor
class Accruals:
    spec = FactorSpec(
        "accruals", "應計項目", "quality", -1, 1, ("financials.accruals",), ("US", "TW"), "較低應計項目代表較高盈餘品質。"
    )

    def compute(self, ctx: DataContext, asof: pd.Timestamp) -> pd.Series:
        return latest_field(ctx, "accruals").reindex(ctx.universe()).rename(self.spec.name)


@register_factor
class DebtToEquity:
    spec = FactorSpec(
        "debt_to_equity",
        "負債權益比",
        "quality",
        -1,
        1,
        ("financials.debt", "financials.equity"),
        ("US", "TW"),
        "較低負債權益比代表較低財務槓桿。",
    )

    def compute(self, ctx: DataContext, asof: pd.Timestamp) -> pd.Series:
        return (
            latest_field(ctx, "debt")
            .div(latest_field(ctx, "equity").replace(0, np.nan))
            .reindex(ctx.universe())
            .rename(self.spec.name)
        )
