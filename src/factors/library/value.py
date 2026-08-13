"""Point-in-time value factors; inactive until a publication-dated feed exists."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..context import DataContext
from ..registry import register_factor
from ..spec import FactorSpec
from ._fundamental import latest_field
from ._price import adjusted_close


def _latest_price(ctx: DataContext, asof: pd.Timestamp) -> pd.Series:
    prices = adjusted_close(ctx, asof, 1)
    return prices.iloc[-1] if not prices.empty else pd.Series(dtype=float)


@register_factor
class EarningsYield:
    spec = FactorSpec(
        "ep", "盈餘殖利率", "value", 1, 1, ("financials.eps",), ("US", "TW"), "最近已公告 EPS 除以訊號日價格。"
    )

    def compute(self, ctx: DataContext, asof: pd.Timestamp) -> pd.Series:
        return latest_field(ctx, "eps").div(_latest_price(ctx, asof)).reindex(ctx.universe()).rename(self.spec.name)


@register_factor
class BookToPrice:
    spec = FactorSpec(
        "bp",
        "淨值市價比",
        "value",
        1,
        1,
        ("financials.book_value_per_share",),
        ("US", "TW"),
        "最近已公告每股淨值除以訊號日價格。",
    )

    def compute(self, ctx: DataContext, asof: pd.Timestamp) -> pd.Series:
        return (
            latest_field(ctx, "book_value_per_share")
            .div(_latest_price(ctx, asof))
            .reindex(ctx.universe())
            .rename(self.spec.name)
        )


@register_factor
class FreeCashFlowYield:
    spec = FactorSpec(
        "fcf_yield",
        "自由現金流殖利率",
        "value",
        1,
        1,
        ("financials.free_cash_flow", "market_cap"),
        ("US", "TW"),
        "最近已公告自由現金流除以 point-in-time 市值。",
    )

    def compute(self, ctx: DataContext, asof: pd.Timestamp) -> pd.Series:
        cap = ctx.market_cap(1)
        market_cap = cap["market_cap"].groupby(level="ticker").last() if "market_cap" in cap else pd.Series(dtype=float)
        return (
            latest_field(ctx, "free_cash_flow")
            .div(market_cap.replace(0, np.nan))
            .reindex(ctx.universe())
            .rename(self.spec.name)
        )


@register_factor
class InverseEvEbitda:
    spec = FactorSpec(
        "ev_ebitda_inv",
        "EV/EBITDA 倒數",
        "value",
        1,
        1,
        ("financials.enterprise_value", "financials.ebitda"),
        ("US", "TW"),
        "最近已公告 EBITDA 除以企業價值。",
    )

    def compute(self, ctx: DataContext, asof: pd.Timestamp) -> pd.Series:
        return (
            latest_field(ctx, "ebitda")
            .div(latest_field(ctx, "enterprise_value").replace(0, np.nan))
            .reindex(ctx.universe())
            .rename(self.spec.name)
        )
