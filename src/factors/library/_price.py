"""Shared helpers for price-only factor implementations."""

from __future__ import annotations

import pandas as pd

from ..context import DataContext


def adjusted_close(
    ctx: DataContext,
    asof: pd.Timestamp,
    window: int,
) -> pd.DataFrame:
    if pd.Timestamp(asof).normalize() != ctx.asof.normalize():
        raise ValueError("Factor asof must match the bound DataContext asof.")
    panel = ctx.prices(("adj_close",), window=window)
    if panel.empty:
        return pd.DataFrame()
    return panel["adj_close"].unstack("ticker").sort_index()
