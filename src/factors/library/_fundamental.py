"""Helpers for publication-aware fundamental factors."""

from __future__ import annotations

import pandas as pd

from ..context import DataContext


def latest_field(ctx: DataContext, field: str) -> pd.Series:
    frame = ctx.financials()
    if field not in frame:
        return pd.Series(dtype=float)
    return frame[field].groupby(level="ticker").last()
