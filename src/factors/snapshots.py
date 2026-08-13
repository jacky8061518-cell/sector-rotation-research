"""Daily point-in-time factor data and ranking snapshots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .context import DataContext, FactorDataStore
from .pipeline import preprocess_factor
from .registry import get_factor, list_factors


def update_market_cap_snapshot(
    path: Path,
    adjusted_close: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:
    """Append today's observable market cap; never backfill latest shares historically."""
    asof = pd.Timestamp(adjusted_close.index.max()).normalize()
    latest = adjusted_close.loc[asof]
    metadata = master.drop_duplicates("Yahoo ticker").set_index("Yahoo ticker")
    shares = pd.to_numeric(metadata["Issued shares"], errors="coerce")
    fresh = pd.concat(
        [
            latest.mul(shares.reindex(latest.index)).rename("market_cap"),
            shares.reindex(latest.index).rename("shares_outstanding"),
        ],
        axis=1,
    ).dropna(subset=["market_cap"])
    fresh.insert(0, "ticker", fresh.index.astype(str))
    fresh.insert(0, "date", asof)
    fresh = fresh.reset_index(drop=True)
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([cached, fresh], ignore_index=True)
    combined = combined.drop_duplicates(["date", "ticker"], keep="last").sort_values(["date", "ticker"])
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    return combined


def build_daily_factor_snapshot(
    adjusted_close: pd.DataFrame,
    master: pd.DataFrame,
    store: FactorDataStore,
    output_path: Path,
    *,
    benchmark: str = "0050.TW",
) -> pd.DataFrame:
    asof = pd.Timestamp(adjusted_close.index.max())
    ctx = DataContext(store, "TW", asof, benchmark)
    columns = []
    for spec in list_factors(market="TW"):
        if not store.supports(spec.requires):
            continue
        factor = get_factor(spec.name)
        result = preprocess_factor(
            factor.compute(ctx, asof),
            spec,
            ctx.universe(),
            PipelineConfig(),
            industry=ctx.industry_map(),
        )
        if not result.values.empty:
            columns.append(result.values.rename(spec.name))
    scores = pd.concat(columns, axis=1) if columns else pd.DataFrame(index=ctx.universe())
    metadata = master.drop_duplicates("Yahoo ticker").set_index("Yahoo ticker")
    snapshot = metadata[["Name", "Market", "Industry"]].join(scores)
    factor_columns = [column for column in scores if column in snapshot]
    minimum_factors = max(1, int(np.ceil(len(factor_columns) * 0.60)))
    snapshot["composite_score"] = (
        snapshot[factor_columns].mean(axis=1).where(snapshot[factor_columns].notna().sum(axis=1).ge(minimum_factors))
    )
    snapshot.insert(0, "asof", asof)
    snapshot = snapshot.reset_index(names="ticker").sort_values("composite_score", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_parquet(output_path, index=False)
    snapshot.to_csv(output_path.with_suffix(".csv"), index=False)
    return snapshot
