"""Pure portfolio-weight construction for factor research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

WeightMethod = Literal["equal", "score", "inverse_volatility"]


@dataclass(frozen=True)
class PortfolioConfig:
    quantiles: int = 5
    long_quantile: int = 5
    short_quantile: int = 1
    long_short: bool = True
    weight_method: WeightMethod = "equal"
    maximum_stock_weight: float = 0.10
    minimum_holdings: int = 10
    maximum_industry_weight: float = 0.35

    def __post_init__(self) -> None:
        if self.quantiles < 2:
            raise ValueError("At least two quantiles are required.")
        if not 1 <= self.long_quantile <= self.quantiles:
            raise ValueError("long_quantile is outside the configured range.")
        if not 1 <= self.short_quantile <= self.quantiles:
            raise ValueError("short_quantile is outside the configured range.")
        if not 0 < self.maximum_stock_weight <= 1:
            raise ValueError("maximum_stock_weight must be within (0, 1].")
        if self.minimum_holdings < 1:
            raise ValueError("minimum_holdings must be positive.")
        if not 0 < self.maximum_industry_weight <= 1:
            raise ValueError("maximum_industry_weight must be within (0, 1].")


DEFAULT_PORTFOLIO_CONFIG = PortfolioConfig()


def assign_quantiles(scores: pd.Series, quantiles: int = 5) -> pd.Series:
    valid = pd.to_numeric(scores, errors="coerce").dropna()
    if len(valid) < quantiles:
        return pd.Series(dtype="Int64", name="quantile")
    ranks = valid.rank(method="first")
    result = pd.qcut(ranks, quantiles, labels=False).astype(int).add(1)
    return result.astype("Int64").rename("quantile")


def _side_weights(
    selected_scores: pd.Series,
    method: WeightMethod,
    volatility: pd.Series | None,
) -> pd.Series:
    if method == "equal":
        raw = pd.Series(1.0, index=selected_scores.index)
    elif method == "score":
        raw = selected_scores.sub(selected_scores.min()).add(1e-12)
    else:
        if volatility is None:
            raise ValueError("inverse_volatility weights require volatility data.")
        raw = volatility.reindex(selected_scores.index).replace(0, np.nan).pow(-1)
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
    return raw.div(raw.sum()) if not raw.empty and raw.sum() > 0 else raw


def _apply_caps(
    weights: pd.Series,
    stock_cap: float,
    industry: pd.Series | None,
    industry_cap: float,
) -> pd.Series:
    """Redistribute toward one unit of gross exposure without violating caps."""
    result = weights.copy()
    labels = (
        industry.reindex(result.index).fillna("未分類")
        if industry is not None
        else pd.Series(result.index, index=result.index)
    )
    for _ in range(50):
        result = result.clip(upper=stock_cap)
        totals = result.groupby(labels).transform("sum")
        result = result.mul((industry_cap / totals).clip(upper=1.0))
        deficit = 1.0 - float(result.sum())
        if deficit <= 1e-10:
            break
        group_totals = result.groupby(labels).transform("sum")
        stock_room = stock_cap - result
        group_room = (industry_cap - group_totals).clip(lower=0)
        room = pd.concat([stock_room, group_room], axis=1).min(axis=1).clip(lower=0)
        total_room = float(room.sum())
        if total_room <= 1e-12:
            break
        result = result.add(room.mul(min(1.0, deficit / total_room)))
    return result


def build_weights(
    scores: pd.Series,
    config: PortfolioConfig = DEFAULT_PORTFOLIO_CONFIG,
    *,
    industry: pd.Series | None = None,
    volatility: pd.Series | None = None,
) -> pd.Series:
    """Construct constrained long-only or dollar-neutral long/short weights."""
    groups = assign_quantiles(scores, config.quantiles)
    long_scores = scores.reindex(groups[groups.eq(config.long_quantile)].index).dropna()
    short_scores = scores.reindex(groups[groups.eq(config.short_quantile)].index).dropna()
    if len(long_scores) < config.minimum_holdings:
        return pd.Series(dtype=float, name="weight")
    sides = [(long_scores, 1.0)]
    if config.long_short:
        if len(short_scores) < config.minimum_holdings:
            return pd.Series(dtype=float, name="weight")
        sides.append((-short_scores, -1.0))
    weights = []
    for side_scores, sign in sides:
        side = _side_weights(side_scores, config.weight_method, volatility)
        side = _apply_caps(
            side,
            config.maximum_stock_weight,
            industry,
            config.maximum_industry_weight,
        )
        weights.append(side.mul(sign))
    result = pd.concat(weights).groupby(level=0).sum().sort_index()
    return result.rename("weight")
