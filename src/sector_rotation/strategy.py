"""Momentum ranking, portfolio construction, and bias-aware backtesting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    """Parameters required to run a sector-rotation backtest."""

    lookback_weights: dict[int, float]
    top_n: int = 3
    weighting: str = "Equal weight"
    require_positive_momentum: bool = True
    risk_adjusted_score: bool = False
    defensive_asset: str | None = "SHY"
    transaction_cost_bps: float = 5.0
    volatility_window: int = 6

    def __post_init__(self) -> None:
        if not self.lookback_weights:
            raise ValueError("At least one lookback period is required.")
        if any(month <= 0 for month in self.lookback_weights):
            raise ValueError("Lookback periods must be positive.")
        if sum(self.lookback_weights.values()) <= 0:
            raise ValueError("Lookback weights must have a positive sum.")
        if self.top_n <= 0:
            raise ValueError("top_n must be positive.")
        if self.weighting not in {"Equal weight", "Momentum weight", "Inverse volatility"}:
            raise ValueError(f"Unknown weighting method: {self.weighting}")
        if self.transaction_cost_bps < 0:
            raise ValueError("Transaction costs cannot be negative.")


@dataclass
class BacktestResult:
    monthly_prices: pd.DataFrame
    scores: pd.DataFrame
    target_weights: pd.DataFrame
    deployed_weights: pd.DataFrame
    gross_returns: pd.Series
    net_returns: pd.Series
    turnover: pd.Series


def to_monthly_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert daily prices to one observation at each calendar month end."""
    if prices.empty:
        raise ValueError("Price data is empty.")
    data = prices.sort_index().copy()
    data.index = pd.to_datetime(data.index).tz_localize(None)
    monthly = data.groupby(data.index.to_period("M")).last()
    monthly.index = monthly.index.to_timestamp("M")
    return monthly.dropna(how="all")


def compute_momentum_scores(
    monthly_prices: pd.DataFrame,
    assets: list[str],
    lookback_weights: dict[int, float],
    risk_adjusted: bool = False,
    volatility_window: int = 6,
) -> pd.DataFrame:
    """Compute a normalized composite momentum score."""
    available = [asset for asset in assets if asset in monthly_prices.columns]
    if not available:
        raise ValueError("None of the selected assets is present in the price data.")

    normalized = {
        month: weight / sum(lookback_weights.values())
        for month, weight in lookback_weights.items()
    }
    score = pd.DataFrame(0.0, index=monthly_prices.index, columns=available)
    ready = pd.DataFrame(True, index=monthly_prices.index, columns=available)

    for months, weight in normalized.items():
        component = monthly_prices[available].pct_change(months, fill_method=None)
        score = score.add(component * weight, fill_value=0)
        ready &= component.notna()

    score = score.where(ready)
    if risk_adjusted:
        monthly_returns = monthly_prices[available].pct_change(fill_method=None)
        volatility = monthly_returns.rolling(volatility_window).std() * np.sqrt(12)
        score = score.div(volatility.replace(0, np.nan))
    return score


def build_target_weights(
    scores: pd.DataFrame,
    monthly_prices: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Translate each month-end ranking into target portfolio weights."""
    columns = list(scores.columns)
    if config.defensive_asset and config.defensive_asset in monthly_prices.columns:
        columns.append(config.defensive_asset)
    weights = pd.DataFrame(0.0, index=scores.index, columns=columns)
    monthly_returns = monthly_prices[scores.columns].pct_change(fill_method=None)
    volatility = monthly_returns.rolling(config.volatility_window).std() * np.sqrt(12)

    for timestamp, row in scores.iterrows():
        candidates = row.dropna()
        if config.require_positive_momentum:
            candidates = candidates[candidates > 0]
        selected = candidates.nlargest(min(config.top_n, len(candidates)))

        if selected.empty:
            if config.defensive_asset and config.defensive_asset in weights.columns:
                weights.loc[timestamp, config.defensive_asset] = 1.0
            continue

        if config.weighting == "Equal weight":
            allocation = pd.Series(1 / len(selected), index=selected.index)
        elif config.weighting == "Momentum weight":
            strength = selected.clip(lower=0)
            allocation = strength / strength.sum()
        else:
            selected_volatility = volatility.loc[timestamp, selected.index]
            inverse = 1 / selected_volatility.replace(0, np.nan)
            inverse = inverse.replace([np.inf, -np.inf], np.nan).dropna()
            allocation = (
                inverse / inverse.sum()
                if not inverse.empty
                else pd.Series(1 / len(selected), index=selected.index)
            )

        weights.loc[timestamp, allocation.index] = allocation
    return weights


def run_backtest(
    prices: pd.DataFrame,
    assets: list[str],
    config: BacktestConfig,
) -> BacktestResult:
    """Run a monthly rotation strategy without look-ahead bias.

    Scores and target weights observed at month-end t are shifted forward and
    earn the return from t to t+1. Transaction costs are charged when the
    deployed portfolio changes.
    """
    monthly_prices = to_monthly_prices(prices)
    scores = compute_momentum_scores(
        monthly_prices,
        assets,
        config.lookback_weights,
        risk_adjusted=config.risk_adjusted_score,
        volatility_window=config.volatility_window,
    )
    target = build_target_weights(scores, monthly_prices, config)
    deployed = target.shift(1).fillna(0)
    asset_returns = monthly_prices.reindex(columns=deployed.columns).pct_change(fill_method=None)
    gross = (deployed * asset_returns).sum(axis=1, min_count=1).fillna(0)
    turnover = deployed.diff().abs().sum(axis=1).div(2).fillna(0)
    costs = turnover * config.transaction_cost_bps / 10_000
    net = gross - costs

    first_valid_signal = scores.notna().any(axis=1)
    if first_valid_signal.any():
        first_signal_position = int(np.flatnonzero(first_valid_signal.to_numpy())[0])
        start_position = min(first_signal_position + 1, len(scores) - 1)
        start = scores.index[start_position]
        scores = scores.loc[start:]
        target = target.loc[start:]
        deployed = deployed.loc[start:]
        gross = gross.loc[start:]
        net = net.loc[start:]
        turnover = turnover.loc[start:]

    return BacktestResult(
        monthly_prices=monthly_prices,
        scores=scores,
        target_weights=target,
        deployed_weights=deployed,
        gross_returns=gross.rename("Gross strategy"),
        net_returns=net.rename("Net strategy"),
        turnover=turnover.rename("Turnover"),
    )
