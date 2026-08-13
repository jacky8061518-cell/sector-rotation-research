"""Point-in-time factor portfolio backtest with next-session execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import DEFAULT_COST_CONFIG, CostConfig, transaction_cost
from .portfolio import DEFAULT_PORTFOLIO_CONFIG, PortfolioConfig, build_weights
from .spec import Market


@dataclass(frozen=True)
class BacktestResult:
    returns: pd.DataFrame
    equity: pd.DataFrame
    holdings: pd.DataFrame
    costs: pd.DataFrame
    metrics: pd.DataFrame


def performance_metrics(returns: pd.Series) -> dict[str, float]:
    valid = returns.dropna()
    if valid.empty:
        return {
            name: float("nan")
            for name in ("annual_return", "annual_volatility", "sharpe", "max_drawdown", "monthly_win_rate")
        }
    equity = valid.add(1).cumprod()
    years = max(len(valid) / 252, 1 / 252)
    annual_return = float(equity.iloc[-1] ** (1 / years) - 1)
    annual_volatility = float(valid.std(ddof=1) * np.sqrt(252))
    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": annual_return / annual_volatility if annual_volatility > 0 else float("nan"),
        "max_drawdown": float(equity.div(equity.cummax()).sub(1).min()),
        "monthly_win_rate": float(
            valid.groupby(valid.index.to_period("M")).apply(lambda values: values.add(1).prod() - 1).gt(0).mean()
        ),
    }


def run_factor_backtest(
    scores: pd.DataFrame,
    adjusted_close: pd.DataFrame,
    market: Market,
    portfolio: PortfolioConfig = DEFAULT_PORTFOLIO_CONFIG,
    costs: CostConfig = DEFAULT_COST_CONFIG,
    *,
    industry: pd.Series | None = None,
) -> BacktestResult:
    """Trade each signal at the next available close; never on the signal date."""
    prices = adjusted_close.sort_index()
    prices.index.name = "date"
    daily_returns = prices.pct_change(fill_method=None)
    signal_dates = pd.DatetimeIndex(scores.index).intersection(prices.index)
    execution_map: dict[pd.Timestamp, pd.Timestamp] = {}
    for signal_date in signal_dates:
        position = prices.index.searchsorted(signal_date, side="right")
        if position < len(prices.index):
            execution_map[signal_date] = pd.Timestamp(prices.index[position])
    targets: dict[pd.Timestamp, pd.Series] = {}
    for signal_date, execution_date in execution_map.items():
        trailing_vol = daily_returns.loc[:signal_date].tail(60).std(ddof=1)
        target = build_weights(scores.loc[signal_date], portfolio, industry=industry, volatility=trailing_vol)
        if not target.empty:
            targets[execution_date] = target

    gross = pd.Series(0.0, index=prices.index, name="gross")
    cost_series = pd.Series(0.0, index=prices.index, name="cost")
    previous = pd.Series(dtype=float)
    holding_rows: list[pd.DataFrame] = []
    cost_rows: list[dict[str, float | pd.Timestamp]] = []
    active = pd.Series(dtype=float)
    for current_date in prices.index:
        # A target executed at today's close only earns returns after that close.
        if not active.empty:
            gross.loc[current_date] = float(daily_returns.loc[current_date].reindex(active.index).fillna(0).dot(active))
        if current_date in targets:
            target = targets[current_date]
            cost, buy_turnover, sell_turnover = transaction_cost(previous, target, market, costs)
            cost_series.loc[current_date] = cost
            active = target
            previous = target
            holding_rows.append(target.rename("weight").to_frame().assign(date=current_date).reset_index(names="ticker"))
            cost_rows.append(
                {"date": current_date, "buy_turnover": buy_turnover, "sell_turnover": sell_turnover, "cost": cost}
            )
    returns = pd.concat([gross, gross.sub(cost_series).rename("net")], axis=1)
    equity = returns.add(1).cumprod()
    holdings = (
        pd.concat(holding_rows, ignore_index=True) if holding_rows else pd.DataFrame(columns=["ticker", "weight", "date"])
    )
    cost_frame = pd.DataFrame(cost_rows)
    metrics = pd.DataFrame({column: performance_metrics(returns[column]) for column in returns}).T
    return BacktestResult(returns, equity, holdings, cost_frame, metrics)
