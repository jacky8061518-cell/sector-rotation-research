"""Performance and drawdown calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    return initial * (1 + returns.fillna(0)).cumprod()


def drawdown(returns: pd.Series) -> pd.Series:
    equity = equity_curve(returns)
    return equity / equity.cummax() - 1


def performance_summary(returns: pd.Series) -> dict[str, float]:
    clean = returns.dropna()
    if clean.empty:
        return {key: np.nan for key in ("CAGR", "Volatility", "Sharpe", "Max drawdown", "Win rate")}

    years = len(clean) / 12
    ending = float((1 + clean).prod())
    cagr = ending ** (1 / years) - 1 if years > 0 and ending > 0 else np.nan
    volatility = float(clean.std(ddof=1) * np.sqrt(12))
    sharpe = float(clean.mean() / clean.std(ddof=1) * np.sqrt(12)) if clean.std(ddof=1) > 0 else np.nan
    return {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max drawdown": float(drawdown(clean).min()),
        "Win rate": float((clean > 0).mean()),
    }


def benchmark_returns(monthly_prices: pd.DataFrame, ticker: str, start: pd.Timestamp) -> pd.Series:
    if ticker not in monthly_prices:
        return pd.Series(dtype=float, name=ticker)
    return (
        monthly_prices[ticker]
        .pct_change(fill_method=None)
        .loc[start:]
        .fillna(0)
        .rename(ticker)
    )
