"""ETF top-holdings retrieval and weighted leadership analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fetch_top_holdings(etf_tickers: list[str]) -> pd.DataFrame:
    """Fetch the reported top holdings for each ETF through yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("yfinance is not installed.") from exc

    frames: list[pd.DataFrame] = []
    for etf in dict.fromkeys(etf_tickers):
        try:
            holdings = yf.Ticker(etf).funds_data.top_holdings
        except Exception:
            continue
        if holdings is None or holdings.empty:
            continue

        normalized = holdings.reset_index().rename(
            columns={
                holdings.index.name or "index": "Holding ticker",
                "Name": "Holding name",
                "Holding Percent": "Holding weight",
            }
        )
        required = {"Holding ticker", "Holding name", "Holding weight"}
        if not required.issubset(normalized.columns):
            continue
        normalized = normalized[list(required)].copy()
        normalized["ETF"] = etf
        normalized["Holding ticker"] = normalized["Holding ticker"].astype(str).str.upper()
        normalized["Holding weight"] = pd.to_numeric(
            normalized["Holding weight"],
            errors="coerce",
        )
        frames.append(normalized.dropna(subset=["Holding ticker", "Holding weight"]))

    if not frames:
        return pd.DataFrame(
            columns=["ETF", "Holding ticker", "Holding name", "Holding weight"]
        )
    result = pd.concat(frames, ignore_index=True)
    return result[["ETF", "Holding ticker", "Holding name", "Holding weight"]]


def analyze_holding_leadership(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    lookback_weights: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Rank ETF holdings using weight and multi-horizon stock momentum.

    ``Leadership score`` is a research priority proxy, not an estimate of exact
    ETF return attribution. It multiplies a holding's reported ETF weight by a
    normalized composite of its own recent price returns.
    """
    if holdings.empty:
        return holdings.copy()
    lookback_weights = lookback_weights or {21: 0.2, 63: 0.3, 126: 0.5}
    total_weight = sum(lookback_weights.values())
    if total_weight <= 0:
        raise ValueError("Holding lookback weights must sum to a positive value.")

    clean_prices = prices.sort_index()
    rows: list[dict[str, object]] = []
    for record in holdings.to_dict("records"):
        ticker = str(record["Holding ticker"])
        if ticker not in clean_prices:
            continue
        series = clean_prices[ticker].dropna()
        if len(series) <= max(lookback_weights):
            continue

        composite = 0.0
        row = dict(record)
        complete = True
        for periods, weight in lookback_weights.items():
            value = float(series.iloc[-1] / series.iloc[-periods - 1] - 1)
            if not np.isfinite(value):
                complete = False
                break
            row[f"{periods}d return"] = value
            composite += value * weight / total_weight
        if not complete:
            continue

        row["Stock momentum"] = composite
        row["Leadership score"] = float(row["Holding weight"]) * composite
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    result["ETF rank"] = (
        result.groupby("ETF")["Leadership score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return result.sort_values(["ETF", "ETF rank"]).reset_index(drop=True)
