"""Market-data adapters and deterministic demo data."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .config import BENCHMARK, DEFENSIVE_ASSET, SECTOR_ETFS


def download_adjusted_prices(
    tickers: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Download adjusted daily closing prices from Yahoo Finance.

    Yahoo treats ``end`` as an exclusive boundary. Callers should therefore
    pass the day after the final date they want included.
    """
    if end <= start:
        raise ValueError(
            "The end date must be later than the start date. "
            "Choose a start date with at least 13 months of history."
        )
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised by the UI
        raise RuntimeError("yfinance is not installed. Run: pip install -r requirements.txt") from exc

    def fetch(requested: list[str]) -> pd.DataFrame:
        return yf.download(
            tickers=requested,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=False,
            timeout=20,
        )

    def close_prices(raw: pd.DataFrame, requested: list[str]) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" not in raw.columns.get_level_values(0):
                return pd.DataFrame()
            return raw["Close"].copy()
        if "Close" not in raw:
            return pd.DataFrame()
        result = raw[["Close"]].copy()
        result.columns = [requested[0]]
        return result

    raw = fetch(tickers)
    prices = close_prices(raw, tickers)

    # A batch request can occasionally fail even when individual tickers are
    # available. Retry one ticker at a time and retain every successful series.
    if prices.empty:
        recovered: list[pd.DataFrame] = []
        for ticker in tickers:
            single = close_prices(fetch([ticker]), [ticker])
            if not single.empty:
                recovered.append(single.rename(columns={single.columns[0]: ticker}))
        if recovered:
            prices = pd.concat(recovered, axis=1)

    if prices.empty:
        raise RuntimeError(
            "Yahoo Finance returned no data. Check the date range and connection, "
            "then retry; the end date must be later than the start date."
        )

    prices = prices.reindex(columns=tickers)
    prices = prices.dropna(how="all").ffill()
    usable = prices.columns[prices.notna().sum() >= 30]
    prices = prices.loc[:, usable]
    if prices.empty:
        raise RuntimeError("No ticker has enough usable observations.")
    return prices


def generate_demo_prices(
    start: str = "2012-01-01",
    end: str | None = None,
    seed: int = 42,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Generate reproducible correlated prices for offline demonstrations.

    The series intentionally contain changing leadership regimes. They are
    synthetic and must never be interpreted as historical investment results.
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    index = pd.bdate_range(start, end)
    tickers = list(dict.fromkeys(tickers or [*SECTOR_ETFS, BENCHMARK, DEFENSIVE_ASSET]))
    research_tickers = [
        ticker for ticker in tickers if ticker not in {BENCHMARK, DEFENSIVE_ASSET}
    ]
    rng = np.random.default_rng(seed)

    market = rng.normal(0.00028, 0.009, len(index))
    returns = pd.DataFrame(index=index, columns=tickers, dtype=float)
    regime_length = 252

    asset_count = max(1, len(research_tickers))
    for number, ticker in enumerate(research_tickers):
        idiosyncratic = rng.normal(0, 0.0065 + (number % 3) * 0.0007, len(index))
        regime_alpha = np.zeros(len(index))
        for regime_start in range(0, len(index), regime_length):
            leader = (regime_start // regime_length) % asset_count
            distance = min(
                (number - leader) % asset_count,
                (leader - number) % asset_count,
            )
            alpha = 0.00045 if distance == 0 else (0.00018 if distance == 1 else -0.00005)
            regime_alpha[regime_start : regime_start + regime_length] = alpha
        returns[ticker] = 0.78 * market + idiosyncratic + regime_alpha

    if BENCHMARK in tickers:
        returns[BENCHMARK] = market + rng.normal(0, 0.002, len(index))
    if DEFENSIVE_ASSET in tickers:
        returns[DEFENSIVE_ASSET] = rng.normal(0.00008, 0.0009, len(index))
    prices = 100 * (1 + returns.clip(lower=-0.20)).cumprod()
    return prices
