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
    """Download adjusted daily closing prices from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised by the UI
        raise RuntimeError("yfinance is not installed. Run: pip install -r requirements.txt") from exc

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no data for the selected period.")

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = [tickers[0]]

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
) -> pd.DataFrame:
    """Generate reproducible correlated prices for offline demonstrations.

    The series intentionally contain changing leadership regimes. They are
    synthetic and must never be interpreted as historical investment results.
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    index = pd.bdate_range(start, end)
    tickers = [*SECTOR_ETFS, BENCHMARK, DEFENSIVE_ASSET]
    rng = np.random.default_rng(seed)

    market = rng.normal(0.00028, 0.009, len(index))
    returns = pd.DataFrame(index=index, columns=tickers, dtype=float)
    regime_length = 252

    for number, ticker in enumerate(SECTOR_ETFS):
        idiosyncratic = rng.normal(0, 0.0065 + (number % 3) * 0.0007, len(index))
        regime_alpha = np.zeros(len(index))
        for regime_start in range(0, len(index), regime_length):
            leader = (regime_start // regime_length) % len(SECTOR_ETFS)
            distance = min(
                (number - leader) % len(SECTOR_ETFS),
                (leader - number) % len(SECTOR_ETFS),
            )
            alpha = 0.00045 if distance == 0 else (0.00018 if distance == 1 else -0.00005)
            regime_alpha[regime_start : regime_start + regime_length] = alpha
        returns[ticker] = 0.78 * market + idiosyncratic + regime_alpha

    returns[BENCHMARK] = market + rng.normal(0, 0.002, len(index))
    returns[DEFENSIVE_ASSET] = rng.normal(0.00008, 0.0009, len(index))
    prices = 100 * (1 + returns.clip(lower=-0.20)).cumprod()
    return prices
