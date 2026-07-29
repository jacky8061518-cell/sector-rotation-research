"""Persistent market-data cache and daily research snapshot helpers."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .config import BENCHMARK, DEFENSIVE_ASSET
from .data import download_adjusted_prices
from .strategy import BacktestConfig, run_backtest
from .universe import UNIVERSE_GROUPS


FREQUENCY_LOOKBACKS = {
    "Daily": {5: 0.10, 21: 0.20, 63: 0.30, 126: 0.40},
    "Weekly": {4: 0.10, 13: 0.20, 26: 0.30, 52: 0.40},
    "Monthly": {1: 0.10, 3: 0.20, 6: 0.30, 12: 0.40},
}


def all_research_tickers() -> list[str]:
    tickers: list[str] = []
    for universe in UNIVERSE_GROUPS.values():
        for group in universe.values():
            tickers.extend(group)
    return list(dict.fromkeys([*tickers, BENCHMARK, DEFENSIVE_ASSET]))


def update_price_cache(
    cache_path: Path,
    tickers: list[str] | None = None,
    initial_start: date = date(2012, 1, 1),
) -> pd.DataFrame:
    """Incrementally update a local Parquet price cache."""
    tickers = tickers or all_research_tickers()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cached = pd.DataFrame()
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached.index = pd.to_datetime(cached.index)

    if cached.empty:
        combined = download_adjusted_prices(
            tickers,
            initial_start,
            date.today() + timedelta(days=1),
        )
    else:
        fetch_start = (cached.index.max() - pd.Timedelta(days=10)).date()
        fresh = download_adjusted_prices(
            tickers,
            fetch_start,
            date.today() + timedelta(days=1),
            min_observations=1,
        )
        missing_history = [
            ticker
            for ticker in tickers
            if ticker not in cached or cached[ticker].notna().sum() < 30
        ]
        history = pd.DataFrame()
        if missing_history:
            try:
                history = download_adjusted_prices(
                    missing_history,
                    initial_start,
                    date.today() + timedelta(days=1),
                )
            except RuntimeError:
                # Keep the recent observations. Newly launched tickers will
                # enter rankings after accumulating sufficient lookback data.
                history = pd.DataFrame()
        combined = pd.concat([cached, history, fresh]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.reindex(columns=tickers).ffill()
    combined.to_parquet(cache_path)
    return combined


def build_rotation_snapshots(
    prices: pd.DataFrame,
    output_dir: Path,
    top_n: int = 10,
    assets: list[str] | None = None,
    benchmark: str = BENCHMARK,
    defensive_asset: str = DEFENSIVE_ASSET,
) -> list[Path]:
    """Write latest daily, weekly, and monthly rankings as CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_assets = assets or all_research_tickers()
    assets = [
        ticker
        for ticker in requested_assets
        if ticker in prices and ticker not in {benchmark, defensive_asset}
    ]
    written: list[Path] = []

    for frequency, lookbacks in FREQUENCY_LOOKBACKS.items():
        result = run_backtest(
            prices,
            assets,
            BacktestConfig(
                lookback_weights=lookbacks,
                frequency=frequency,
                top_n=top_n,
                weighting="Equal weight",
                defensive_asset=defensive_asset,
            ),
        )
        latest = result.scores.iloc[-1].dropna().sort_values(ascending=False)
        ranking = pd.DataFrame(
            {
                "Ticker": latest.index,
                "Momentum score": latest.values,
                "Target weight": [
                    result.target_weights.iloc[-1].get(ticker, 0.0)
                    for ticker in latest.index
                ],
                "Signal date": result.scores.index[-1],
                "Frequency": frequency,
            }
        )
        dated_dir = output_dir / f"{result.scores.index[-1]:%Y-%m-%d}"
        dated_dir.mkdir(parents=True, exist_ok=True)
        dated_path = dated_dir / f"{frequency.lower()}-ranking.csv"
        latest_path = output_dir / f"latest-{frequency.lower()}-ranking.csv"
        ranking.to_csv(dated_path, index=False)
        ranking.to_csv(latest_path, index=False)
        written.extend([dated_path, latest_path])
    return written
