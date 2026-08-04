"""Market-data adapters and deterministic demo data."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BENCHMARK, DEFENSIVE_ASSET, SECTOR_ETFS


def repair_taiwan_price_discontinuities(
    prices: pd.DataFrame,
    threshold: float = 0.40,
) -> pd.DataFrame:
    """Normalize unadjusted corporate-action jumps in Taiwan price histories.

    Yahoo's adjusted-price history occasionally omits a split, capital
    reduction, or other corporate-action adjustment. Taiwan-listed securities
    normally cannot move by more than the exchange price limit in one session,
    so an internal jump larger than ``threshold`` is a data discontinuity, not
    an investable return.

    For every such event, the history before the event is rebased by the
    observed price ratio. This preserves all ordinary day-to-day returns while
    making the corporate-action boundary continuous.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1.")

    repaired = prices.copy()
    for ticker in repaired.columns:
        if not isinstance(ticker, str) or not ticker.endswith((".TW", ".TWO")):
            continue

        series = repaired[ticker].dropna()
        if len(series) < 2:
            continue
        ratios = series.div(series.shift(1))
        events = ratios[(ratios < 1 - threshold) | (ratios > 1 + threshold)]

        for event_date, ratio in events.items():
            if pd.isna(ratio) or ratio <= 0:
                continue
            repaired.loc[repaired.index < event_date, ticker] *= float(ratio)

    return repaired


def download_adjusted_prices(
    tickers: list[str],
    start: date,
    end: date,
    min_observations: int = 30,
    *,
    auto_adjust: bool = True,
    batch_size: int = 100,
) -> pd.DataFrame:
    """Download daily closing prices from Yahoo Finance.

    Yahoo treats ``end`` as an exclusive boundary. Callers should therefore
    pass the day after the final date they want included. ``auto_adjust=True``
    returns dividend- and split-adjusted prices. ``False`` returns closing
    prices, after the Taiwan split-scale continuity guard is applied.
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

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    def fetch(requested: list[str]) -> pd.DataFrame:
        return yf.download(
            tickers=requested,
            start=start,
            end=end,
            auto_adjust=auto_adjust,
            progress=False,
            group_by="column",
            threads=min(16, len(requested)),
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

    batches = [
        tickers[offset : offset + batch_size]
        for offset in range(0, len(tickers), batch_size)
    ]
    downloaded: list[pd.DataFrame] = []
    for requested in batches:
        raw = fetch(requested)
        batch_prices = close_prices(raw, requested)

        # A batch request can occasionally fail even when individual tickers
        # are available. Retry its members one at a time.
        if batch_prices.empty:
            recovered: list[pd.DataFrame] = []
            for ticker in requested:
                single = close_prices(fetch([ticker]), [ticker])
                if not single.empty:
                    recovered.append(
                        single.rename(columns={single.columns[0]: ticker})
                    )
            if recovered:
                batch_prices = pd.concat(recovered, axis=1)
        if not batch_prices.empty:
            downloaded.append(batch_prices)

    prices = pd.concat(downloaded, axis=1) if downloaded else pd.DataFrame()

    if prices.empty:
        raise RuntimeError(
            "Yahoo Finance returned no data. Check the date range and connection, "
            "then retry; the end date must be later than the start date."
        )

    prices = prices.reindex(columns=tickers)
    prices = prices.dropna(how="all").ffill()
    if min_observations <= 0:
        raise ValueError("min_observations must be positive.")
    usable = prices.columns[prices.notna().sum() >= min_observations]
    prices = prices.loc[:, usable]
    if prices.empty:
        raise RuntimeError("No ticker has enough usable observations.")
    return repair_taiwan_price_discontinuities(prices)


def load_cached_or_download_prices(
    cache_path: Path,
    tickers: list[str],
    start: date,
    end: date,
    min_observations: int = 30,
    *,
    downloader: Callable[..., pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Load bundled prices first and download only uncovered tickers.

    Streamlit Community Cloud can be temporarily rate-limited by Yahoo.  A
    checked-in research database therefore remains the authoritative fallback:
    a failed supplemental request must not take the entire application down.
    ``end`` follows Yahoo's exclusive-boundary convention.
    """
    if end <= start:
        raise ValueError("The end date must be later than the start date.")
    if min_observations <= 0:
        raise ValueError("min_observations must be positive.")

    requested = list(dict.fromkeys(tickers))
    cached = pd.DataFrame()
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached.index = pd.to_datetime(cached.index)
        cached = cached.loc[
            (cached.index >= pd.Timestamp(start))
            & (cached.index < pd.Timestamp(end))
        ]
        cached = cached.reindex(columns=requested)

    missing = [
        ticker
        for ticker in requested
        if ticker not in cached or cached[ticker].notna().sum() < min_observations
    ]
    fresh = pd.DataFrame()
    download = downloader or download_adjusted_prices
    if missing:
        try:
            fresh = download(
                missing,
                start,
                end,
                min_observations=min_observations,
                batch_size=25,
            )
        except Exception:
            # The bundled database is intentionally allowed to carry the app
            # through a transient vendor outage or rate limit.
            fresh = pd.DataFrame()

    prices = pd.concat([cached, fresh], axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated(keep="last")]
    prices = prices.reindex(columns=requested).dropna(how="all").ffill()
    usable = prices.columns[prices.notna().sum() >= min_observations]
    prices = prices.loc[:, usable]
    if prices.empty:
        raise RuntimeError(
            "No ticker has enough usable observations in the bundled database "
            "or the live data source."
        )
    return repair_taiwan_price_discontinuities(prices)


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
