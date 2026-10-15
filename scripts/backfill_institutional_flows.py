"""Backfill official TWSE/TPEx daily institutional flows with bounded concurrency."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd

from sector_rotation.fund_flow import (
    FLOW_COLUMNS,
    fetch_tpex_institutional_flows,
    fetch_twse_institutional_flows,
)


def _fetch_market(session: date, market: str) -> pd.DataFrame:
    fetcher = fetch_twse_institutional_flows if market == "TWSE" else fetch_tpex_institutional_flows
    for attempt in range(3):
        try:
            frame = fetcher(session)
        except (OSError, TimeoutError, ValueError):
            frame = pd.DataFrame(columns=FLOW_COLUMNS)
        if not frame.empty:
            return frame
        time.sleep(0.75 * (attempt + 1))
    return pd.DataFrame(columns=FLOW_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    cache = pd.read_parquet(args.cache) if args.cache.exists() else pd.DataFrame(columns=FLOW_COLUMNS)
    if not cache.empty:
        cache["Date"] = pd.to_datetime(cache["Date"]).dt.normalize()
    market_names = {"TWSE": "上市", "TPEx": "上櫃"}
    existing = (
        set(zip(cache["Date"], cache["Market"], strict=True)) if not cache.empty else set()
    )
    prices = pd.read_parquet(args.prices)
    sessions = pd.DatetimeIndex(prices.index).normalize()
    sessions = sessions[(sessions >= pd.Timestamp(args.start)) & (sessions <= pd.Timestamp(args.end))]
    jobs = [
        (timestamp.date(), market)
        for timestamp in sessions
        for market in ("TWSE", "TPEx")
        if (timestamp, market_names[market]) not in existing
    ]
    print(f"Fetching {len(jobs)} missing market sessions")

    fresh: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(_fetch_market, session, market): (session, market) for session, market in jobs}
        for count, future in enumerate(as_completed(futures), 1):
            frame = future.result()
            if not frame.empty:
                fresh.append(frame)
            if count % 100 == 0 or count == len(jobs):
                print(f"Completed {count}/{len(jobs)} requests")

    combined = pd.concat([cache, *fresh], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"]).dt.normalize()
    combined = combined.drop_duplicates(["Date", "Ticker"], keep="last").sort_values(["Date", "Ticker"])
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.cache, index=False)
    print(
        f"Saved {len(combined):,} rows, {combined['Date'].nunique():,} sessions, "
        f"{combined['Date'].min():%Y-%m-%d} through {combined['Date'].max():%Y-%m-%d}"
    )


if __name__ == "__main__":
    main()
