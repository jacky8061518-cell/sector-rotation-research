"""Refresh market data and save daily/weekly/monthly ranking snapshots."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sector_rotation.snapshots import build_rotation_snapshots, update_price_cache


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "adjusted-prices.parquet"
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots"


def main() -> None:
    prices = update_price_cache(CACHE_PATH)
    written = build_rotation_snapshots(prices, SNAPSHOT_DIR)
    print(
        f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
        f"updated {len(prices.columns)} tickers through {prices.index.max():%Y-%m-%d}; "
        f"wrote {len(written)} snapshot files"
    )


if __name__ == "__main__":
    main()
