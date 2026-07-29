"""Refresh market data and save daily/weekly/monthly ranking snapshots."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sector_rotation.config import BENCHMARK, DEFENSIVE_ASSET
from sector_rotation.snapshots import (
    all_research_tickers,
    build_rotation_snapshots,
    update_price_cache,
)
from sector_rotation.taiwan import (
    TW_BENCHMARK,
    TW_DEFENSIVE_ASSET,
    all_taiwan_research_assets,
    fetch_taiwan_company_master,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_DIR = PROJECT_ROOT / "data" / "databases"
SNAPSHOT_ROOT = PROJECT_ROOT / "data" / "snapshots"


def main() -> None:
    us_tickers = all_research_tickers()
    us_prices = update_price_cache(
        DATABASE_DIR / "us" / "adjusted-prices.parquet",
        us_tickers,
    )
    us_written = build_rotation_snapshots(
        us_prices,
        SNAPSHOT_ROOT / "us",
        assets=us_tickers,
        benchmark=BENCHMARK,
        defensive_asset=DEFENSIVE_ASSET,
    )

    taiwan_master = fetch_taiwan_company_master()
    taiwan_assets = all_taiwan_research_assets(taiwan_master)
    taiwan_tickers = list(
        dict.fromkeys([*taiwan_assets, TW_BENCHMARK, TW_DEFENSIVE_ASSET])
    )
    tw_prices = update_price_cache(
        DATABASE_DIR / "tw" / "adjusted-prices.parquet",
        taiwan_tickers,
    )
    tw_written = build_rotation_snapshots(
        tw_prices,
        SNAPSHOT_ROOT / "tw",
        assets=list(taiwan_assets),
        benchmark=TW_BENCHMARK,
        defensive_asset=TW_DEFENSIVE_ASSET,
    )

    print(
        f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
        f"US: {len(us_prices.columns)} tickers through {us_prices.index.max():%Y-%m-%d}, "
        f"{len(us_written)} files; "
        f"TW: {len(tw_prices.columns)} tickers through {tw_prices.index.max():%Y-%m-%d}, "
        f"{len(tw_written)} files"
    )


if __name__ == "__main__":
    main()
