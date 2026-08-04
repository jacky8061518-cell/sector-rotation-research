"""Refresh market data and save daily/weekly/monthly ranking snapshots."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sector_rotation.config import BENCHMARK, DEFENSIVE_ASSET
from sector_rotation.fund_flow import (
    calculate_fund_flow_signals,
    update_institutional_flow_cache,
)
from sector_rotation.snapshots import (
    all_research_tickers,
    build_rrg_snapshots,
    build_rotation_snapshots,
    update_price_cache,
)
from sector_rotation.taiwan import (
    TW_BENCHMARK,
    TW_DEFENSIVE_ASSET,
    TW_THEME_CODES,
    all_taiwan_research_assets,
    assets_from_taiwan_themes,
    fetch_taiwan_security_master,
)
from sector_rotation.universe import assets_for, groups_for


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
    us_rrg_metadata = assets_for(
        "Detailed industries — ETFs",
        groups_for("Detailed industries — ETFs"),
    )
    us_written.extend(
        build_rrg_snapshots(
            us_prices,
            SNAPSHOT_ROOT / "us",
            us_rrg_metadata,
            BENCHMARK,
        )
    )

    taiwan_master = fetch_taiwan_security_master()
    taiwan_database_dir = DATABASE_DIR / "tw"
    taiwan_database_dir.mkdir(parents=True, exist_ok=True)
    taiwan_master.to_csv(
        taiwan_database_dir / "security-master.csv",
        index=False,
    )
    taiwan_assets = all_taiwan_research_assets(taiwan_master)
    taiwan_tickers = list(
        dict.fromkeys([*taiwan_assets, TW_BENCHMARK, TW_DEFENSIVE_ASSET])
    )
    tw_prices = update_price_cache(
        taiwan_database_dir / "adjusted-prices.parquet",
        taiwan_tickers,
    )
    coverage = taiwan_master[
        [
            "Code",
            "Yahoo ticker",
            "Name",
            "Market",
            "Asset type",
            "Industry",
        ]
    ].copy()
    coverage["Price observations"] = coverage["Yahoo ticker"].map(
        tw_prices.notna().sum().to_dict()
    ).fillna(0).astype(int)
    coverage["In price database"] = coverage["Price observations"] > 0
    # Roughly one trading year is required for the 52-week / 12-month
    # momentum components used by the default research configurations.
    coverage["Backtest ready"] = coverage["Price observations"] >= 252
    coverage.to_csv(taiwan_database_dir / "coverage-report.csv", index=False)
    institutional_flows = update_institutional_flow_cache(
        taiwan_database_dir / "institutional-flows.parquet",
        tw_prices.index[-30:],
    )
    stock_flows, industry_flows = calculate_fund_flow_signals(
        tw_prices,
        institutional_flows,
        taiwan_master,
    )
    taiwan_snapshot_dir = SNAPSHOT_ROOT / "tw"
    taiwan_snapshot_dir.mkdir(parents=True, exist_ok=True)
    if not stock_flows.empty:
        stock_flows.to_csv(
            taiwan_snapshot_dir / "latest-stock-fund-flow.csv",
            index=False,
        )
    if not industry_flows.empty:
        industry_flows.to_csv(
            taiwan_snapshot_dir / "latest-industry-fund-flow.csv",
            index=False,
        )
    tw_written = build_rotation_snapshots(
        tw_prices,
        SNAPSHOT_ROOT / "tw",
        assets=list(taiwan_assets),
        benchmark=TW_BENCHMARK,
        defensive_asset=TW_DEFENSIVE_ASSET,
    )
    tw_rrg_metadata = assets_from_taiwan_themes(
        taiwan_master,
        list(TW_THEME_CODES),
    )
    tw_written.extend(
        build_rrg_snapshots(
            tw_prices,
            SNAPSHOT_ROOT / "tw",
            tw_rrg_metadata,
            TW_BENCHMARK,
        )
    )

    print(
        f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
        f"US: {len(us_prices.columns)} tickers through {us_prices.index.max():%Y-%m-%d}, "
        f"{len(us_written)} files; "
        f"TW master: {len(taiwan_master)} securities "
        f"({(taiwan_master['Asset type'] == '股票').sum()} stocks, "
        f"{(taiwan_master['Asset type'] == 'ETF').sum()} ETFs); "
        f"TW prices: {coverage['In price database'].sum()} tickers through "
        f"{tw_prices.index.max():%Y-%m-%d}, "
        f"{len(tw_written)} files"
    )


if __name__ == "__main__":
    main()
