"""Refresh market data and save daily/weekly/monthly ranking snapshots."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from factors.context import FactorDataStore
from factors.data_tw import (
    normalize_institutional_flows,
    save_universe_snapshot,
    update_financial_cache,
    update_monthly_revenue_cache,
)
from factors.snapshots import build_daily_factor_snapshot, update_market_cap_snapshot
from sector_rotation.broker_branch import (
    fetch_histock_broker_branches,
)
from sector_rotation.fund_flow import (
    calculate_fund_flow_signals,
    update_institutional_flow_cache,
)
from sector_rotation.snapshots import (
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_DIR = PROJECT_ROOT / "data" / "databases"
SNAPSHOT_ROOT = PROJECT_ROOT / "data" / "snapshots"


def main() -> None:
    taiwan_master = fetch_taiwan_security_master()
    taiwan_database_dir = DATABASE_DIR / "tw"
    taiwan_database_dir.mkdir(parents=True, exist_ok=True)
    taiwan_master.to_csv(
        taiwan_database_dir / "security-master.csv",
        index=False,
    )
    save_universe_snapshot(
        taiwan_master,
        taiwan_database_dir / "universe-snapshots",
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
        # Keep at least one full trading year so the weekly flow strategy can
        # be evaluated across roughly 52 independent rebalance observations.
        tw_prices.index[-270:],
    )
    try:
        revenue = update_monthly_revenue_cache(
            taiwan_database_dir / "monthly-revenue.parquet"
        )
    except (OSError, TimeoutError, ValueError):
        revenue_path = taiwan_database_dir / "monthly-revenue.parquet"
        revenue = pd.read_parquet(revenue_path) if revenue_path.exists() else pd.DataFrame()
    market_cap = update_market_cap_snapshot(
        taiwan_database_dir / "market-cap.parquet",
        tw_prices,
        taiwan_master,
    )
    try:
        financials = update_financial_cache(
            taiwan_database_dir / "financials-pit.parquet"
        )
    except (OSError, TimeoutError, ValueError):
        financial_path = taiwan_database_dir / "financials-pit.parquet"
        financials = (
            pd.read_parquet(financial_path)
            if financial_path.exists()
            else pd.DataFrame()
        )
    stock_flows, industry_flows = calculate_fund_flow_signals(
        tw_prices,
        institutional_flows,
        taiwan_master,
    )
    taiwan_snapshot_dir = SNAPSHOT_ROOT / "tw"
    taiwan_snapshot_dir.mkdir(parents=True, exist_ok=True)
    factor_universe = pd.DataFrame(
        {
            "ticker": taiwan_master.loc[taiwan_master["Asset type"].eq("股票"), "Yahoo ticker"],
            "market": "TW",
            "industry": taiwan_master.loc[taiwan_master["Asset type"].eq("股票"), "Industry"],
            "eligible": True,
        }
    )
    factor_store = FactorDataStore(
        pd.DataFrame(columns=["date", "ticker", "adj_close"]),
        factor_universe,
        market_cap_data=market_cap,
        inst_flow_data=normalize_institutional_flows(institutional_flows),
        revenue_data=revenue,
        financial_data=financials,
        adjusted_close_wide=tw_prices,
    )
    factor_snapshot = build_daily_factor_snapshot(
        tw_prices,
        taiwan_master,
        factor_store,
        taiwan_snapshot_dir / "latest-factor-snapshot.parquet",
        benchmark=TW_BENCHMARK,
    )
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

    # Keep the branch workload focused: for each research horizon select the ten
    # stocks with the strongest institutional inflow, then fetch the matching
    # public cumulative branch table from HiStock.
    broker_cache_path = taiwan_database_dir / "broker-branches.parquet"
    broker_cache = pd.DataFrame()
    broker_status = "no candidates"
    if not stock_flows.empty:
        stock_only = stock_flows[stock_flows["Asset type"] == "股票"].copy()
        fresh_branch_rows = []
        completed: dict[str, int] = {}
        for horizon, flow_column in {
            "Daily": "1D net value",
            "Weekly": "5D net value",
            "Monthly": "20D net value",
        }.items():
            focus_tickers = (
                stock_only.nlargest(10, flow_column)["Ticker"]
                .str.replace(r"\.(TW|TWO)$", "", regex=True)
                .tolist()
            )
            completed[horizon] = 0
            for ticker in focus_tickers:
                try:
                    frame = fetch_histock_broker_branches(ticker, horizon)
                except (OSError, TimeoutError, ValueError):
                    continue
                if not frame.empty:
                    fresh_branch_rows.append(frame)
                    completed[horizon] += 1
        if fresh_branch_rows:
            combined = (
                pd.concat(fresh_branch_rows, ignore_index=True)
                .drop_duplicates(
                    ["Date", "Ticker", "Broker ID", "Horizon"], keep="last"
                )
                .sort_values(["Horizon", "Ticker", "Broker ID"])
                .reset_index(drop=True)
            )
            combined.to_parquet(broker_cache_path, index=False)
            broker_cache = combined
            broker_status = "HiStock " + ", ".join(
                f"{horizon}={count}" for horizon, count in completed.items()
            )
        else:
            broker_status = "no new rows"
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
        f"TW master: {len(taiwan_master)} securities "
        f"({(taiwan_master['Asset type'] == '股票').sum()} stocks, "
        f"{(taiwan_master['Asset type'] == 'ETF').sum()} ETFs); "
        f"TW prices: {coverage['In price database'].sum()} tickers through "
        f"{tw_prices.index.max():%Y-%m-%d}, "
        f"{len(tw_written)} files; broker branches: {broker_status}, "
        f"{len(broker_cache):,} cached rows; factor snapshot: "
        f"{len(factor_snapshot):,} stocks"
    )


if __name__ == "__main__":
    main()
