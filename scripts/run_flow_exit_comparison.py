"""Run and persist the three-year flow-score exit comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from factors.backtest import performance_metrics
from sector_rotation.data import repair_taiwan_price_discontinuities
from sector_rotation.flow_exit_compare import build_flow_score_history, compare_exit_variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--flows", type=Path, required=True)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    flows = pd.read_parquet(args.flows)
    master = pd.read_csv(args.master)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    history = build_flow_score_history(prices, flows, master)
    metrics, equity, results = compare_exit_variants(history, master, start=start, end=end)

    benchmark_price = repair_taiwan_price_discontinuities(prices[["0050.TW"]], threshold=0.12)["0050.TW"].loc[start:end]
    benchmark_returns = benchmark_price.pct_change(fill_method=None).fillna(0.0)
    benchmark_metrics = pd.Series(performance_metrics(benchmark_returns))
    benchmark_equity = benchmark_returns.add(1).cumprod()
    benchmark_metrics["total_return"] = float(benchmark_equity.iloc[-1] - 1)
    benchmark_metrics["gross_total_return"] = benchmark_metrics["total_return"]
    benchmark_metrics["terminal_cost_drag"] = 0.0
    benchmark_metrics["total_cost"] = 0.0
    benchmark_metrics["total_turnover"] = 0.0
    benchmark_metrics["annual_turnover"] = 0.0
    benchmark_metrics["trade_count"] = 1.0
    benchmark_metrics["execution_days"] = 1.0
    benchmark_metrics["average_names"] = 1.0
    benchmark_metrics["average_exposure"] = 1.0
    metrics.loc["0050 買進持有"] = benchmark_metrics
    metrics["excess_vs_0050"] = metrics["total_return"] - benchmark_metrics["total_return"]
    metrics["start"] = start.strftime("%Y-%m-%d")
    metrics["end"] = end.strftime("%Y-%m-%d")
    metrics["sessions"] = len(benchmark_returns)
    equity["0050 買進持有"] = benchmark_equity.reindex(equity.index).ffill()

    yearly = pd.DataFrame(
        {
            name: result.returns["net"].add(1).groupby(result.returns.index.year).prod().sub(1)
            for name, result in results.items()
        }
    ).T
    yearly.columns = yearly.columns.astype(str)
    yearly["全期"] = metrics["total_return"]

    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "flow-exit-comparison-3y.csv", encoding="utf-8-sig")
    equity.to_csv(args.output / "flow-exit-equity-3y.csv", encoding="utf-8-sig")
    yearly.to_csv(args.output / "flow-exit-yearly-3y.csv", encoding="utf-8-sig")
    print(metrics.to_string())
    print("\nYearly returns\n", yearly.to_string())


if __name__ == "__main__":
    main()
