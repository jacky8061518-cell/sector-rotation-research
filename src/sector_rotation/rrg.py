"""RRG-style group rotation analytics and explainable leadership diagnostics."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .universe import AssetInfo


QUADRANT_LABELS = {
    "Leading": "領先",
    "Weakening": "轉弱",
    "Lagging": "落後",
    "Improving": "轉強",
}


def classify_quadrant(rs_ratio: float, rs_momentum: float) -> str:
    """Classify one relative-strength observation into an RRG-style quadrant."""
    if pd.isna(rs_ratio) or pd.isna(rs_momentum):
        return "Unavailable"
    if rs_ratio >= 100 and rs_momentum >= 0:
        return "Leading"
    if rs_ratio >= 100 and rs_momentum < 0:
        return "Weakening"
    if rs_ratio < 100 and rs_momentum < 0:
        return "Lagging"
    return "Improving"


def build_group_indices(
    sampled_prices: pd.DataFrame,
    metadata: Mapping[str, AssetInfo],
) -> pd.DataFrame:
    """Create equal-weight rebalanced group indices from available asset returns."""
    groups: dict[str, list[str]] = {}
    for ticker, info in metadata.items():
        if ticker in sampled_prices:
            groups.setdefault(info.group, []).append(ticker)

    asset_returns = sampled_prices.pct_change(fill_method=None)
    group_returns = {
        group: asset_returns[tickers].mean(axis=1, skipna=True)
        for group, tickers in groups.items()
    }
    if not group_returns:
        return pd.DataFrame(index=sampled_prices.index)
    returns = pd.DataFrame(group_returns).fillna(0.0)
    return (1 + returns).cumprod()


def calculate_group_rrg(
    sampled_prices: pd.DataFrame,
    metadata: Mapping[str, AssetInfo],
    benchmark: str,
    long_window: int,
    momentum_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate normalized relative-strength ratio and momentum by group.

    This is an explainable RRG-style calculation, not the proprietary JdK
    formula. Group indices are equal-weighted each observation. Relative
    strength is normalized to its rolling mean at 100, and momentum is the
    percentage change of that normalized ratio.
    """
    if long_window < 2 or momentum_window < 1:
        raise ValueError("RRG windows must be positive and long_window must exceed 1.")
    if benchmark not in sampled_prices:
        raise ValueError(f"Benchmark {benchmark} is unavailable.")

    group_indices = build_group_indices(sampled_prices, metadata)
    benchmark_index = sampled_prices[benchmark].div(
        sampled_prices[benchmark].dropna().iloc[0]
    )
    relative_strength = group_indices.div(benchmark_index, axis=0)
    rs_ratio = relative_strength.div(
        relative_strength.rolling(long_window, min_periods=long_window).mean()
    ) * 100
    rs_momentum = rs_ratio.pct_change(
        momentum_window,
        fill_method=None,
    ) * 100
    return rs_ratio, rs_momentum, group_indices


def build_rotation_summary(
    sampled_prices: pd.DataFrame,
    metadata: Mapping[str, AssetInfo],
    benchmark: str,
    rs_ratio: pd.DataFrame,
    rs_momentum: pd.DataFrame,
    group_indices: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> pd.DataFrame:
    """Build latest quadrant, breadth, leader, and explanation fields."""
    if rs_ratio.empty or rs_momentum.empty:
        return pd.DataFrame()
    benchmark_returns = sampled_prices[benchmark].pct_change(
        short_window,
        fill_method=None,
    )
    benchmark_long_returns = sampled_prices[benchmark].pct_change(
        long_window,
        fill_method=None,
    )
    latest_date = rs_ratio.dropna(how="all").index.max()
    rows: list[dict[str, object]] = []

    for group in rs_ratio.columns:
        ratio = rs_ratio.at[latest_date, group]
        momentum = rs_momentum.at[latest_date, group]
        if pd.isna(ratio) or pd.isna(momentum):
            continue
        quadrant = classify_quadrant(float(ratio), float(momentum))
        group_short_return = group_indices[group].pct_change(
            short_window,
            fill_method=None,
        ).at[latest_date]
        group_long_return = group_indices[group].pct_change(
            long_window,
            fill_method=None,
        ).at[latest_date]
        short_excess = group_short_return - benchmark_returns.at[latest_date]
        long_excess = group_long_return - benchmark_long_returns.at[latest_date]

        tickers = [
            ticker
            for ticker, info in metadata.items()
            if info.group == group and ticker in sampled_prices
        ]
        asset_short = sampled_prices[tickers].pct_change(
            short_window,
            fill_method=None,
        ).loc[latest_date]
        asset_excess = asset_short - benchmark_returns.at[latest_date]
        breadth = float((asset_excess > 0).mean()) if not asset_excess.empty else np.nan
        leaders = asset_excess.dropna().nlargest(3)
        leader_labels = [
            f"{ticker} {metadata[ticker].name} ({value:+.1%})"
            for ticker, value in leaders.items()
        ]

        direction = {
            "Leading": "相對強度高於基準且動能仍在增強",
            "Improving": "相對強度尚低於基準，但動能已翻正",
            "Weakening": "相對強度仍高於基準，但動能正在降溫",
            "Lagging": "相對強度低於基準且動能仍為負",
        }[quadrant]
        breadth_text = (
            f"{breadth:.0%} 成分股跑贏基準" if not pd.isna(breadth) else "廣度不足"
        )
        reason = (
            f"{direction}；短期超額 {short_excess:+.1%}、"
            f"中期超額 {long_excess:+.1%}，{breadth_text}。"
        )
        rows.append(
            {
                "Group": group,
                "Quadrant": quadrant,
                "狀態": QUADRANT_LABELS[quadrant],
                "RS-Ratio": float(ratio),
                "RS-Momentum": float(momentum),
                "短期超額報酬": float(short_excess),
                "中期超額報酬": float(long_excess),
                "相對廣度": breadth,
                "主要帶動股票": "；".join(leader_labels) if leader_labels else "—",
                "轉強／轉弱原因": reason,
                "Signal date": latest_date,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    quadrant_priority = {"Improving": 0, "Leading": 1, "Weakening": 2, "Lagging": 3}
    summary["_priority"] = summary["Quadrant"].map(quadrant_priority)
    return (
        summary.sort_values(
            ["_priority", "RS-Momentum", "短期超額報酬"],
            ascending=[True, False, False],
        )
        .drop(columns="_priority")
        .reset_index(drop=True)
    )
