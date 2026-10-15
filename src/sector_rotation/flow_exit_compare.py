"""Historical flow-score portfolio comparison with executable exit timing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from factors.backtest import performance_metrics
from factors.costs import DEFAULT_COST_CONFIG, CostConfig, transaction_cost
from sector_rotation.data import repair_taiwan_price_discontinuities


@dataclass(frozen=True)
class FlowScoreHistory:
    score: pd.DataFrame
    flow_5d: pd.DataFrame
    flow_20d: pd.DataFrame
    close: pd.DataFrame
    ma_5: pd.DataFrame
    ma_10: pd.DataFrame
    ma_20: pd.DataFrame
    history_count: pd.DataFrame
    benchmark_close: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


@dataclass(frozen=True)
class ExitVariant:
    name: str
    hold_rank: int = 5
    minimum_hold_score: float = 0.0
    ma_window: int | None = 5
    ma_confirmation_days: int = 1
    forced_weekly_liquidation: bool = False
    hard_stop: float | None = None
    trailing_start: float | None = None
    trailing_drawdown: float | None = None
    maximum_industry_names: int | None = None
    preserve_overlap_weights: bool = False
    market_regime_filter: bool = False
    risk_on_exposure: float = 1.0
    neutral_exposure: float = 0.6
    risk_off_exposure: float = 0.2


@dataclass(frozen=True)
class ExitBacktestResult:
    returns: pd.DataFrame
    equity: pd.DataFrame
    holdings: pd.DataFrame
    trades: pd.DataFrame
    metrics: pd.Series


DEFAULT_EXIT_VARIANTS = (
    ExitVariant("A｜現況：每週全換＋5MA", forced_weekly_liquidation=True),
    ExitVariant(
        "A2｜前五不變、只交易進出名單＋5MA",
        preserve_overlap_weights=True,
    ),
    ExitVariant(
        "A3｜前五進／前八出＋分數70＋10MA兩日",
        hold_rank=8,
        minimum_hold_score=70,
        ma_window=10,
        ma_confirmation_days=2,
        preserve_overlap_weights=True,
    ),
    ExitVariant(
        "A4｜A3＋大盤多中空曝險",
        hold_rank=8,
        minimum_hold_score=70,
        ma_window=10,
        ma_confirmation_days=2,
        preserve_overlap_weights=True,
        market_regime_filter=True,
    ),
    ExitVariant(
        "A5｜前五進／分數70且5日流入正續抱",
        hold_rank=1000,
        minimum_hold_score=70,
        ma_window=None,
        preserve_overlap_weights=True,
    ),
)


def build_flow_score_history(
    prices: pd.DataFrame,
    flows: pd.DataFrame,
    master: pd.DataFrame,
    *,
    weights: tuple[float, float, float, float] = (55.0, 20.0, 15.0, 10.0),
) -> FlowScoreHistory:
    """Rebuild the app's weekly flow score for every historical session."""
    common = master.loc[master["Asset type"].eq("股票")].drop_duplicates("Yahoo ticker")
    tickers = common["Yahoo ticker"].astype(str)
    columns = pd.Index(ticker for ticker in tickers if ticker in prices.columns)
    close = repair_taiwan_price_discontinuities(prices.reindex(columns=columns), threshold=0.12).sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    selected = flows.loc[flows["Ticker"].isin(columns)].copy()
    selected["Date"] = pd.to_datetime(selected["Date"]).dt.tz_localize(None).dt.normalize()

    def pivot(column: str) -> pd.DataFrame:
        return (
            selected.pivot_table(index="Date", columns="Ticker", values=column, aggfunc="sum")
            .reindex(index=close.index, columns=columns)
            .fillna(0.0)
        )

    total = pivot("Total net shares")
    trust = pivot("Trust net shares")
    flow_5d_shares = total.rolling(5, min_periods=5).sum()
    flow_20d_shares = total.rolling(20, min_periods=20).sum()
    trust_5d_shares = trust.rolling(5, min_periods=5).sum()
    issued = pd.to_numeric(common.set_index("Yahoo ticker")["Issued shares"], errors="coerce").reindex(columns)
    issued = issued.where(issued.gt(0))
    flow_5d = flow_5d_shares.div(issued, axis=1)
    flow_20d = flow_20d_shares.div(issued, axis=1)
    trust_5d = trust_5d_shares.div(issued, axis=1)
    return_5d = close.pct_change(5, fill_method=None)
    components = (flow_5d, flow_20d, trust_5d, return_5d)
    total_weight = sum(weights)
    score = sum(component.rank(axis=1, pct=True) * weight for component, weight in zip(components, weights, strict=True))
    score = score.div(total_weight).mul(100)
    flow_sessions = pd.DatetimeIndex(selected["Date"].unique())
    score = score.loc[score.index.isin(flow_sessions)]
    benchmark_close = pd.Series(dtype=float)
    if "0050.TW" in prices:
        benchmark_close = repair_taiwan_price_discontinuities(
            prices[["0050.TW"]],
            threshold=0.12,
        )["0050.TW"].sort_index()
        benchmark_close.index = pd.to_datetime(benchmark_close.index).tz_localize(None).normalize()
    return FlowScoreHistory(
        score=score,
        flow_5d=flow_5d.reindex(score.index),
        flow_20d=flow_20d.reindex(score.index),
        close=close,
        ma_5=close.rolling(5, min_periods=5).mean(),
        ma_10=close.rolling(10, min_periods=10).mean(),
        ma_20=close.rolling(20, min_periods=20).mean(),
        history_count=close.notna().rolling(252, min_periods=1).sum(),
        benchmark_close=benchmark_close,
    )


def _weekly_signal_dates(score: pd.DataFrame) -> set[pd.Timestamp]:
    available = score.dropna(how="all").index
    if available.empty:
        return set()
    periods = available.to_period("W-FRI")
    return {pd.Timestamp(item) for item in pd.Series(available, index=available).groupby(periods).last().to_numpy()}


def _select_target(
    date: pd.Timestamp,
    history: FlowScoreHistory,
    active: pd.Series,
    variant: ExitVariant,
    industries: pd.Series,
    *,
    top_n: int,
) -> pd.Series:
    score = history.score.loc[date].dropna().sort_values(ascending=False)
    eligible = score.index[
        history.flow_5d.loc[date, score.index].gt(0)
        & history.flow_20d.loc[date, score.index].gt(0)
        & history.close.loc[date, score.index].ge(10)
        & history.close.loc[date, score.index].gt(history.ma_20.loc[date, score.index])
        & history.history_count.loc[date, score.index].ge(200)
    ]
    ranked = score.reindex(eligible).dropna().sort_values(ascending=False)
    rank = score.rank(method="first", ascending=False)
    held = list(active[active.gt(0)].index)
    retained = [
        ticker
        for ticker in held
        if ticker in score.index
        and rank.get(ticker, np.inf) <= variant.hold_rank
        and score[ticker] >= variant.minimum_hold_score
        and history.flow_5d.at[date, ticker] > 0
    ]
    if variant.hold_rank <= top_n:
        retained = []

    chosen: list[str] = []
    industry_counts: dict[str, int] = {}
    for ticker in [*retained, *ranked.index]:
        if ticker in chosen:
            continue
        industry = str(industries.get(ticker, "未分類"))
        maximum = variant.maximum_industry_names
        if maximum is not None and industry_counts.get(industry, 0) >= maximum:
            continue
        chosen.append(ticker)
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if len(chosen) == top_n:
            break
    return pd.Series(1 / top_n, index=chosen, name="weight", dtype=float)


def _market_exposure(date: pd.Timestamp, history: FlowScoreHistory, variant: ExitVariant) -> float:
    """Return the pre-declared 100%/60%/20% exposure for the current regime."""
    if not variant.market_regime_filter or history.benchmark_close.empty:
        return 1.0
    benchmark = history.benchmark_close.loc[:date].dropna()
    if len(benchmark) < 65:
        return variant.neutral_exposure
    ma_20 = benchmark.rolling(20, min_periods=20).mean()
    ma_60 = benchmark.rolling(60, min_periods=60).mean()
    above_60 = bool(benchmark.iloc[-1] > ma_60.iloc[-1])
    rising_20 = bool(ma_20.iloc[-1] > ma_20.iloc[-6])
    if above_60 and rising_20:
        return variant.risk_on_exposure
    if above_60 or rising_20:
        return variant.neutral_exposure
    return variant.risk_off_exposure


def _preserve_overlap_weights(
    target: pd.Series,
    active: pd.Series,
    variant: ExitVariant,
) -> pd.Series:
    """Keep existing weights and spend turnover only on actual membership changes."""
    if not variant.preserve_overlap_weights or target.empty or active.empty:
        return target
    overlap = target.index.intersection(active.index)
    new_names = target.index.difference(active.index)
    adjusted = pd.Series(0.0, index=target.index, name="weight")
    adjusted.loc[new_names] = target.loc[new_names]
    remaining_capacity = max(0.0, float(target.sum() - adjusted.sum()))
    overlap_weights = active.reindex(overlap).clip(lower=0.0)
    overlap_total = float(overlap_weights.sum())
    if overlap_total > remaining_capacity and overlap_total > 0:
        overlap_weights = overlap_weights.mul(remaining_capacity / overlap_total)
    adjusted.loc[overlap] = overlap_weights
    return adjusted[adjusted.gt(0)]


def run_exit_variant_backtest(
    history: FlowScoreHistory,
    master: pd.DataFrame,
    variant: ExitVariant,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    top_n: int = 5,
    costs: CostConfig = DEFAULT_COST_CONFIG,
) -> ExitBacktestResult:
    """Trade all close-derived signals at the following session's close."""
    dates = history.close.loc[start:end].index
    daily_returns = history.close.pct_change(fill_method=None).reindex(dates)
    industries = master.drop_duplicates("Yahoo ticker").set_index("Yahoo ticker")["Industry"]
    weekly_signals = _weekly_signal_dates(history.score.loc[start:end])
    active = pd.Series(dtype=float)
    pending: tuple[pd.Series, str, bool] | None = None
    entries: dict[str, float] = {}
    highs: dict[str, float] = {}
    below_counts: dict[str, int] = {}
    gross = pd.Series(0.0, index=dates, name="gross")
    cost_series = pd.Series(0.0, index=dates, name="cost")
    turnover = pd.Series(0.0, index=dates, name="turnover")
    holding_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []

    for date_index, date in enumerate(dates):
        asset_return = daily_returns.loc[date]
        if not active.empty:
            aligned = asset_return.reindex(active.index).fillna(0.0)
            gross.loc[date] = float(aligned.dot(active))
            denominator = 1 + gross.loc[date]
            if denominator > 0:
                active = active.mul(1 + aligned).div(denominator)

        if pending is not None:
            target, reason, force_round_trip = pending
            before = active.copy()
            if force_round_trip:
                zero = pd.Series(dtype=float)
                first = transaction_cost(before, zero, "TW", costs)
                second = transaction_cost(zero, target, "TW", costs)
                cost, buys, sells = first[0] + second[0], second[1], first[2]
            else:
                cost, buys, sells = transaction_cost(before, target, "TW", costs)
            cost_series.loc[date] = cost
            turnover.loc[date] = (buys + sells) / 2
            removed = before.index.difference(target.index)
            added = target.index.difference(before.index)
            if force_round_trip:
                removed = before.index
                added = target.index
            for ticker in removed:
                trade_rows.append({"date": date, "ticker": ticker, "action": "sell", "reason": reason})
                entries.pop(ticker, None)
                highs.pop(ticker, None)
                below_counts.pop(ticker, None)
            active = target.copy()
            for ticker in added:
                price = history.close.at[date, ticker]
                entries[ticker] = float(price)
                highs[ticker] = float(price)
                below_counts[ticker] = 0
                trade_rows.append({"date": date, "ticker": ticker, "action": "buy", "reason": reason})
            pending = None

        for ticker in active.index:
            price = history.close.at[date, ticker]
            if pd.notna(price):
                highs[ticker] = max(highs.get(ticker, float(price)), float(price))
            holding_rows.append({"date": date, "ticker": ticker, "weight": active[ticker]})

        decision_target = active.copy()
        decision_reason = ""
        force_round_trip = False
        if date in weekly_signals and date in history.score.index:
            decision_target = _select_target(date, history, active, variant, industries, top_n=top_n)
            decision_target = decision_target.mul(_market_exposure(date, history, variant))
            decision_target = _preserve_overlap_weights(decision_target, active, variant)
            decision_reason = "weekly_rebalance"
            force_round_trip = variant.forced_weekly_liquidation

        exits: list[str] = []
        ma = history.ma_5 if variant.ma_window == 5 else history.ma_10
        for ticker in active.index:
            close = history.close.at[date, ticker]
            moving_average = ma.at[date, ticker]
            if variant.ma_window is None:
                below_counts[ticker] = 0
                ma_exit = False
            else:
                below_counts[ticker] = below_counts.get(ticker, 0) + 1 if close < moving_average else 0
                ma_exit = below_counts[ticker] >= variant.ma_confirmation_days
            hard_exit = bool(
                variant.hard_stop is not None
                and ticker in entries
                and close <= entries[ticker] * (1 - variant.hard_stop)
            )
            trailing_exit = bool(
                variant.trailing_start is not None
                and variant.trailing_drawdown is not None
                and ticker in entries
                and highs.get(ticker, close) >= entries[ticker] * (1 + variant.trailing_start)
                and close <= highs.get(ticker, close) * (1 - variant.trailing_drawdown)
            )
            if ma_exit or hard_exit or trailing_exit:
                exits.append(ticker)
        if exits:
            decision_target = decision_target.drop(index=exits, errors="ignore")
            decision_reason = "risk_exit"
            force_round_trip = False

        if date_index + 1 < len(dates) and (
            not decision_target.sort_index().equals(active.sort_index()) or force_round_trip
        ):
            pending = (decision_target, decision_reason, force_round_trip)

    net = gross.sub(cost_series).rename("net")
    returns = pd.concat([gross, net, cost_series, turnover], axis=1)
    equity = returns[["gross", "net"]].add(1).cumprod()
    metrics = pd.Series(performance_metrics(net))
    metrics["total_return"] = float(equity["net"].iloc[-1] - 1)
    metrics["gross_total_return"] = float(equity["gross"].iloc[-1] - 1)
    metrics["terminal_cost_drag"] = metrics["gross_total_return"] - metrics["total_return"]
    metrics["total_cost"] = float(cost_series.sum())
    metrics["total_turnover"] = float(turnover.sum())
    metrics["annual_turnover"] = float(turnover.sum() / max(len(turnover) / 252, 1 / 252))
    metrics["trade_count"] = sum(row["action"] == "buy" for row in trade_rows)
    metrics["execution_days"] = int(cost_series.gt(0).sum())
    metrics["average_names"] = (
        float(pd.DataFrame(holding_rows).groupby("date")["ticker"].nunique().mean()) if holding_rows else 0.0
    )
    metrics["average_exposure"] = (
        float(pd.DataFrame(holding_rows).groupby("date")["weight"].sum().mean()) if holding_rows else 0.0
    )
    return ExitBacktestResult(
        returns=returns,
        equity=equity,
        holdings=pd.DataFrame(holding_rows),
        trades=pd.DataFrame(trade_rows),
        metrics=metrics,
    )


def compare_exit_variants(
    history: FlowScoreHistory,
    master: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    variants: tuple[ExitVariant, ...] = DEFAULT_EXIT_VARIANTS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, ExitBacktestResult]]:
    results = {
        variant.name: run_exit_variant_backtest(history, master, variant, start=start, end=end)
        for variant in variants
    }
    metrics = pd.DataFrame({name: result.metrics for name, result in results.items()}).T
    equity = pd.DataFrame({name: result.equity["net"] for name, result in results.items()})
    metrics.index.name = "strategy"
    equity.index.name = "date"
    return metrics, equity, results
