"""Weekly institutional-flow selection with a daily moving-average exit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FlowStrategyConfig:
    """Configuration for the weekly flow strategy."""

    top_n: int = 5
    flow_window: int = 5
    ma_window: int = 10
    entry_mode: str = "Weekly top inflow"
    new_high_window: int = 60
    require_above_ma: bool = True
    transaction_cost_bps: float = 10.0

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError("top_n must be positive.")
        if self.flow_window <= 0 or self.ma_window <= 0:
            raise ValueError("flow_window and ma_window must be positive.")
        if self.new_high_window <= 1:
            raise ValueError("new_high_window must exceed one session.")
        if self.entry_mode not in {"Weekly top inflow", "New high + inflow"}:
            raise ValueError(f"Unknown entry mode: {self.entry_mode}")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction costs cannot be negative.")


@dataclass
class FlowStrategyResult:
    """Research outputs for a flow-selection strategy."""

    daily_flow_value: pd.DataFrame
    weekly_rankings: pd.DataFrame
    target_weights: pd.DataFrame
    deployed_weights: pd.DataFrame
    gross_returns: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    trade_log: pd.DataFrame
    latest_candidates: pd.DataFrame
    latest_outflows: pd.DataFrame
    current_holdings: pd.DataFrame
    exit_alerts: pd.DataFrame


def estimate_daily_security_flow(
    prices: pd.DataFrame,
    flows: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate daily institutional cash flow as net shares times close."""
    if prices.empty or flows.empty:
        return pd.DataFrame()
    price_data = prices.sort_index().copy()
    price_data.index = pd.to_datetime(price_data.index).tz_localize(None).normalize()
    selected = flows.copy()
    selected["Date"] = pd.to_datetime(selected["Date"]).dt.tz_localize(None).dt.normalize()
    selected = selected[selected["Ticker"].isin(price_data.columns)]
    selected = selected[selected["Date"].isin(price_data.index)]
    if selected.empty:
        return pd.DataFrame()
    closes = (
        price_data.stack(future_stack=True)
        .rename("Close")
        .rename_axis(["Date", "Ticker"])
        .reset_index()
    )
    selected = selected.merge(closes, on=["Date", "Ticker"], how="left")
    selected["Estimated flow value"] = selected["Total net shares"] * selected["Close"]
    return (
        selected.pivot_table(
            index="Date",
            columns="Ticker",
            values="Estimated flow value",
            aggfunc="sum",
        )
        .sort_index()
        .fillna(0.0)
    )


def _weekly_signal_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    periods = index.to_period("W-FRI")
    dates = pd.Series(index, index=index).groupby(periods).last()
    return pd.DatetimeIndex(dates.to_numpy())


def run_weekly_flow_strategy(
    prices: pd.DataFrame,
    flows: pd.DataFrame,
    master: pd.DataFrame,
    config: FlowStrategyConfig,
) -> FlowStrategyResult:
    """Select weekly flow leaders and exit after a close below the daily MA.

    A signal observed at close on day t is deployed for the return from t to
    t+1.  This shift prevents the strategy from earning a return before the
    weekly ranking or moving-average exit was observable.
    """
    if prices.empty or flows.empty:
        raise ValueError("prices and institutional flows are required.")
    price_data = prices.sort_index().copy()
    price_data.index = pd.to_datetime(price_data.index).tz_localize(None).normalize()
    daily_flow = estimate_daily_security_flow(price_data, flows)
    if daily_flow.empty:
        raise ValueError("institutional flows do not overlap the price database.")

    start = daily_flow.index.min()
    available = [ticker for ticker in daily_flow if ticker in price_data.columns]
    full_prices = price_data.reindex(columns=available)
    full_moving_average = full_prices.rolling(
        config.ma_window,
        min_periods=config.ma_window,
    ).mean()
    full_rolling_high = full_prices.rolling(
        config.new_high_window,
        min_periods=config.new_high_window,
    ).max()
    strategy_prices = full_prices.loc[start:].copy()
    daily_flow = daily_flow.reindex(index=strategy_prices.index, columns=available).fillna(0.0)
    rolling_flow = daily_flow.rolling(config.flow_window, min_periods=config.flow_window).sum()
    moving_average = full_moving_average.reindex(strategy_prices.index)
    rolling_high = full_rolling_high.reindex(strategy_prices.index)
    signal_dates = set(_weekly_signal_dates(strategy_prices.index))

    metadata_columns = [
        column
        for column in [
            "Yahoo ticker",
            "Name",
            "Industry",
            "Detailed industry",
            "Investment theme",
            "Supply-chain role",
        ]
        if column in master.columns
    ]
    metadata = (
        master[metadata_columns]
        .drop_duplicates("Yahoo ticker")
        .set_index("Yahoo ticker")
    )

    ranking_rows: list[dict[str, object]] = []
    weekly_selections: dict[pd.Timestamp, list[str]] = {}
    for timestamp in strategy_prices.index:
        if timestamp not in signal_dates or timestamp not in rolling_flow.index:
            continue
        values = rolling_flow.loc[timestamp].dropna()
        candidates = values[values > 0]
        if config.require_above_ma:
            above_ma = strategy_prices.loc[timestamp] >= moving_average.loc[timestamp]
            candidates = candidates[above_ma.reindex(candidates.index).fillna(False)]
        if config.entry_mode == "New high + inflow":
            at_high = strategy_prices.loc[timestamp] >= rolling_high.loc[timestamp]
            candidates = candidates[at_high.reindex(candidates.index).fillna(False)]
        selected = list(candidates.nlargest(min(config.top_n, len(candidates))).index)
        weekly_selections[pd.Timestamp(timestamp)] = selected
        for rank, (ticker, flow_value) in enumerate(
            values.sort_values(ascending=False).items(),
            start=1,
        ):
            row: dict[str, object] = {
                "Signal date": pd.Timestamp(timestamp),
                "Rank": rank,
                "Ticker": ticker,
                "Weekly flow value": float(flow_value),
                "Close": float(strategy_prices.at[timestamp, ticker]),
                "MA": float(moving_average.at[timestamp, ticker]),
                "New high": bool(
                    pd.notna(rolling_high.at[timestamp, ticker])
                    and strategy_prices.at[timestamp, ticker] >= rolling_high.at[timestamp, ticker]
                ),
                "Eligible": ticker in candidates.index,
                "Selected": ticker in selected,
            }
            if ticker in metadata.index:
                row.update(metadata.loc[ticker].to_dict())
            ranking_rows.append(row)
    weekly_rankings = pd.DataFrame(ranking_rows)

    index = strategy_prices.index
    tickers = list(strategy_prices.columns)
    target = pd.DataFrame(0.0, index=index, columns=tickers)
    actions: list[dict[str, object]] = []
    current_target = pd.Series(0.0, index=tickers)
    for position, timestamp in enumerate(index):
        if timestamp in weekly_selections:
            selected = weekly_selections[timestamp]
            new_target = pd.Series(0.0, index=tickers)
            if selected:
                new_target.loc[selected] = 1.0 / len(selected)
            removed = current_target[current_target > 0].index.difference(selected)
            added = pd.Index(selected).difference(current_target[current_target > 0].index)
            execution_date = index[min(position + 1, len(index) - 1)]
            actions.extend(
                {
                    "Signal date": timestamp,
                    "Execution date": execution_date,
                    "Ticker": ticker,
                    "Action": "買入（週排名）",
                    "Reason": f"{config.flow_window}日法人淨流入前{config.top_n}",
                }
                for ticker in added
            )
            actions.extend(
                {
                    "Signal date": timestamp,
                    "Execution date": execution_date,
                    "Ticker": ticker,
                    "Action": "賣出（週調整）",
                    "Reason": "退出本週前五",
                }
                for ticker in removed
            )
            current_target = new_target

        held = current_target[current_target > 0].index
        if len(held):
            below = strategy_prices.loc[timestamp, held] < moving_average.loc[timestamp, held]
            exits = list(below[below.fillna(False)].index)
            if exits:
                execution_date = index[min(position + 1, len(index) - 1)]
                for ticker in exits:
                    current_target.loc[ticker] = 0.0
                    actions.append(
                        {
                            "Signal date": timestamp,
                            "Execution date": execution_date,
                            "Ticker": ticker,
                            "Action": f"賣出（跌破{config.ma_window}日線）",
                            "Reason": (
                                f"收盤 {strategy_prices.at[timestamp, ticker]:.2f} < "
                                f"MA{config.ma_window} {moving_average.at[timestamp, ticker]:.2f}"
                            ),
                        }
                    )
        target.loc[timestamp] = current_target

    deployed = target.shift(1).fillna(0.0)
    asset_returns = strategy_prices.pct_change(fill_method=None)
    gross = (deployed * asset_returns).sum(axis=1, min_count=1).fillna(0.0)
    cash = 1.0 - deployed.sum(axis=1)
    prior_cash = cash.shift(1).fillna(1.0)
    turnover = (
        deployed.diff().abs().sum(axis=1).fillna(0.0)
        + (cash - prior_cash).abs()
    ) / 2
    net = gross - turnover * config.transaction_cost_bps / 10_000

    trade_log = pd.DataFrame(actions)
    if not trade_log.empty:
        trade_log = trade_log.merge(
            metadata.reset_index(),
            left_on="Ticker",
            right_on="Yahoo ticker",
            how="left",
        ).drop(columns=["Yahoo ticker"], errors="ignore")

    latest_signal = max(weekly_selections) if weekly_selections else None
    latest_candidates = (
        weekly_rankings[
            weekly_rankings["Signal date"].eq(latest_signal)
            & weekly_rankings["Eligible"]
        ]
        .sort_values("Weekly flow value", ascending=False)
        .head(max(10, config.top_n))
        .reset_index(drop=True)
        if latest_signal is not None and not weekly_rankings.empty
        else pd.DataFrame()
    )
    if not latest_candidates.empty:
        latest_candidates.insert(
            0,
            "Strategy rank",
            range(1, len(latest_candidates) + 1),
        )
    latest_outflows = (
        weekly_rankings[weekly_rankings["Signal date"].eq(latest_signal)]
        .nsmallest(10, "Weekly flow value")
        .reset_index(drop=True)
        if latest_signal is not None and not weekly_rankings.empty
        else pd.DataFrame()
    )
    latest_target = target.iloc[-1]
    holding_tickers = list(latest_target[latest_target > 0].index)
    holding_rows = []
    for ticker in holding_tickers:
        row = {
            "Ticker": ticker,
            "Target weight": latest_target[ticker],
            "Close": strategy_prices.iloc[-1][ticker],
            f"MA{config.ma_window}": moving_average.iloc[-1][ticker],
            "Distance to MA": strategy_prices.iloc[-1][ticker] / moving_average.iloc[-1][ticker] - 1,
        }
        if ticker in metadata.index:
            row.update(metadata.loc[ticker].to_dict())
        holding_rows.append(row)
    current_holdings = pd.DataFrame(holding_rows)
    exit_alerts = (
        current_holdings[
            current_holdings["Close"] < current_holdings[f"MA{config.ma_window}"]
        ].copy()
        if not current_holdings.empty
        else pd.DataFrame()
    )

    return FlowStrategyResult(
        daily_flow_value=daily_flow,
        weekly_rankings=weekly_rankings,
        target_weights=target,
        deployed_weights=deployed,
        gross_returns=gross.rename("Gross flow strategy"),
        net_returns=net.rename("Net flow strategy"),
        turnover=turnover.rename("Turnover"),
        trade_log=trade_log,
        latest_candidates=latest_candidates,
        latest_outflows=latest_outflows,
        current_holdings=current_holdings,
        exit_alerts=exit_alerts,
    )
