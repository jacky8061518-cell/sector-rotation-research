from __future__ import annotations

import numpy as np
import pandas as pd

from sector_rotation.flow_exit_compare import (
    ExitVariant,
    FlowScoreHistory,
    build_flow_score_history,
    run_exit_variant_backtest,
)


def _master(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Yahoo ticker": tickers,
            "Asset type": "股票",
            "Issued shares": 1_000_000,
            "Industry": [f"I{index % 3}" for index in range(len(tickers))],
        }
    )


def test_historical_flow_score_has_expected_direction() -> None:
    dates = pd.bdate_range("2024-01-01", periods=30)
    tickers = [f"S{index}.TW" for index in range(6)]
    prices = pd.DataFrame(
        {ticker: np.linspace(20 + index, 25 + index, len(dates)) for index, ticker in enumerate(tickers)},
        index=dates,
    )
    rows = []
    for date in dates:
        for index, ticker in enumerate(tickers):
            rows.append(
                {
                    "Date": date,
                    "Ticker": ticker,
                    "Total net shares": 100 * (index + 1),
                    "Trust net shares": 10 * (index + 1),
                }
            )
    history = build_flow_score_history(prices, pd.DataFrame(rows), _master(tickers))

    latest = history.score.iloc[-1]
    assert latest[tickers[-1]] > latest[tickers[0]]
    assert history.flow_20d.iloc[-1].notna().all()


def test_forced_weekly_liquidation_costs_more_than_rank_buffer() -> None:
    dates = pd.bdate_range("2024-01-01", periods=310)
    tickers = [f"S{index}.TW" for index in range(6)]
    close = pd.DataFrame(
        {ticker: np.linspace(20 + index, 40 + index, len(dates)) for index, ticker in enumerate(tickers)},
        index=dates,
    )
    score = pd.DataFrame(
        {ticker: 100 - index for index, ticker in enumerate(tickers)},
        index=dates,
    )
    positive = pd.DataFrame(0.01, index=dates, columns=tickers)
    history = FlowScoreHistory(
        score=score,
        flow_5d=positive,
        flow_20d=positive,
        close=close,
        ma_5=close.mul(0.99),
        ma_10=close.mul(0.99),
        ma_20=close.mul(0.99),
        history_count=pd.DataFrame(252, index=dates, columns=tickers),
    )
    start = dates[260]
    forced = run_exit_variant_backtest(
        history,
        _master(tickers),
        ExitVariant("forced", forced_weekly_liquidation=True),
        start=start,
        end=dates[-1],
    )
    buffered = run_exit_variant_backtest(
        history,
        _master(tickers),
        ExitVariant("buffer", hold_rank=10, minimum_hold_score=60),
        start=start,
        end=dates[-1],
    )

    assert forced.returns["cost"].sum() > buffered.returns["cost"].sum()
    assert forced.equity["net"].iloc[-1] < buffered.equity["net"].iloc[-1]
    assert forced.trades["date"].min().weekday() == 0


def test_a2_keeps_same_top_five_without_forced_round_trip() -> None:
    dates = pd.bdate_range("2024-01-01", periods=310)
    tickers = [f"S{index}.TW" for index in range(6)]
    close = pd.DataFrame(
        {
            ticker: np.linspace(30.0, 30.0 + index * 4, len(dates))
            for index, ticker in enumerate(tickers)
        },
        index=dates,
    )
    score = pd.DataFrame(
        {ticker: 100 - index for index, ticker in enumerate(tickers)},
        index=dates,
    )
    positive = pd.DataFrame(0.01, index=dates, columns=tickers)
    history = FlowScoreHistory(
        score=score,
        flow_5d=positive,
        flow_20d=positive,
        close=close,
        ma_5=close.mul(0.99),
        ma_10=close.mul(0.99),
        ma_20=close.mul(0.99),
        history_count=pd.DataFrame(252, index=dates, columns=tickers),
    )

    result = run_exit_variant_backtest(
        history,
        _master(tickers),
        ExitVariant("A2", preserve_overlap_weights=True),
        start=dates[260],
        end=dates[-1],
    )

    assert result.metrics["execution_days"] == 1
    assert result.metrics["trade_count"] == 5


def test_market_regime_filter_scales_weekly_target() -> None:
    dates = pd.bdate_range("2024-01-01", periods=310)
    tickers = [f"S{index}.TW" for index in range(6)]
    close = pd.DataFrame(30.0, index=dates, columns=tickers)
    score = pd.DataFrame(
        {ticker: 100 - index for index, ticker in enumerate(tickers)},
        index=dates,
    )
    positive = pd.DataFrame(0.01, index=dates, columns=tickers)
    falling_benchmark = pd.Series(np.linspace(100, 50, len(dates)), index=dates)
    history = FlowScoreHistory(
        score=score,
        flow_5d=positive,
        flow_20d=positive,
        close=close,
        ma_5=close.mul(0.99),
        ma_10=close.mul(0.99),
        ma_20=close.mul(0.99),
        history_count=pd.DataFrame(252, index=dates, columns=tickers),
        benchmark_close=falling_benchmark,
    )

    result = run_exit_variant_backtest(
        history,
        _master(tickers),
        ExitVariant("A4", market_regime_filter=True),
        start=dates[260],
        end=dates[-1],
    )

    assert result.holdings.groupby("date")["weight"].sum().max() <= 0.2 + 1e-12


def test_next_close_execution_does_not_earn_execution_day_return() -> None:
    dates = pd.bdate_range("2024-01-01", periods=310)
    tickers = [f"S{index}.TW" for index in range(6)]
    close = pd.DataFrame(30.0, index=dates, columns=tickers)
    score = pd.DataFrame(
        {ticker: 100 - index for index, ticker in enumerate(tickers)},
        index=dates,
    )
    positive = pd.DataFrame(0.01, index=dates, columns=tickers)
    history = FlowScoreHistory(
        score=score,
        flow_5d=positive,
        flow_20d=positive,
        close=close,
        ma_5=close.mul(0.99),
        ma_10=close.mul(0.99),
        ma_20=close.mul(0.99),
        history_count=pd.DataFrame(252, index=dates, columns=tickers),
    )

    result = run_exit_variant_backtest(
        history,
        _master(tickers),
        ExitVariant("timing", ma_window=None),
        start=dates[260],
        end=dates[-1],
    )

    first_execution = pd.Timestamp(result.trades["date"].min())
    assert result.returns.at[first_execution, "gross"] == 0.0
