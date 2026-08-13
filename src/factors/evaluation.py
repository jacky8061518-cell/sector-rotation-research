"""Factor IC construction, diagnostics, and Newey-West summaries."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import EvaluationConfig
from .context import DataContext
from .pipeline import PipelineDiagnostics, preprocess_factor
from .portfolio import assign_quantiles
from .spec import Factor


@dataclass(frozen=True)
class ICSummary:
    horizon: int
    observations: int
    mean: float
    standard_deviation: float
    information_ratio: float
    newey_west_t: float
    positive_ratio: float
    suspicious: bool


@dataclass(frozen=True)
class EvaluationResult:
    scores: pd.DataFrame
    forward_returns: pd.DataFrame
    ic: pd.DataFrame
    summaries: tuple[ICSummary, ...]
    coverage: pd.Series
    diagnostics: tuple[PipelineDiagnostics, ...]


@dataclass(frozen=True)
class QuantileResult:
    period_returns: pd.DataFrame
    cumulative_returns: pd.DataFrame
    annualized_returns: pd.Series
    monotonicity: float


ContextFactory = Callable[[pd.Timestamp], DataContext]
DEFAULT_EVALUATION_CONFIG = EvaluationConfig()


def build_factor_panel(
    factor: Factor,
    context_factory: ContextFactory,
    signal_dates: Sequence[pd.Timestamp],
    config: EvaluationConfig = DEFAULT_EVALUATION_CONFIG,
) -> tuple[pd.DataFrame, pd.Series, tuple[PipelineDiagnostics, ...]]:
    rows: list[pd.Series] = []
    diagnostics: list[PipelineDiagnostics] = []
    coverage: dict[pd.Timestamp, float] = {}
    for raw_date in signal_dates:
        signal_date = pd.Timestamp(raw_date).normalize()
        ctx = context_factory(signal_date)
        raw = factor.compute(ctx, signal_date)
        market_cap = None
        if config.pipeline.neutralize_size:
            cap = ctx.market_cap(window=1)
            if not cap.empty and "market_cap" in cap:
                market_cap = cap["market_cap"].groupby(level="ticker").last()
        result = preprocess_factor(
            raw,
            factor.spec,
            ctx.universe(),
            config.pipeline,
            industry=ctx.industry_map(),
            market_cap=market_cap,
        )
        diagnostics.append(result.diagnostics)
        coverage[signal_date] = result.diagnostics.coverage
        if not result.values.empty:
            row = result.values.copy()
            row.name = signal_date
            rows.append(row)
    scores = pd.DataFrame(rows).sort_index() if rows else pd.DataFrame()
    scores.index.name = "date"
    coverage_series = pd.Series(coverage, name="coverage", dtype=float).sort_index()
    coverage_series.index.name = "date"
    return scores, coverage_series, tuple(diagnostics)


def compute_forward_returns(
    adjusted_close: pd.DataFrame,
    horizons: Sequence[int] = (5, 20, 60),
    signal_dates: Sequence[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Return long-form close(t+h)/close(t)-1 without filling price gaps."""
    prices = adjusted_close.copy().sort_index()
    prices.index = pd.to_datetime(prices.index)
    frames = []
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError("Forward-return horizons must be positive.")
        forward = prices.shift(-horizon).div(prices).sub(1)
        if signal_dates is not None:
            selected_dates = pd.DatetimeIndex(signal_dates).intersection(forward.index)
            forward = forward.loc[selected_dates]
        forward.index.name = "date"
        forward.columns.name = "ticker"
        frames.append(forward.stack(future_stack=True).dropna().rename(f"forward_{horizon}d"))
    return pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()


def compute_rank_ic(
    scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.DataFrame:
    if scores.empty or forward_returns.empty:
        return pd.DataFrame(columns=forward_returns.columns, dtype=float)
    long_scores = scores.copy()
    long_scores.index.name = "date"
    long_scores.columns.name = "ticker"
    score_series = long_scores.stack(future_stack=True).dropna().rename("score")
    paired = forward_returns.join(score_series, how="inner")

    def rank_correlation(frame: pd.DataFrame, return_column: str) -> float:
        if len(frame) < 3 or frame["score"].nunique(dropna=True) < 2 or frame[return_column].nunique(dropna=True) < 2:
            return float("nan")
        return float(frame["score"].corr(frame[return_column], method="spearman"))

    rows: dict[str, pd.Series] = {}
    for column in forward_returns.columns:
        valid = paired[["score", column]].dropna()
        current_column = column
        rows[column] = valid.groupby(level="date").apply(
            lambda frame, name=current_column: rank_correlation(frame, name),
            include_groups=False,
        )
    result = pd.DataFrame(rows).sort_index()
    result.index.name = "date"
    return result


def summarize_ic(
    ic: pd.Series,
    horizon: int,
    newey_west_lags: int | None = None,
) -> ICSummary:
    valid = pd.to_numeric(ic, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    observations = len(valid)
    if observations == 0:
        return ICSummary(horizon, 0, *(float("nan"),) * 5, suspicious=False)
    mean = float(valid.mean())
    standard_deviation = float(valid.std(ddof=1)) if observations > 1 else float("nan")
    information_ratio = (
        mean / standard_deviation if np.isfinite(standard_deviation) and standard_deviation > 0 else float("nan")
    )
    lags = newey_west_lags
    if lags is None:
        lags = min(observations - 1, max(0, horizon - 1))
    newey_west_t = float("nan")
    if observations >= 2:
        model = sm.OLS(valid.to_numpy(), np.ones((observations, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        newey_west_t = float(model.tvalues[0])
    return ICSummary(
        horizon=horizon,
        observations=observations,
        mean=mean,
        standard_deviation=standard_deviation,
        information_ratio=information_ratio,
        newey_west_t=newey_west_t,
        positive_ratio=float(valid.gt(0).mean()),
        suspicious=bool(np.isfinite(information_ratio) and information_ratio > 1.0),
    )


def evaluate_factor(
    factor: Factor,
    context_factory: ContextFactory,
    adjusted_close: pd.DataFrame,
    signal_dates: Sequence[pd.Timestamp],
    config: EvaluationConfig = DEFAULT_EVALUATION_CONFIG,
) -> EvaluationResult:
    scores, coverage, diagnostics = build_factor_panel(factor, context_factory, signal_dates, config)
    forward_returns = compute_forward_returns(adjusted_close, config.horizons, signal_dates=signal_dates)
    ic = compute_rank_ic(scores, forward_returns)
    summaries = tuple(
        summarize_ic(
            ic.get(f"forward_{horizon}d", pd.Series(dtype=float)),
            horizon,
            config.newey_west_lags,
        )
        for horizon in config.horizons
    )
    return EvaluationResult(scores, forward_returns, ic, summaries, coverage, diagnostics)


def evaluate_quantiles(
    scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    horizon: int = 20,
    quantiles: int = 5,
) -> QuantileResult:
    column = f"forward_{horizon}d"
    if scores.empty or column not in forward_returns:
        empty = pd.DataFrame()
        return QuantileResult(empty, empty, pd.Series(dtype=float), float("nan"))
    score_series = scores.rename_axis(index="date", columns="ticker").stack(future_stack=True).dropna().rename("score")
    paired = forward_returns[[column]].join(score_series, how="inner").dropna()
    paired["quantile"] = paired.groupby(level="date")["score"].transform(
        lambda values: assign_quantiles(values, quantiles).reindex(values.index)
    )
    period_returns = (
        paired.dropna(subset=["quantile"])
        .groupby([paired.dropna(subset=["quantile"]).index.get_level_values("date"), "quantile"])[column]
        .mean()
        .unstack("quantile")
    )
    period_returns.columns = [f"Q{int(item)}" for item in period_returns.columns]
    cumulative = period_returns.add(1).cumprod().sub(1)
    periods_per_year = 252 / horizon
    annualized = period_returns.mean().mul(periods_per_year)
    quantile_numbers = pd.Series(range(1, len(annualized) + 1), index=annualized.index)
    monotonicity = float(annualized.corr(quantile_numbers, method="spearman")) if len(annualized) >= 2 else float("nan")
    return QuantileResult(period_returns, cumulative, annualized, monotonicity)


def factor_correlation(score_panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Average same-date cross-sectional factor correlation matrix."""
    names = list(score_panels)
    matrices = []
    dates = sorted(set().union(*(panel.index for panel in score_panels.values()))) if score_panels else []
    for date in dates:
        cross_section = pd.DataFrame({name: panel.loc[date] for name, panel in score_panels.items() if date in panel.index})
        if len(cross_section.columns) >= 2:
            matrices.append(cross_section.corr(method="spearman"))
    if not matrices:
        return pd.DataFrame(index=names, columns=names, dtype=float)
    return pd.concat(matrices).groupby(level=0).mean().reindex(index=names, columns=names)


def grouped_rank_ic(
    scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
    groups: pd.Series,
    *,
    horizon: int = 20,
) -> pd.DataFrame:
    """Recompute rank IC inside each supplied market, industry, or size group."""
    column = f"forward_{horizon}d"
    if scores.empty or column not in forward_returns:
        return pd.DataFrame()
    long_scores = scores.rename_axis(index="date", columns="ticker").stack(future_stack=True).dropna().rename("score")
    paired = forward_returns[[column]].join(long_scores, how="inner").dropna()
    paired["group"] = groups.reindex(paired.index.get_level_values("ticker")).to_numpy()

    def correlation(frame: pd.DataFrame) -> float:
        if len(frame) < 3 or frame["score"].nunique() < 2 or frame[column].nunique() < 2:
            return float("nan")
        return float(frame["score"].corr(frame[column], method="spearman"))

    result = (
        paired.dropna(subset=["group"])
        .groupby([paired.dropna(subset=["group"]).index.get_level_values("date"), "group"])
        .apply(correlation, include_groups=False)
    )
    return result.unstack("group").sort_index()
