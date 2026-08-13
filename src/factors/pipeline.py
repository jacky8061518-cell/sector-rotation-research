"""Pure, single-cross-section factor preprocessing functions."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from .config import PipelineConfig, StandardizeMethod, WinsorMethod
from .spec import FactorSpec

DEFAULT_PIPELINE_CONFIG = PipelineConfig()


@dataclass(frozen=True)
class PipelineDiagnostics:
    universe_count: int
    raw_valid_count: int
    output_count: int
    excluded_missing_count: int
    coverage: float
    signal_produced: bool
    reason: str | None = None


@dataclass(frozen=True)
class PipelineResult:
    values: pd.Series
    diagnostics: PipelineDiagnostics


def _finite(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def coverage_ratio(values: pd.Series, universe: pd.Index) -> float:
    if len(universe) == 0:
        return 0.0
    return float(_finite(values.reindex(universe)).notna().sum() / len(universe))


def winsorize_cross_section(
    values: pd.Series,
    method: WinsorMethod = "quantile",
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    mad_threshold: float = 5.0,
) -> pd.Series:
    clean = _finite(values)
    valid = clean.dropna()
    if method == "none" or valid.empty:
        return clean
    result = clean.copy()
    if method == "quantile":
        lower, upper = valid.quantile([lower_quantile, upper_quantile])
    elif method == "mad":
        median = float(valid.median())
        mad = float((valid - median).abs().median())
        if mad == 0:
            return result
        lower = median - mad_threshold * mad
        upper = median + mad_threshold * mad
    else:
        raise ValueError(f"Unknown winsorization method: {method}")
    result.loc[valid.index] = valid.clip(lower=float(lower), upper=float(upper))
    return result


def standardize_cross_section(
    values: pd.Series,
    method: StandardizeMethod = "zscore",
) -> pd.Series:
    clean = _finite(values)
    valid = clean.dropna()
    result = pd.Series(np.nan, index=clean.index, dtype=float, name=clean.name)
    if method == "none" or valid.empty:
        result.loc[valid.index] = valid
        return result
    if method == "zscore":
        deviation = float(valid.std(ddof=0))
        if deviation == 0:
            result.loc[valid.index] = 0.0
        else:
            result.loc[valid.index] = (valid - float(valid.mean())) / deviation
    elif method == "rank_normal":
        probabilities = (valid.rank(method="average") - 0.5) / len(valid)
        normal = NormalDist()
        result.loc[valid.index] = probabilities.map(normal.inv_cdf)
    else:
        raise ValueError(f"Unknown standardization method: {method}")
    return result


def neutralize_cross_section(
    values: pd.Series,
    industry: pd.Series | None = None,
    log_market_cap: pd.Series | None = None,
    *,
    neutralize_industry: bool = True,
    neutralize_size: bool = True,
) -> pd.Series:
    """Return OLS residuals using only the supplied point-in-time cross-section."""
    clean = _finite(values)
    design_parts: list[pd.DataFrame] = []
    if neutralize_industry:
        if industry is None:
            raise ValueError("industry is required for industry neutralization.")
        categories = industry.reindex(clean.index).astype("string")
        design_parts.append(pd.get_dummies(categories, prefix="industry", drop_first=True, dtype=float))
    if neutralize_size:
        if log_market_cap is None:
            raise ValueError("log_market_cap is required for size neutralization.")
        design_parts.append(_finite(log_market_cap.reindex(clean.index)).rename("log_market_cap").to_frame())
    if not design_parts:
        return clean
    design = pd.concat(design_parts, axis=1)
    design.insert(0, "intercept", 1.0)
    combined = pd.concat([clean.rename("value"), design], axis=1).dropna()
    result = pd.Series(np.nan, index=clean.index, dtype=float, name=clean.name)
    if combined.empty or len(combined) <= combined.shape[1]:
        return result
    matrix = combined.drop(columns="value").to_numpy(dtype=float)
    target = combined["value"].to_numpy(dtype=float)
    coefficients, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    result.loc[combined.index] = target - matrix @ coefficients
    return result


def preprocess_factor(
    values: pd.Series,
    spec: FactorSpec,
    universe: pd.Index,
    config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
    *,
    industry: pd.Series | None = None,
    market_cap: pd.Series | None = None,
) -> PipelineResult:
    raw = _finite(values.reindex(universe))
    raw_valid_count = int(raw.notna().sum())
    coverage = coverage_ratio(raw, universe)
    if coverage < config.minimum_coverage:
        diagnostics = PipelineDiagnostics(
            universe_count=len(universe),
            raw_valid_count=raw_valid_count,
            output_count=0,
            excluded_missing_count=len(universe) - raw_valid_count,
            coverage=coverage,
            signal_produced=False,
            reason="coverage_below_threshold",
        )
        return PipelineResult(pd.Series(dtype=float, name=spec.name), diagnostics)

    processed = winsorize_cross_section(
        raw,
        config.winsor_method,
        lower_quantile=config.lower_quantile,
        upper_quantile=config.upper_quantile,
        mad_threshold=config.mad_threshold,
    )
    processed = standardize_cross_section(processed, config.standardize_method)
    if config.neutralize_industry or config.neutralize_size:
        log_market_cap = None
        if market_cap is not None:
            positive = _finite(market_cap.reindex(universe)).where(lambda item: item > 0)
            log_market_cap = np.log(positive)
        processed = neutralize_cross_section(
            processed,
            industry,
            log_market_cap,
            neutralize_industry=config.neutralize_industry,
            neutralize_size=config.neutralize_size,
        )
    processed = (processed * spec.direction).dropna()
    processed.name = spec.name
    diagnostics = PipelineDiagnostics(
        universe_count=len(universe),
        raw_valid_count=raw_valid_count,
        output_count=len(processed),
        excluded_missing_count=len(universe) - raw_valid_count,
        coverage=coverage,
        signal_produced=not processed.empty,
    )
    return PipelineResult(processed, diagnostics)
