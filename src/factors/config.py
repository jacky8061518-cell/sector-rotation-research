"""Configuration objects for Phase 1 factor processing and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

WinsorMethod = Literal["quantile", "mad", "none"]
StandardizeMethod = Literal["zscore", "rank_normal", "none"]


@dataclass(frozen=True)
class PipelineConfig:
    minimum_coverage: float = 0.60
    winsor_method: WinsorMethod = "quantile"
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    mad_threshold: float = 5.0
    standardize_method: StandardizeMethod = "zscore"
    neutralize_industry: bool = False
    neutralize_size: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_coverage <= 1:
            raise ValueError("minimum_coverage must be between zero and one.")
        if not 0 <= self.lower_quantile < self.upper_quantile <= 1:
            raise ValueError("Winsor quantiles must be ordered within [0, 1].")
        if self.mad_threshold <= 0:
            raise ValueError("mad_threshold must be positive.")


@dataclass(frozen=True)
class EvaluationConfig:
    horizons: tuple[int, ...] = (5, 20, 60)
    pipeline: PipelineConfig = PipelineConfig()
    newey_west_lags: int | None = None

    def __post_init__(self) -> None:
        if not self.horizons or any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("Evaluation horizons must be positive.")
        if self.newey_west_lags is not None and self.newey_west_lags < 0:
            raise ValueError("newey_west_lags cannot be negative.")
