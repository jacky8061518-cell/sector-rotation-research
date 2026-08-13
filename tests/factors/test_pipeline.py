from __future__ import annotations

import numpy as np
import pandas as pd

from factors.config import PipelineConfig
from factors.pipeline import (
    neutralize_cross_section,
    preprocess_factor,
    standardize_cross_section,
    winsorize_cross_section,
)
from factors.spec import FactorSpec

SPEC = FactorSpec(
    name="test_factor",
    label="Test",
    category="momentum",
    direction=-1,
    lookback_days=2,
    requires=("prices",),
    markets=("TW",),
    description="test",
)


def test_quantile_winsorization_preserves_missing_values() -> None:
    values = pd.Series([0.0, 1.0, 2.0, 100.0, np.nan], index=list("abcde"))
    result = winsorize_cross_section(values, lower_quantile=0.25, upper_quantile=0.75)

    assert result["a"] == 0.75
    assert result["d"] == 26.5
    assert np.isnan(result["e"])


def test_zscore_uses_only_one_cross_section() -> None:
    result = standardize_cross_section(pd.Series([1.0, 2.0, 3.0]))

    assert np.isclose(result.mean(), 0.0)
    assert np.isclose(result.std(ddof=0), 1.0)


def test_neutralization_removes_size_loading() -> None:
    size = pd.Series([1.0, 2.0, 3.0, 4.0], index=list("abcd"))
    values = 2.0 + 3.0 * size
    result = neutralize_cross_section(
        values, log_market_cap=size, neutralize_industry=False, neutralize_size=True
    )

    assert np.allclose(result.dropna(), 0.0, atol=1e-12)


def test_low_coverage_suppresses_signal_without_imputation() -> None:
    universe = pd.Index(list("abcd"))
    result = preprocess_factor(
        pd.Series({"a": 1.0, "b": np.nan}),
        SPEC,
        universe,
        PipelineConfig(minimum_coverage=0.60),
    )

    assert result.values.empty
    assert result.diagnostics.reason == "coverage_below_threshold"
    assert result.diagnostics.excluded_missing_count == 3


def test_direction_is_applied_after_processing() -> None:
    result = preprocess_factor(
        pd.Series({"a": 1.0, "b": 2.0, "c": 3.0}),
        SPEC,
        pd.Index(list("abc")),
        PipelineConfig(minimum_coverage=1.0, winsor_method="none"),
    )

    assert result.values["a"] > result.values["c"]
