from __future__ import annotations

import numpy as np
import pandas as pd

from factors.evaluation import compute_forward_returns, compute_rank_ic, summarize_ic


def test_forward_returns_are_aligned_from_signal_date() -> None:
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0], "B": [100.0, 90.0, 81.0]},
        index=pd.bdate_range("2025-01-01", periods=3),
    )

    result = compute_forward_returns(prices, (1,))

    assert np.isclose(result.loc[(prices.index[0], "A"), "forward_1d"], 0.10)
    assert (prices.index[-1], "A") not in result.dropna().index


def test_rank_ic_uses_cross_sectional_spearman() -> None:
    dates = pd.bdate_range("2025-01-01", periods=2)
    scores = pd.DataFrame([[1.0, 2.0, 3.0]], index=[dates[0]], columns=list("ABC"))
    forward = pd.DataFrame(
        {"forward_1d": [0.01, 0.02, 0.03]},
        index=pd.MultiIndex.from_product([[dates[0]], list("ABC")], names=["date", "ticker"]),
    )

    result = compute_rank_ic(scores, forward)

    assert result.loc[dates[0], "forward_1d"] == 1.0


def test_ic_summary_flags_implausibly_high_ir() -> None:
    summary = summarize_ic(pd.Series([0.20, 0.21, 0.19, 0.20]), 5, newey_west_lags=1)

    assert summary.information_ratio > 1.0
    assert summary.suspicious
    assert summary.positive_ratio == 1.0
