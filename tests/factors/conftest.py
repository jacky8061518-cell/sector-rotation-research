from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.context import DataContext, FactorDataStore, wide_prices_to_panel


@pytest.fixture
def price_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=300)
    returns = {
        "AAA.TW": np.full(len(dates), 0.001),
        "BBB.TW": np.linspace(-0.002, 0.002, len(dates)),
        "0050.TW": np.full(len(dates), 0.0005),
    }
    return pd.DataFrame(
        {ticker: 100 * np.cumprod(1 + values) for ticker, values in returns.items()},
        index=dates,
    )


@pytest.fixture
def context(price_frame: pd.DataFrame) -> DataContext:
    universe = pd.DataFrame(
        {
            "ticker": ["AAA.TW", "BBB.TW", "0050.TW"],
            "market": ["TW", "TW", "TW"],
            "industry": ["科技", "金融", "市場基準"],
            "valid_from": [pd.Timestamp("2020-01-01")] * 3,
            "valid_to": [pd.NaT] * 3,
        }
    )
    return DataContext(
        FactorDataStore(wide_prices_to_panel(price_frame), universe),
        "TW",
        price_frame.index[-1],
        "0050.TW",
    )
