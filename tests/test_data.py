import numpy as np
import pandas as pd
import pytest

from sector_rotation.data import repair_taiwan_price_discontinuities


def test_repairs_unadjusted_taiwan_four_for_one_split():
    index = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame(
        {
            "0050.TW": [100.0, 102.0, 25.5, 26.0, 26.5],
            "SPY": [100.0, 102.0, 25.5, 26.0, 26.5],
        },
        index=index,
    )

    repaired = repair_taiwan_price_discontinuities(prices)

    assert repaired.loc[index[0], "0050.TW"] == pytest.approx(25.0)
    assert repaired.loc[index[1], "0050.TW"] == pytest.approx(25.5)
    assert repaired["0050.TW"].pct_change().min() > -0.10
    assert repaired["SPY"].equals(prices["SPY"])


def test_preserves_normal_taiwan_market_moves():
    index = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame(
        {"2330.TW": [100.0, 90.0, 99.0, np.nan]},
        index=index,
    )

    repaired = repair_taiwan_price_discontinuities(prices)

    pd.testing.assert_frame_equal(repaired, prices)


def test_rejects_invalid_discontinuity_threshold():
    with pytest.raises(ValueError, match="threshold"):
        repair_taiwan_price_discontinuities(pd.DataFrame(), threshold=1.0)
