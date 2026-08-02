import numpy as np
import pandas as pd

from sector_rotation.rrg import (
    build_group_indices,
    build_rotation_summary,
    calculate_group_rrg,
    classify_quadrant,
)
from sector_rotation.universe import AssetInfo


def test_quadrant_classification():
    assert classify_quadrant(101, 1) == "Leading"
    assert classify_quadrant(101, -1) == "Weakening"
    assert classify_quadrant(99, -1) == "Lagging"
    assert classify_quadrant(99, 1) == "Improving"


def test_group_indices_are_equal_weighted_from_asset_returns():
    index = pd.date_range("2024-01-31", periods=3, freq="ME")
    prices = pd.DataFrame(
        {
            "AAA": [100.0, 110.0, 121.0],
            "BBB": [100.0, 100.0, 100.0],
            "SPY": [100.0, 102.0, 104.0],
        },
        index=index,
    )
    metadata = {
        "AAA": AssetInfo("AAA", "Alpha", "Technology"),
        "BBB": AssetInfo("BBB", "Beta", "Technology"),
    }
    group_indices = build_group_indices(prices, metadata)
    assert group_indices.loc[index[1], "Technology"] == 1.05


def test_rrg_summary_identifies_leaders_and_explains_breadth():
    index = pd.date_range("2020-01-31", periods=36, freq="ME")
    prices = pd.DataFrame(
        {
            "AAA": 100 * np.cumprod(np.repeat(1.03, len(index))),
            "BBB": 100 * np.cumprod(np.repeat(1.02, len(index))),
            "CCC": 100 * np.cumprod(np.repeat(0.995, len(index))),
            "SPY": 100 * np.cumprod(np.repeat(1.01, len(index))),
        },
        index=index,
    )
    metadata = {
        "AAA": AssetInfo("AAA", "Alpha", "Growth"),
        "BBB": AssetInfo("BBB", "Beta", "Growth"),
        "CCC": AssetInfo("CCC", "Gamma", "Defensive"),
    }
    ratio, momentum, group_indices = calculate_group_rrg(
        prices,
        metadata,
        "SPY",
        long_window=12,
        momentum_window=3,
    )
    summary = build_rotation_summary(
        prices,
        metadata,
        "SPY",
        ratio,
        momentum,
        group_indices,
        short_window=3,
        long_window=12,
    )
    growth = summary.set_index("Group").loc["Growth"]
    assert growth["Quadrant"] == "Leading"
    assert growth["相對廣度"] == 1.0
    assert "AAA" in growth["主要帶動股票"]
    assert "短期超額" in growth["轉強／轉弱原因"]
