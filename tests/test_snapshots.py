import numpy as np
import pandas as pd

from sector_rotation.snapshots import build_rotation_snapshots, build_rrg_snapshots
from sector_rotation.universe import AssetInfo


def test_snapshot_builder_writes_all_frequencies(tmp_path):
    index = pd.bdate_range("2024-01-01", periods=420)
    prices = pd.DataFrame(
        {
            "XLK": 100 * np.cumprod(np.repeat(1.001, len(index))),
            "XLF": 100 * np.cumprod(np.repeat(1.0005, len(index))),
            "SPY": 100 * np.cumprod(np.repeat(1.0007, len(index))),
            "SHY": 100 * np.cumprod(np.repeat(1.0001, len(index))),
        },
        index=index,
    )
    paths = build_rotation_snapshots(prices, tmp_path, top_n=1)
    assert len(paths) == 6
    assert (tmp_path / "latest-daily-ranking.csv").exists()
    assert (tmp_path / "latest-weekly-ranking.csv").exists()
    assert (tmp_path / "latest-monthly-ranking.csv").exists()


def test_rrg_snapshot_builder_writes_all_frequencies(tmp_path):
    index = pd.bdate_range("2022-01-03", periods=900)
    prices = pd.DataFrame(
        {
            "AAA": 100 * np.cumprod(np.repeat(1.0010, len(index))),
            "BBB": 100 * np.cumprod(np.repeat(1.0006, len(index))),
            "CCC": 100 * np.cumprod(np.repeat(0.9999, len(index))),
            "SPY": 100 * np.cumprod(np.repeat(1.0004, len(index))),
        },
        index=index,
    )
    metadata = {
        "AAA": AssetInfo("AAA", "Alpha", "Growth"),
        "BBB": AssetInfo("BBB", "Beta", "Growth"),
        "CCC": AssetInfo("CCC", "Gamma", "Defensive"),
    }
    paths = build_rrg_snapshots(prices, tmp_path, metadata, "SPY")
    assert len(paths) == 6
    assert (tmp_path / "latest-daily-industry-rotation.csv").exists()
    assert (tmp_path / "latest-weekly-industry-rotation.csv").exists()
    assert (tmp_path / "latest-monthly-industry-rotation.csv").exists()
