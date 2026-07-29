import numpy as np
import pandas as pd

from sector_rotation.snapshots import build_rotation_snapshots


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
