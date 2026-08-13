from __future__ import annotations

import pandas as pd

from factors.context import DataContext, FactorDataStore, wide_prices_to_panel


def test_prices_are_hard_truncated_at_asof() -> None:
    prices = pd.DataFrame(
        {"2330.TW": [100.0, 101.0, 999.0]},
        index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
    )
    universe = pd.DataFrame({"ticker": ["2330.TW"], "market": ["TW"]})
    ctx = DataContext(
        FactorDataStore(wide_prices_to_panel(prices), universe),
        "TW",
        pd.Timestamp("2025-01-03"),
        "0050.TW",
    )

    visible = ctx.prices(("adj_close",))

    assert visible.index.get_level_values("date").max() == pd.Timestamp("2025-01-03")
    assert 999.0 not in visible["adj_close"].to_numpy()


def test_revenue_uses_publication_timestamp_not_period_end() -> None:
    revenue = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-12-31")] * 2,
            "ticker": ["2330.TW", "2317.TW"],
            "revenue": [100.0, 200.0],
            "published_at": [
                pd.Timestamp("2025-01-10 12:00"),
                pd.Timestamp("2025-01-10 18:00"),
            ],
        }
    )
    universe = pd.DataFrame({"ticker": ["2330.TW", "2317.TW"], "market": ["TW", "TW"]})
    ctx = DataContext(
        FactorDataStore(
            pd.DataFrame(columns=["date", "ticker", "adj_close"]),
            universe,
            revenue_data=revenue,
        ),
        "TW",
        pd.Timestamp("2025-01-10"),
        "0050.TW",
    )

    visible = ctx.revenue()

    assert visible.index.get_level_values("ticker").tolist() == ["2330.TW"]


def test_universe_respects_effective_dates() -> None:
    universe = pd.DataFrame(
        {
            "ticker": ["ACTIVE.TW", "FUTURE.TW", "DELISTED.TW"],
            "market": ["TW"] * 3,
            "valid_from": ["2020-01-01", "2026-01-01", "2020-01-01"],
            "valid_to": [None, None, "2024-12-31"],
        }
    )
    ctx = DataContext(
        FactorDataStore(pd.DataFrame(columns=["date", "ticker", "adj_close"]), universe),
        "TW",
        pd.Timestamp("2025-01-10"),
        "0050.TW",
    )

    assert ctx.universe().tolist() == ["ACTIVE.TW"]
