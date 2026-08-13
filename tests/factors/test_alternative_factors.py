from __future__ import annotations

import pandas as pd
import pytest

from factors.context import DataContext, FactorDataStore
from factors.registry import get_factor


def test_foreign_flow_persistence_counts_only_trailing_positive_days() -> None:
    dates = pd.bdate_range("2025-01-01", periods=5)
    flow = pd.DataFrame(
        {
            "date": dates,
            "ticker": "A.TW",
            "foreign_net_shares": [-1, 2, 3, 4, 5],
        }
    )
    universe = pd.DataFrame({"ticker": ["A.TW"], "market": ["TW"], "eligible": [True]})
    store = FactorDataStore(pd.DataFrame(), universe, inst_flow_data=flow)
    ctx = DataContext(store, "TW", dates[-1], "0050.TW")

    score = get_factor("flow_foreign_persist").compute(ctx, dates[-1])

    assert score["A.TW"] == 4


def test_revenue_factor_uses_only_published_observation() -> None:
    revenue = pd.DataFrame(
        {
            "date": ["2025-02-10", "2025-03-10"],
            "ticker": ["A.TW", "A.TW"],
            "period": ["2025-01", "2025-02"],
            "revenue_yoy": [0.12, 9.99],
            "published_at": ["2025-02-10 12:00", "2025-03-10 18:00"],
        }
    )
    universe = pd.DataFrame({"ticker": ["A.TW"], "market": ["TW"], "eligible": [True]})
    store = FactorDataStore(pd.DataFrame(), universe, revenue_data=revenue)
    ctx = DataContext(store, "TW", pd.Timestamp("2025-03-10 13:30"), "0050.TW")

    score = get_factor("rev_yoy").compute(ctx, ctx.asof)

    assert score["A.TW"] == pytest.approx(0.12)


def test_financial_factor_cannot_see_later_publication() -> None:
    financials = pd.DataFrame(
        {
            "date": ["2025-05-10", "2025-08-10"],
            "ticker": ["A.TW", "A.TW"],
            "period": ["2025Q1", "2025Q2"],
            "roe": [0.08, 0.99],
            "published_at": ["2025-05-10 12:00", "2025-08-10 18:00"],
        }
    )
    universe = pd.DataFrame({"ticker": ["A.TW"], "market": ["TW"], "eligible": [True]})
    store = FactorDataStore(pd.DataFrame(), universe, financial_data=financials)
    ctx = DataContext(store, "TW", pd.Timestamp("2025-08-10 13:30"), "0050.TW")

    score = get_factor("roe").compute(ctx, ctx.asof)

    assert score["A.TW"] == pytest.approx(0.08)
