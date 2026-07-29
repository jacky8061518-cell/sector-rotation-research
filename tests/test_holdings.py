import numpy as np
import pandas as pd

from sector_rotation.holdings import analyze_holding_leadership


def test_holding_leadership_combines_weight_and_momentum():
    index = pd.bdate_range("2025-01-01", periods=160)
    prices = pd.DataFrame(
        {
            "FAST": 100 * np.cumprod(np.repeat(1.003, len(index))),
            "SLOW": 100 * np.cumprod(np.repeat(1.001, len(index))),
        },
        index=index,
    )
    holdings = pd.DataFrame(
        {
            "ETF": ["TEST", "TEST"],
            "Holding ticker": ["FAST", "SLOW"],
            "Holding name": ["Fast Corp", "Slow Corp"],
            "Holding weight": [0.10, 0.10],
        }
    )
    result = analyze_holding_leadership(holdings, prices)
    assert result.iloc[0]["Holding ticker"] == "FAST"
    assert result.iloc[0]["ETF rank"] == 1
    assert result.iloc[0]["Leadership score"] > result.iloc[1]["Leadership score"]


def test_holding_leadership_skips_insufficient_history():
    prices = pd.DataFrame(
        {"NEW": np.arange(20.0)},
        index=pd.bdate_range("2026-01-01", periods=20),
    )
    holdings = pd.DataFrame(
        {
            "ETF": ["TEST"],
            "Holding ticker": ["NEW"],
            "Holding name": ["New Corp"],
            "Holding weight": [0.10],
        }
    )
    assert analyze_holding_leadership(holdings, prices).empty
