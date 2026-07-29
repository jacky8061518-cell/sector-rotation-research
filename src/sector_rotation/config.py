"""Default research universe and display metadata."""

SECTOR_ETFS = {
    "XLB": "Materials",
    "XLC": "Communication Services",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
}

BENCHMARK = "SPY"
DEFENSIVE_ASSET = "SHY"

DEFAULT_LOOKBACK_WEIGHTS = {
    1: 0.10,
    3: 0.20,
    6: 0.30,
    12: 0.40,
}
