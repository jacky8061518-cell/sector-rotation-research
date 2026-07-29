# Sector Rotation Research Lab

An explainable, parameter-driven Streamlit application for researching monthly
momentum rotation across U.S. sector ETFs.

The project is designed for stock researchers who want to identify sector
leadership before moving into individual-company analysis.

## What the application answers

- Which U.S. sectors currently have the strongest composite momentum?
- How does the ranking change when momentum is adjusted for volatility?
- What would a monthly top-N sector portfolio have held historically?
- Does the strategy remain useful after explicit turnover costs?
- How do its return, volatility, Sharpe ratio, and drawdown compare with SPY?

## Research workflow

```text
Adjusted ETF prices
        ↓
Calendar month-end observations
        ↓
1 / 3 / 6 / 12-month momentum
        ↓
Composite or risk-adjusted score
        ↓
Positive-momentum filter and top-N ranking
        ↓
Equal, momentum, or inverse-volatility weights
        ↓
One-month signal shift
        ↓
Cost-aware backtest and research dashboard
```

The one-month shift is deliberate: a ranking observed at month-end is applied
to the following month. This prevents the backtest from earning a return before
the signal was known.

## Universe

The default research universe contains the 11 U.S. Select Sector SPDR ETFs:

| ETF | Sector |
|---|---|
| XLB | Materials |
| XLC | Communication Services |
| XLE | Energy |
| XLF | Financials |
| XLI | Industrials |
| XLK | Technology |
| XLP | Consumer Staples |
| XLRE | Real Estate |
| XLU | Utilities |
| XLV | Health Care |
| XLY | Consumer Discretionary |

SPY is the benchmark. SHY is the defensive asset when the positive-momentum
filter rejects every sector.

## Run locally

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Select **Offline demo** in the sidebar if Yahoo Finance is unavailable. Demo
prices are synthetic and reproducible; they are not historical market data.

## Run tests

```bash
pip install -e ".[dev]"
pytest
```

## Project structure

```text
.
├── app.py
├── src/sector_rotation/
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   └── strategy.py
└── tests/
    └── test_strategy.py
```

## Current scope

Version 0.1 focuses on a transparent monthly ETF rotation model. Logical future
extensions include:

1. walk-forward parameter testing;
2. subperiod and market-regime analysis;
3. bootstrap confidence intervals;
4. individual-stock factor ranking inside selected sectors;
5. downloadable research reports and target allocations.

## Important limitations

This software is for research and education. It does not include every source
of data-vendor bias, survivorship bias, taxes, bid-ask spread, market impact, or
execution uncertainty. Yahoo Finance data may be delayed, revised, or
unavailable. Backtested and synthetic results do not guarantee future
performance. Nothing in this repository is personalized investment advice.
