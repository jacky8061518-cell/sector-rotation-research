# Multi-Layer Rotation Research Lab

An explainable, parameter-driven Streamlit application for researching daily,
weekly, and monthly momentum rotation across sectors, detailed industries,
structural themes, investment styles, and curated thematic stock baskets.

The project is designed for stock researchers who want to identify sector
leadership before moving into individual-company analysis.

## What the application answers

- Which sectors, industries, themes, styles, or stocks currently have the
  strongest composite momentum?
- How does the ranking change when momentum is adjusted for volatility?
- What would a daily, weekly, or monthly top-N portfolio have held historically?
- Does the strategy remain useful after explicit turnover costs?
- How do its return, volatility, Sharpe ratio, and drawdown compare with SPY?

## Research workflow

```text
Adjusted ETF prices
        ↓
Daily / weekly / monthly observations
        ↓
Frequency-specific multi-horizon momentum
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

The one-period shift is deliberate: a ranking observed at the end of a day,
week, or month is applied to the following period. This prevents the backtest
from earning a return before the signal was known.

## Research universes

Version 0.2 includes:

- all 11 U.S. Select Sector SPDR ETFs;
- detailed technology, financial, health-care, industrial, consumer, energy,
  mining, real-estate, and infrastructure industry ETFs;
- AI, robotics, cloud, cybersecurity, clean-energy, nuclear, EV, space,
  genomics, FinTech, blockchain, digital-consumer, and sustainability themes;
- growth, value, dividend, quality, low-volatility, size, international,
  real-asset, and rates ETFs;
- curated stock baskets for AI compute, cloud software, cybersecurity,
  semiconductor equipment, data-center power, nuclear, space and defense,
  EVs, metabolic health, FinTech, and computational biology;
- user-entered custom tickers.

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

Version 0.2 focuses on transparent multi-frequency rotation research. Logical
future extensions include:

1. walk-forward parameter testing;
2. subperiod and market-regime analysis;
3. bootstrap confidence intervals;
4. fundamental and quality factors inside selected sectors;
5. downloadable research reports and target allocations.

## Important limitations

This software is for research and education. It does not include every source
of data-vendor bias, survivorship bias, taxes, bid-ask spread, market impact, or
execution uncertainty. Yahoo Finance data may be delayed, revised, or
unavailable. Backtested and synthetic results do not guarantee future
performance. Nothing in this repository is personalized investment advice.
