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

Version 0.3 adds:

- an incremental Parquet market-data cache;
- dated daily, weekly, and monthly ranking snapshots;
- scheduled weekday updates after the U.S. close;
- ETF top-holdings retrieval;
- holding-weight × stock-momentum leadership analysis;
- an in-app "attention and leading players" workflow.

Version 0.4 adds a Taiwan market system with:

- official TWSE and TPEx company-master ingestion;
- automatic `.TW` and `.TWO` ticker resolution;
- official Taiwan industry groups and curated Taiwan themes;
- Taiwan-listed equity and bond ETF universes;
- `0050.TW` as the equity benchmark and `00679B.TWO` as the defensive asset;
- separate U.S. and Taiwan Parquet databases and ranking histories.

Version 0.4.1 adds a Taiwan corporate-action data-quality guard. Yahoo Finance
occasionally leaves a split or capital-reduction boundary unadjusted. Internal
Taiwan price jumps above 40% are now rebased before signal and return
calculation, and the same repair is applied to previously saved caches. This
fixes the false 75% drop in the 0050 history at the start of 2014.

Version 0.4.2 makes the 0050 benchmark return basis explicit. The performance
chart now shows both dividend-reinvested total return and split-adjusted,
price-only return. The official 4-for-1 split effective 2025-06-18 is treated
as a unit conversion and never as an investment gain or loss.

Version 0.4.3 versions the Streamlit price-data cache so stale pre-fix 0050
history cannot survive a code update. The Taiwan performance view also reports
the active pipeline version, benchmark start date, cumulative benchmark growth,
and maximum adjusted daily move. Curated theme backtests now display an
explicit survivorship- and selection-bias warning.

Version 0.4.4 adds an independently adjustable backtest end date. Every
sidebar parameter change now reruns the requested historical window
automatically, including market, universe, groups, rebalance frequency,
start/end dates, top-N, weighting, risk adjustment, momentum filter,
defensive asset, turnover cost, and multi-horizon momentum weights.

Version 0.4.5 presents the main performance chart as cumulative return
percentages instead of growth-of-$1 multiples. Hover labels now identify each
series and report an unambiguous cumulative-return percentage.

Version 0.4.6 adds direct numeric inputs for multi-horizon momentum weights and
a `Custom rank weight` portfolio method. Users can enter Rank 1 through Rank N
position percentages; the app normalizes the inputs and applies them to each
period's newly selected securities.

Version 0.5.0 adds an explainable industry-rotation workbench:

- RRG-style Leading, Improving, Weakening, and Lagging quadrants;
- configurable RS-Ratio, RS-Momentum, and trajectory windows;
- equal-weight industry indices relative to SPY or 0050;
- short- and medium-horizon excess returns;
- relative breadth showing how many constituents beat the benchmark;
- top stocks driving each industry's move;
- deterministic explanations for why each group is strengthening or weakening;
- downloadable industry-rotation diagnostics.

Version 0.6.0 expands Taiwan coverage from curated baskets to the current
official full market:

- every company in the TWSE and TPEx company masters;
- every currently quoted TWSE and TPEx ETF, excluding ETNs;
- uncapped official-industry selection;
- official market, asset-type, industry, and ETF-type metadata;
- chunked full-history downloads with a daily incremental Parquet cache;
- a security master and coverage report showing observation counts and
  backtest readiness for every security.

Version 0.7.0 adds a Taiwan institutional fund-flow layer:

- official TWSE and TPEx foreign-investor, investment-trust, and dealer flows;
- 5-day flow acceleration and 20-day accumulation;
- estimated net-flow value and market-cap-normalized flow intensity;
- industry flow breadth and the leading stocks behind each group;
- separate accumulation, price-confirmation, deceleration, and outflow stages;
- adjustable flow/price weights and downloadable stock/industry flow tables.

Version 0.8.0 makes fund flow the primary Taiwan research surface:

- a fund-flow summary appears before momentum and performance;
- the fund-flow radar is the first analysis tab;
- investor-to-industry Sankey flows;
- five-dimensional industry radar comparisons;
- daily industry flow trajectories and investor-structure heatmaps;
- flow acceleration, breadth, dominant investor, and concentration diagnostics;
- deterministic flow-reason narratives and research-action labels;
- optional headline lookup for the stocks driving a selected industry's flow.

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

## Deploy on Streamlit Community Cloud

1. Push the project root to a GitHub repository. `app.py`,
   `requirements.txt`, `pyproject.toml`, `src/`, and `.streamlit/config.toml`
   must all be included.
2. In the local app, click **Deploy** and choose
   **Streamlit Community Cloud → Deploy now**.
3. Connect GitHub, then select the repository and the `main` branch.
4. Set **Main file path** to `app.py`.
5. In **Advanced settings**, select Python 3.11. No secrets are required for
   the current public-data configuration.
6. Click **Deploy**. Subsequent pushes to the selected GitHub branch update the
   hosted application automatically.

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
│   ├── fund_flow.py
│   ├── holdings.py
│   ├── metrics.py
│   ├── rrg.py
│   ├── snapshots.py
│   └── strategy.py
├── scripts/
│   └── daily_update.py
└── tests/
    ├── test_app.py
    ├── test_holdings.py
    ├── test_rrg.py
    ├── test_snapshots.py
    └── test_strategy.py
```

## Automated daily update

Run the update manually with:

```bash
.venv/bin/python scripts/daily_update.py
```

The first run downloads full history. Later runs retrieve only a short overlap,
merge it into two separate databases:

```text
data/databases/us/adjusted-prices.parquet
data/databases/tw/adjusted-prices.parquet
data/databases/tw/security-master.csv
data/databases/tw/coverage-report.csv
data/databases/tw/institutional-flows.parquet
```

Market-specific snapshots are written to:

```text
data/snapshots/us/latest-daily-ranking.csv
data/snapshots/us/latest-weekly-ranking.csv
data/snapshots/us/latest-monthly-ranking.csv
data/snapshots/tw/latest-daily-ranking.csv
data/snapshots/tw/latest-weekly-ranking.csv
data/snapshots/tw/latest-monthly-ranking.csv
data/snapshots/us/latest-{daily,weekly,monthly}-industry-rotation.csv
data/snapshots/tw/latest-{daily,weekly,monthly}-industry-rotation.csv
```

The Codex automation created for this project runs at 18:00 America/New_York
on weekdays. Weekend and exchange-holiday runs simply retain the most recent
available trading session.

## Current scope

Version 0.8.0 focuses on fund-flow-first Taiwan research, complete Taiwan
listed/OTC stock and ETF coverage, institutional fund-flow confirmation, and
ETF constituent leadership. Logical
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
unavailable. The Taiwan discontinuity guard reduces large omitted
corporate-action errors but cannot guarantee that every vendor revision is
identified. Backtested and synthetic results do not guarantee future
performance. Nothing in this repository is personalized investment advice.
