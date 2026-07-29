from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sector_rotation.config import (
    BENCHMARK,
    DEFAULT_LOOKBACK_WEIGHTS,
    DEFENSIVE_ASSET,
    SECTOR_ETFS,
)
from sector_rotation.data import download_adjusted_prices, generate_demo_prices
from sector_rotation.metrics import benchmark_returns, drawdown, equity_curve, performance_summary
from sector_rotation.strategy import BacktestConfig, run_backtest


st.set_page_config(
    page_title="Sector Rotation Research Lab",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.8rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.46);
        border: 1px solid rgba(148, 163, 184, 0.20);
        padding: 0.85rem;
        border-radius: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner=False)
def load_live_data(tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    return download_adjusted_prices(list(tickers), start, end)


@st.cache_data(show_spinner=False)
def load_demo_data() -> pd.DataFrame:
    return generate_demo_prices()


def format_percent(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.1%}"


def format_number(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.2f}"


def performance_chart(strategy: pd.Series, benchmark: pd.Series) -> go.Figure:
    frame = pd.concat(
        [equity_curve(strategy).rename("Strategy"), equity_curve(benchmark).rename(BENCHMARK)],
        axis=1,
    ).dropna(how="all")
    figure = px.line(
        frame,
        labels={"value": "Growth of $1", "index": "", "variable": ""},
        color_discrete_sequence=["#38bdf8", "#94a3b8"],
    )
    figure.update_layout(hovermode="x unified", legend_orientation="h", height=430)
    return figure


def allocation_chart(weights: pd.DataFrame) -> go.Figure:
    visible = weights.loc[:, (weights.abs().sum() > 0)]
    renamed = visible.rename(columns={**SECTOR_ETFS, DEFENSIVE_ASSET: "Defensive / SHY"})
    figure = px.area(
        renamed,
        labels={"value": "Portfolio weight", "index": "", "variable": ""},
    )
    figure.update_layout(hovermode="x unified", legend_orientation="h", height=430)
    figure.update_yaxes(tickformat=".0%", range=[0, 1])
    return figure


st.title("Sector Rotation Research Lab")
st.caption(
    "Rank U.S. sector ETFs by composite momentum, rotate monthly, and verify the result "
    "with explicit signal timing, costs, and risk metrics."
)

with st.sidebar:
    st.header("Research controls")
    data_mode = st.radio(
        "Data source",
        ["Live Yahoo Finance", "Offline demo"],
        help="Offline demo uses synthetic data and is not historical performance.",
    )
    start_date = st.date_input("Start date", value=date(2012, 1, 1), max_value=date.today())
    top_n = st.slider("Number of sectors to hold", 1, 6, 3)
    weighting = st.selectbox(
        "Position weighting",
        ["Equal weight", "Momentum weight", "Inverse volatility"],
    )
    risk_adjusted = st.toggle(
        "Risk-adjust momentum",
        value=False,
        help="Divide composite momentum by trailing annualized volatility.",
    )
    positive_filter = st.toggle(
        "Require positive momentum",
        value=True,
        help=f"Move to {DEFENSIVE_ASSET} when no sector has positive momentum.",
    )
    cost_bps = st.number_input(
        "Transaction cost (bps per turnover)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=1.0,
    )

    st.subheader("Composite momentum")
    lookback_weights: dict[int, float] = {}
    for months, default_weight in DEFAULT_LOOKBACK_WEIGHTS.items():
        lookback_weights[months] = st.slider(
            f"{months}-month weight",
            min_value=0,
            max_value=100,
            value=int(default_weight * 100),
            step=5,
        )

if sum(lookback_weights.values()) == 0:
    st.error("At least one momentum weight must be greater than zero.")
    st.stop()

all_tickers = tuple([*SECTOR_ETFS, BENCHMARK, DEFENSIVE_ASSET])
try:
    with st.spinner("Preparing market data and running the research model…"):
        if data_mode == "Live Yahoo Finance":
            prices = load_live_data(all_tickers, start_date, date.today())
        else:
            prices = load_demo_data()
            prices = prices.loc[pd.Timestamp(start_date) :]

        available_sectors = [ticker for ticker in SECTOR_ETFS if ticker in prices]
        config = BacktestConfig(
            lookback_weights=lookback_weights,
            top_n=top_n,
            weighting=weighting,
            require_positive_momentum=positive_filter,
            risk_adjusted_score=risk_adjusted,
            defensive_asset=DEFENSIVE_ASSET,
            transaction_cost_bps=cost_bps,
        )
        result = run_backtest(prices, available_sectors, config)
except Exception as exc:
    st.error(f"Research run failed: {exc}")
    st.info("Try Offline demo to test the application without a market-data connection.")
    st.stop()

if result.net_returns.empty:
    st.warning("The selected date range is too short for the longest lookback period.")
    st.stop()

benchmark = benchmark_returns(result.monthly_prices, BENCHMARK, result.net_returns.index[0])
strategy_metrics = performance_summary(result.net_returns)
benchmark_metrics = performance_summary(benchmark)
latest_signal_date = result.scores.index[-1]

if data_mode == "Offline demo":
    st.warning("Offline demo is active. All prices and performance shown are synthetic.")

st.subheader("Latest model signal")
st.caption(
    f"Signal date: {latest_signal_date:%Y-%m-%d}. This target allocation is applied only "
    "after the signal date in the historical backtest."
)

latest_scores = result.scores.iloc[-1].dropna().sort_values(ascending=False)
latest_weights = result.target_weights.iloc[-1]
ranking = pd.DataFrame(
    {
        "ETF": latest_scores.index,
        "Sector": [SECTOR_ETFS.get(ticker, ticker) for ticker in latest_scores.index],
        "Momentum score": latest_scores.values,
        "Target weight": [latest_weights.get(ticker, 0.0) for ticker in latest_scores.index],
    }
)
ranking.index = range(1, len(ranking) + 1)

allocation = latest_weights[latest_weights > 0].sort_values(ascending=False)
if allocation.empty:
    st.info("The model currently holds cash.")
else:
    cards = st.columns(len(allocation))
    for card, (ticker, weight) in zip(cards, allocation.items()):
        label = SECTOR_ETFS.get(ticker, "Defensive / SHY")
        card.metric(ticker, f"{weight:.1%}", label)

tab_overview, tab_rankings, tab_portfolio, tab_method = st.tabs(
    ["Performance", "Sector rankings", "Portfolio history", "Methodology"]
)

with tab_overview:
    metric_columns = st.columns(5)
    metric_columns[0].metric("Strategy CAGR", format_percent(strategy_metrics["CAGR"]))
    metric_columns[1].metric("Strategy Sharpe", format_number(strategy_metrics["Sharpe"]))
    metric_columns[2].metric("Max drawdown", format_percent(strategy_metrics["Max drawdown"]))
    metric_columns[3].metric("Annual volatility", format_percent(strategy_metrics["Volatility"]))
    metric_columns[4].metric("Average annual turnover", f"{result.turnover.mean() * 12:.1f}x")

    st.plotly_chart(performance_chart(result.net_returns, benchmark), use_container_width=True)

    left, right = st.columns([2, 1])
    with left:
        drawdowns = pd.concat(
            [
                drawdown(result.net_returns).rename("Strategy"),
                drawdown(benchmark).rename(BENCHMARK),
            ],
            axis=1,
        )
        drawdown_figure = px.area(
            drawdowns,
            labels={"value": "Drawdown", "index": "", "variable": ""},
            color_discrete_sequence=["#fb7185", "#94a3b8"],
        )
        drawdown_figure.update_yaxes(tickformat=".0%")
        drawdown_figure.update_layout(hovermode="x unified", height=350)
        st.plotly_chart(drawdown_figure, use_container_width=True)
    with right:
        comparison = pd.DataFrame(
            {"Strategy": strategy_metrics, BENCHMARK: benchmark_metrics}
        )
        display = comparison.astype(object)
        for row in ["CAGR", "Volatility", "Max drawdown", "Win rate"]:
            display.loc[row] = display.loc[row].map(format_percent)
        display.loc["Sharpe"] = display.loc["Sharpe"].map(format_number)
        st.dataframe(display, use_container_width=True)

with tab_rankings:
    st.dataframe(
        ranking.style.format({"Momentum score": "{:.3f}", "Target weight": "{:.1%}"}),
        use_container_width=True,
        height=430,
    )
    score_history = result.scores.rename(columns=SECTOR_ETFS)
    st.plotly_chart(
        px.line(
            score_history.tail(36),
            labels={"value": "Momentum score", "index": "", "variable": ""},
            title="Momentum leadership — trailing 36 months",
        ).update_layout(hovermode="x unified", height=440),
        use_container_width=True,
    )

with tab_portfolio:
    st.plotly_chart(allocation_chart(result.deployed_weights), use_container_width=True)
    st.subheader("Recent deployed weights")
    recent = result.deployed_weights.tail(12).rename(
        columns={**SECTOR_ETFS, DEFENSIVE_ASSET: "Defensive / SHY"}
    )
    recent = recent.loc[:, (recent.abs().sum() > 0)]
    st.dataframe(recent.style.format("{:.1%}"), use_container_width=True)

with tab_method:
    st.markdown(
        f"""
        ### Research sequence

        1. Convert adjusted daily prices to calendar month-end observations.
        2. Calculate {", ".join(f"{month}-month" for month in lookback_weights)} total returns.
        3. Normalize the selected lookback weights and combine them into one score.
        4. Optionally divide the score by trailing annualized volatility.
        5. Keep the top **{top_n}** sectors{" with positive scores" if positive_filter else ""}.
        6. Allocate with **{weighting.lower()}**.
        7. Shift every signal forward by one month before calculating portfolio returns.
        8. Charge **{cost_bps:.0f} bps** according to monthly portfolio turnover.

        ### Interpretation

        This application is a research and education tool. It does not account for taxes,
        bid–ask spreads, market impact, ETF closures, all corporate actions, or every source
        of survivorship and data-vendor bias. Yahoo Finance data may be delayed, revised, or
        unavailable. Synthetic demo results are not historical results. Nothing displayed
        here is personalized investment advice.
        """
    )
