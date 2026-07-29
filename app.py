from __future__ import annotations

from datetime import date, timedelta
import os
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sector_rotation.config import BENCHMARK, DEFENSIVE_ASSET
from sector_rotation.data import download_adjusted_prices, generate_demo_prices
from sector_rotation.holdings import analyze_holding_leadership, fetch_top_holdings
from sector_rotation.metrics import benchmark_returns, drawdown, equity_curve, performance_summary
from sector_rotation.strategy import (
    PERIODS_PER_YEAR,
    BacktestConfig,
    compute_momentum_components,
    run_backtest,
)
from sector_rotation.taiwan import (
    TW_BENCHMARK,
    TW_DEFENSIVE_ASSET,
    TW_ETF_GROUPS,
    TW_THEME_CODES,
    assets_from_official_industries,
    assets_from_taiwan_etfs,
    assets_from_taiwan_themes,
    custom_taiwan_assets,
    fetch_taiwan_company_master,
    official_industry_groups,
)
from sector_rotation.universe import (
    UNIVERSE_GROUPS,
    AssetInfo,
    assets_for,
    custom_assets,
    groups_for,
)


FREQUENCY_SETTINGS = {
    "Daily": {
        "label": "日線",
        "unit": "交易日",
        "lookbacks": {5: 10, 21: 20, 63: 30, 126: 40},
        "volatility_window": 63,
    },
    "Weekly": {
        "label": "週線",
        "unit": "週",
        "lookbacks": {4: 10, 13: 20, 26: 30, 52: 40},
        "volatility_window": 13,
    },
    "Monthly": {
        "label": "月線",
        "unit": "月",
        "lookbacks": {1: 10, 3: 20, 6: 30, 12: 40},
        "volatility_window": 6,
    },
}


st.set_page_config(
    page_title="Multi-Layer Rotation Research Lab",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
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
def load_demo_data(tickers: tuple[str, ...]) -> pd.DataFrame:
    return generate_demo_prices(tickers=list(tickers))


@st.cache_data(ttl=86400, show_spinner=False)
def load_top_holdings(etfs: tuple[str, ...]) -> pd.DataFrame:
    return fetch_top_holdings(list(etfs))


@st.cache_data(ttl=86400, show_spinner=False)
def load_taiwan_company_master() -> pd.DataFrame:
    return fetch_taiwan_company_master()


def parse_tickers(raw: str) -> list[str]:
    return [
        token.upper()
        for token in re.split(r"[\s,;]+", raw.strip())
        if token and re.fullmatch(r"[A-Za-z0-9.^=-]+", token)
    ]


def format_percent(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.1%}"


def format_number(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.2f}"


def performance_chart(
    strategy: pd.Series,
    benchmark: pd.Series,
    benchmark_label: str,
) -> go.Figure:
    frame = pd.concat(
        [
            equity_curve(strategy).rename("Strategy"),
            equity_curve(benchmark).rename(benchmark_label),
        ],
        axis=1,
    ).dropna(how="all")
    figure = px.line(
        frame,
        labels={"value": "Growth of $1", "index": "", "variable": ""},
        color_discrete_sequence=["#38bdf8", "#94a3b8"],
    )
    figure.update_layout(hovermode="x unified", legend_orientation="h", height=430)
    return figure


def allocation_chart(
    weights: pd.DataFrame,
    metadata: dict[str, AssetInfo],
    defensive_ticker: str,
    defensive_label: str,
) -> go.Figure:
    visible = weights.loc[:, (weights.abs().sum() > 0)]
    renamed = visible.rename(
        columns={
            **{ticker: f"{ticker} · {info.name}" for ticker, info in metadata.items()},
            defensive_ticker: defensive_label,
        }
    )
    figure = px.area(
        renamed,
        labels={"value": "Portfolio weight", "index": "", "variable": ""},
    )
    figure.update_layout(hovermode="x unified", legend_orientation="h", height=440)
    figure.update_yaxes(tickformat=".0%", range=[0, 1])
    return figure


st.title("U.S. & Taiwan Multi-Layer Rotation Lab")
st.caption(
    "美股＋台股雙資料庫｜每日／每週／每月動能｜產業、主題、ETF與主要玩家"
)

with st.sidebar:
    st.header("研究範圍")
    market = st.segmented_control(
        "市場資料庫",
        ["美股", "台股"],
        default="美股",
    )
    if market is None:
        market = "美股"

    data_mode = st.radio(
        "資料來源",
        ["Live Yahoo Finance", "Offline demo"],
        index=1 if os.environ.get("SECTOR_ROTATION_TEST_MODE") == "1" else 0,
        help="Offline demo 使用合成資料，只能測試功能。",
    )

    if market == "美股":
        benchmark_ticker = BENCHMARK
        benchmark_label = "SPY"
        defensive_ticker = DEFENSIVE_ASSET
        defensive_label = "SHY · Defensive"
        universe_options = [*UNIVERSE_GROUPS, "Custom tickers — 自訂"]
        universe_name = st.selectbox("研究層級", universe_options, index=1)
        is_etf_universe = "ETFs" in universe_name

        if universe_name == "Custom tickers — 自訂":
            custom_input = st.text_area(
                "輸入 ticker",
                value="NVDA AMD AVGO MSFT AMZN GOOGL",
                help="用空格、逗號或分號分隔。",
            )
            metadata = custom_assets(parse_tickers(custom_input))
            selected_groups = ["Custom"]
        else:
            group_options = groups_for(universe_name)
            if universe_name in {"Broad sectors — ETFs", "Detailed industries — ETFs"}:
                default_groups = group_options
            else:
                default_groups = group_options[:3]
            selected_groups = st.multiselect(
                "產業／主題分類",
                group_options,
                default=default_groups,
            )
            metadata = assets_for(universe_name, selected_groups)
    else:
        benchmark_ticker = TW_BENCHMARK
        benchmark_label = "0050 · 台灣50"
        defensive_ticker = TW_DEFENSIVE_ASSET
        defensive_label = "00679B · 20年美債"
        taiwan_master = load_taiwan_company_master()
        universe_options = [
            "台股官方產業分類",
            "台股主題股票籃子",
            "台股 ETFs",
            "自訂台股代號",
        ]
        universe_name = st.selectbox("研究層級", universe_options, index=1)
        is_etf_universe = universe_name == "台股 ETFs"

        if universe_name == "台股官方產業分類":
            group_options = official_industry_groups(taiwan_master)
            preferred = ["半導體業", "電腦及週邊設備", "電子零組件"]
            selected_groups = st.multiselect(
                "官方產業分類",
                group_options,
                default=[group for group in preferred if group in group_options],
                help="每個產業依已發行股數保留前60家公司，避免一次下載過大。",
            )
            metadata = assets_from_official_industries(
                taiwan_master,
                selected_groups,
            )
        elif universe_name == "台股主題股票籃子":
            group_options = list(TW_THEME_CODES)
            selected_groups = st.multiselect(
                "台股主題",
                group_options,
                default=group_options[:5],
            )
            metadata = assets_from_taiwan_themes(taiwan_master, selected_groups)
        elif universe_name == "台股 ETFs":
            group_options = list(TW_ETF_GROUPS)
            selected_groups = st.multiselect(
                "ETF類型",
                group_options,
                default=group_options[:4],
            )
            metadata = assets_from_taiwan_etfs(selected_groups)
        else:
            custom_input = st.text_area(
                "輸入台股代號",
                value="2330 2454 2317 2308 3711",
                help="直接輸入四位數代號，系統會自動判斷上市.TW或上櫃.TWO。",
            )
            metadata = custom_taiwan_assets(
                parse_tickers(custom_input),
                taiwan_master,
            )
            selected_groups = ["自訂台股"]

    frequency = st.segmented_control(
        "訊號與再平衡頻率",
        ["Daily", "Weekly", "Monthly"],
        default="Weekly",
        format_func=lambda value: FREQUENCY_SETTINGS[value]["label"],
    )
    if frequency is None:
        frequency = "Weekly"
    frequency_settings = FREQUENCY_SETTINGS[frequency]

    latest_allowed_start = date.today() - timedelta(days=400)
    if (
        "research_start_date_v2" in st.session_state
        and st.session_state.research_start_date_v2 > latest_allowed_start
    ):
        st.session_state.research_start_date_v2 = date(2012, 1, 1)
    start_date = st.date_input(
        "回測開始日期",
        value=date(2012, 1, 1),
        max_value=latest_allowed_start,
        key="research_start_date_v2",
    )

    st.header("投資組合規則")
    asset_count = max(1, len(metadata))
    top_n = st.slider("持有排名前 N 個標的", 1, min(15, asset_count), min(5, asset_count))
    weighting = st.selectbox(
        "配置方式",
        ["Equal weight", "Momentum weight", "Inverse volatility"],
    )
    risk_adjusted = st.toggle("使用風險調整動能", value=False)
    positive_filter = st.toggle("只持有正動能標的", value=True)
    use_defensive = st.toggle(
        f"無合格標的時持有 {defensive_ticker}",
        value=True,
    )
    cost_bps = st.number_input(
        "每次換手成本（bps）",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=1.0,
    )

    st.header("多週期動能")
    lookback_weights: dict[int, float] = {}
    for periods, default_weight in frequency_settings["lookbacks"].items():
        lookback_weights[periods] = st.slider(
            f"{periods} {frequency_settings['unit']}",
            min_value=0,
            max_value=100,
            value=default_weight,
            step=5,
        )

if not metadata:
    st.error("請至少選擇一個分類或輸入一個有效 ticker。")
    st.stop()
if sum(lookback_weights.values()) == 0:
    st.error("至少一個動能週期的權重必須大於零。")
    st.stop()

asset_tickers = list(metadata)
download_tickers = tuple(
    dict.fromkeys([*asset_tickers, benchmark_ticker, defensive_ticker])
)

try:
    with st.spinner(
        f"正在下載 {len(download_tickers)} 個標的並執行{frequency_settings['label']}研究…"
    ):
        if data_mode == "Live Yahoo Finance":
            prices = load_live_data(
                download_tickers,
                start_date,
                date.today() + timedelta(days=1),
            )
        else:
            prices = load_demo_data(download_tickers).loc[pd.Timestamp(start_date) :]

        available_assets = [ticker for ticker in asset_tickers if ticker in prices]
        if not available_assets:
            raise RuntimeError(
                "Selected assets have no usable data. Choose Live Yahoo Finance or another universe."
            )

        config = BacktestConfig(
            lookback_weights=lookback_weights,
            frequency=frequency,
            top_n=min(top_n, len(available_assets)),
            weighting=weighting,
            require_positive_momentum=positive_filter,
            risk_adjusted_score=risk_adjusted,
            defensive_asset=defensive_ticker if use_defensive else None,
            transaction_cost_bps=cost_bps,
            volatility_window=frequency_settings["volatility_window"],
        )
        result = run_backtest(prices, available_assets, config)
except Exception as exc:
    st.error(f"Research run failed: {exc}")
    st.info("請確認日期、ticker 與網路；若只是測試介面，可切換 Offline demo。")
    st.stop()

if result.net_returns.empty or result.scores.empty:
    st.warning("資料歷史不足以計算最長動能週期，請把開始日期往前調。")
    st.stop()

periods_per_year = PERIODS_PER_YEAR[frequency]
benchmark = benchmark_returns(
    result.sampled_prices,
    benchmark_ticker,
    result.net_returns.index[0],
)
strategy_metrics = performance_summary(result.net_returns, periods_per_year)
benchmark_metrics = performance_summary(benchmark, periods_per_year)
latest_signal_date = result.scores.index[-1]

components = compute_momentum_components(
    result.sampled_prices,
    available_assets,
    list(lookback_weights),
)
latest_scores = result.scores.iloc[-1].dropna().sort_values(ascending=False)
latest_weights = result.target_weights.iloc[-1]

ranking_rows = []
for ticker, score in latest_scores.items():
    info = metadata[ticker]
    row = {
        "Ticker": ticker,
        "Name": info.name,
        "Group": info.group,
        "Momentum score": score,
        "Target weight": latest_weights.get(ticker, 0.0),
    }
    for periods in lookback_weights:
        row[f"{periods} {frequency_settings['unit']} return"] = components[periods].loc[
            latest_signal_date, ticker
        ]
    ranking_rows.append(row)
ranking = pd.DataFrame(ranking_rows)
ranking.index = range(1, len(ranking) + 1)

if data_mode == "Offline demo":
    st.warning("Offline demo 使用合成資料，不是歷史投資績效。")

coverage_columns = st.columns(5)
coverage_columns[0].metric("市場資料庫", market)
coverage_columns[1].metric("研究標的", len(available_assets))
coverage_columns[2].metric("分類數量", len({metadata[ticker].group for ticker in available_assets}))
coverage_columns[3].metric("頻率", frequency_settings["label"])
coverage_columns[4].metric("資料截止", f"{result.sampled_prices.index[-1]:%Y-%m-%d}")

st.subheader("最新模型訊號")
st.caption(
    f"訊號日期：{latest_signal_date:%Y-%m-%d}。排名會延後一個"
    f"{frequency_settings['unit']}才計入歷史報酬，避免前視偏誤。"
)

allocation = latest_weights[latest_weights > 0].sort_values(ascending=False)
if allocation.empty:
    st.info("目前模型持有現金。")
else:
    card_count = min(6, len(allocation))
    cards = st.columns(card_count)
    for number, (ticker, weight) in enumerate(allocation.items()):
        info = metadata.get(ticker)
        label = info.name if info else "Defensive asset"
        cards[number % card_count].metric(ticker, f"{weight:.1%}", label)

tab_overview, tab_attention, tab_rankings, tab_groups, tab_portfolio, tab_method = st.tabs(
    ["績效", "關注與主要玩家", "詳細排名", "分類比較", "配置歷史", "方法與限制"]
)

with tab_overview:
    metric_columns = st.columns(5)
    metric_columns[0].metric("Strategy CAGR", format_percent(strategy_metrics["CAGR"]))
    metric_columns[1].metric("Strategy Sharpe", format_number(strategy_metrics["Sharpe"]))
    metric_columns[2].metric("Max drawdown", format_percent(strategy_metrics["Max drawdown"]))
    metric_columns[3].metric("Annual volatility", format_percent(strategy_metrics["Volatility"]))
    metric_columns[4].metric(
        "Annual turnover",
        f"{result.turnover.mean() * periods_per_year:.1f}x",
    )

    st.plotly_chart(
        performance_chart(result.net_returns, benchmark, benchmark_label),
        width="stretch",
    )
    left, right = st.columns([2, 1])
    with left:
        drawdowns = pd.concat(
            [
                drawdown(result.net_returns).rename("Strategy"),
                drawdown(benchmark).rename(benchmark_label),
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
        st.plotly_chart(drawdown_figure, width="stretch")
    with right:
        comparison = pd.DataFrame(
            {"Strategy": strategy_metrics, benchmark_label: benchmark_metrics}
        ).astype(object)
        for row in ["CAGR", "Volatility", "Max drawdown", "Win rate"]:
            comparison.loc[row] = comparison.loc[row].map(format_percent)
        comparison.loc["Sharpe"] = comparison.loc["Sharpe"].map(format_number)
        st.dataframe(comparison, width="stretch")

with tab_attention:
    st.subheader("目前優先關注方向")
    st.caption(
        "先用分類平均動能找資金主線，再由領先 ETF 的主要持股找出值得深入研究的公司。"
    )

    attention_groups = (
        ranking.groupby("Group", as_index=False)
        .agg(
            Average_score=("Momentum score", "mean"),
            Best_asset_score=("Momentum score", "max"),
            Positive_assets=("Momentum score", lambda values: int((values > 0).sum())),
            Asset_count=("Ticker", "count"),
        )
        .sort_values("Average_score", ascending=False)
    )
    group_cards = st.columns(min(3, len(attention_groups)))
    for position, row in attention_groups.head(3).reset_index(drop=True).iterrows():
        group_cards[position].metric(
            f"#{position + 1} {row['Group']}",
            f"{row['Average_score']:.3f}",
            f"{int(row['Positive_assets'])}/{int(row['Asset_count'])} positive",
        )

    st.markdown("#### 領先 ETF／股票")
    attention_assets = ranking.head(15).copy()
    st.dataframe(
        attention_assets.style.format(
            {
                "Momentum score": "{:.3f}",
                "Target weight": "{:.1%}",
                **{
                    column: "{:.1%}"
                    for column in attention_assets
                    if column.endswith(f"{frequency_settings['unit']} return")
                },
            }
        ),
        width="stretch",
        height=420,
    )

    if data_mode != "Live Yahoo Finance":
        st.info("切換到 Live Yahoo Finance 才能讀取 ETF 最新主要持股。")
    elif not is_etf_universe:
        st.info(
            "目前選擇的是股票籃子；上表本身就是該主題的股票關注清單。"
            "切換到 ETF 研究層級可繼續拆解 ETF 主要持股。"
        )
    else:
        candidate_etfs = list(latest_scores.head(10).index)
        selected_etfs = st.multiselect(
            "拆解哪些領先 ETF",
            candidate_etfs,
            default=candidate_etfs[: min(3, len(candidate_etfs))],
            help="持股資料快取 24 小時；基金持股可能隨時變動。",
        )
        if selected_etfs:
            with st.spinner("正在讀取 ETF 主要持股並計算股票領先度…"):
                holdings = load_top_holdings(tuple(selected_etfs))
                if holdings.empty:
                    leadership = pd.DataFrame()
                else:
                    holding_tickers = tuple(holdings["Holding ticker"].unique())
                    holding_prices = load_live_data(
                        holding_tickers,
                        date.today() - timedelta(days=400),
                        date.today() + timedelta(days=1),
                    )
                    leadership = analyze_holding_leadership(holdings, holding_prices)

            if leadership.empty:
                st.warning("Yahoo Finance 暫時沒有回傳所選 ETF 的可用持股資料。")
            else:
                leadership["ETF momentum score"] = leadership["ETF"].map(
                    latest_scores.to_dict()
                )
                leadership["ETF group"] = leadership["ETF"].map(
                    {ticker: metadata[ticker].group for ticker in selected_etfs}
                )
                leadership = leadership.sort_values(
                    ["ETF momentum score", "ETF", "ETF rank"],
                    ascending=[False, True, True],
                )

                st.markdown("#### ETF 中的主要帶領玩家")
                leader_display = leadership[
                    [
                        "ETF",
                        "ETF group",
                        "Holding ticker",
                        "Holding name",
                        "Holding weight",
                        "21d return",
                        "63d return",
                        "126d return",
                        "Stock momentum",
                        "Leadership score",
                        "ETF rank",
                    ]
                ]
                st.dataframe(
                    leader_display.style.format(
                        {
                            "Holding weight": "{:.1%}",
                            "21d return": "{:.1%}",
                            "63d return": "{:.1%}",
                            "126d return": "{:.1%}",
                            "Stock momentum": "{:.1%}",
                            "Leadership score": "{:.4f}",
                        }
                    ),
                    width="stretch",
                    height=560,
                )

                chart_data = leadership.copy()
                chart_data["Player"] = (
                    chart_data["Holding ticker"] + " · " + chart_data["Holding name"]
                )
                leader_chart = px.bar(
                    chart_data,
                    x="Leadership score",
                    y="Player",
                    color="ETF",
                    facet_col="ETF",
                    facet_col_wrap=min(3, len(selected_etfs)),
                    orientation="h",
                    title="持股權重 × 股票多週期動能（研究優先度代理值）",
                )
                leader_chart.update_layout(
                    height=max(480, int(len(chart_data) / max(1, len(selected_etfs))) * 35)
                )
                st.plotly_chart(leader_chart, width="stretch")

                st.download_button(
                    "下載 ETF 主要玩家 CSV",
                    data=leader_display.to_csv(index=False).encode("utf-8"),
                    file_name=f"etf-leading-players-{latest_signal_date:%Y-%m-%d}.csv",
                    mime="text/csv",
                )

with tab_rankings:
    return_columns = [
        column for column in ranking if column.endswith(f"{frequency_settings['unit']} return")
    ]
    formatters = {
        "Momentum score": "{:.3f}",
        "Target weight": "{:.1%}",
        **{column: "{:.1%}" for column in return_columns},
    }
    st.dataframe(ranking.style.format(formatters), width="stretch", height=520)
    st.download_button(
        "下載最新排名 CSV",
        data=ranking.to_csv(index=False).encode("utf-8"),
        file_name=f"rotation-ranking-{latest_signal_date:%Y-%m-%d}.csv",
        mime="text/csv",
    )

    history = result.scores.rename(
        columns={ticker: f"{ticker} · {metadata[ticker].name}" for ticker in available_assets}
    )
    history_periods = {"Daily": 126, "Weekly": 52, "Monthly": 36}[frequency]
    st.plotly_chart(
        px.line(
            history.tail(history_periods),
            labels={"value": "Momentum score", "index": "", "variable": ""},
            title=f"近期動能領先變化（{history_periods} 個{frequency_settings['unit']}）",
        ).update_layout(hovermode="x unified", height=470),
        width="stretch",
    )

with tab_groups:
    group_scores = (
        ranking.groupby("Group", as_index=False)
        .agg(
            Average_score=("Momentum score", "mean"),
            Best_score=("Momentum score", "max"),
            Positive_assets=("Momentum score", lambda values: int((values > 0).sum())),
            Asset_count=("Ticker", "count"),
        )
        .sort_values("Average_score", ascending=True)
    )
    group_figure = px.bar(
        group_scores,
        x="Average_score",
        y="Group",
        orientation="h",
        color="Average_score",
        color_continuous_scale="RdYlGn",
        title="分類平均動能",
    )
    group_figure.update_layout(height=max(360, len(group_scores) * 48))
    st.plotly_chart(group_figure, width="stretch")

    heatmap_data = ranking.set_index("Ticker")[return_columns]
    heatmap_data.columns = [column.replace(" return", "") for column in heatmap_data.columns]
    heatmap = px.imshow(
        heatmap_data,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        text_auto=".1%",
        title="每個標的的多週期報酬",
    )
    heatmap.update_layout(height=max(420, len(heatmap_data) * 27))
    st.plotly_chart(heatmap, width="stretch")

with tab_portfolio:
    st.plotly_chart(
        allocation_chart(
            result.deployed_weights,
            metadata,
            defensive_ticker,
            defensive_label,
        ),
        width="stretch",
    )
    recent_rows = {"Daily": 30, "Weekly": 26, "Monthly": 12}[frequency]
    recent = result.deployed_weights.tail(recent_rows).rename(
        columns={
            **{ticker: f"{ticker} · {metadata[ticker].name}" for ticker in available_assets},
            defensive_ticker: defensive_label,
        }
    )
    recent = recent.loc[:, (recent.abs().sum() > 0)]
    st.dataframe(recent.style.format("{:.1%}"), width="stretch")
    st.download_button(
        "下載配置歷史 CSV",
        data=result.deployed_weights.to_csv().encode("utf-8"),
        file_name=f"rotation-weights-{frequency.lower()}.csv",
        mime="text/csv",
    )

with tab_method:
    st.markdown(
        f"""
        ### 本次研究設定

        - 市場資料庫：**{market}**
        - 研究層級：**{universe_name}**
        - 分類：**{", ".join(selected_groups)}**
        - 訊號與再平衡：**{frequency_settings["label"]}**
        - 標的數量：**{len(available_assets)}**
        - 持有數量：**Top {min(top_n, len(available_assets))}**
        - 配置方法：**{weighting}**
        - 風險調整：**{"開啟" if risk_adjusted else "關閉"}**
        - 正動能過濾：**{"開啟" if positive_filter else "關閉"}**
        - 防禦資產：**{defensive_ticker if use_defensive else "現金"}**
        - 換手成本：**{cost_bps:.0f} bps**

        ### 計算流程

        1. 將調整後日價格轉換成指定的日線、週線或月線資料。
        2. 計算 {", ".join(f"{periods} {frequency_settings['unit']}" for periods in lookback_weights)} 報酬。
        3. 正規化權重並組合成一個 momentum score。
        4. 可選擇把分數除以同期年化波動率。
        5. 通過正動能篩選後，選擇排名前 N 個標的。
        6. 依照指定方式配置，並把訊號延後一期才計算報酬。
        7. 根據每期換手率扣除交易成本。

        ### 重要限制

        主題股票籃子是研究分類，不是官方指數，也不代表完整產業成分。
        日線輪動對交易成本、滑價和訊號雜訊特別敏感。Yahoo Finance 資料可能延遲、
        修正或缺失。本工具未完整處理稅務、市場衝擊、退市偏誤和實際成交限制。
        回測不保證未來績效，畫面內容不是個人化投資建議。
        """
    )
