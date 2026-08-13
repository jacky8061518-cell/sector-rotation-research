"""Streamlit Factor Explorer for Phase 1 single-factor diagnostics."""

from __future__ import annotations

from typing import cast

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .config import (
    EvaluationConfig,
    PipelineConfig,
    StandardizeMethod,
    WinsorMethod,
)
from .context import DataContext, FactorDataStore
from .evaluation import EvaluationResult, evaluate_factor
from .registry import get_factor, list_factors

PALETTE = ["#38bdf8", "#f59e0b", "#22c55e", "#fb7185", "#a78bfa"]


def _universe_from_master(master: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    stocks = master[master["Asset type"].eq("股票")].copy()
    universe = pd.DataFrame(
        {
            "ticker": stocks["Yahoo ticker"].astype(str),
            "market": "TW",
            "industry": stocks["Industry"].astype("string"),
            "eligible": True,
        }
    )
    if benchmark not in set(universe["ticker"]):
        universe = pd.concat(
            [
                universe,
                pd.DataFrame(
                    {
                        "ticker": [benchmark],
                        "market": ["TW"],
                        "industry": ["市場基準"],
                        "eligible": [False],
                    }
                ),
            ],
            ignore_index=True,
        )
    return universe


def _signal_dates(index: pd.DatetimeIndex, observations: int = 24) -> tuple[pd.Timestamp, ...]:
    sessions = pd.Series(index=index, data=index)
    month_ends = sessions.groupby(index.to_period("M")).max()
    return tuple(pd.Timestamp(item) for item in month_ends.iloc[-observations:])


@st.cache_data(show_spinner=False)
def _evaluate_cached(
    factor_name: str,
    data_version: str,
    _adjusted_close: pd.DataFrame,
    _master: pd.DataFrame,
    benchmark: str,
    minimum_coverage: float,
    winsor_method: WinsorMethod,
    standardize_method: StandardizeMethod,
) -> EvaluationResult:
    del data_version
    universe = _universe_from_master(_master, benchmark)
    store = FactorDataStore(
        pd.DataFrame(columns=["date", "ticker", "adj_close"]),
        universe,
        adjusted_close_wide=_adjusted_close,
    )
    config = EvaluationConfig(
        pipeline=PipelineConfig(
            minimum_coverage=minimum_coverage,
            winsor_method=winsor_method,
            standardize_method=standardize_method,
        )
    )

    def context_factory(asof: pd.Timestamp) -> DataContext:
        return DataContext(store, "TW", asof, benchmark)

    return evaluate_factor(
        get_factor(factor_name),
        context_factory,
        _adjusted_close,
        _signal_dates(pd.DatetimeIndex(_adjusted_close.index)),
        config,
    )


def render_factor_explorer(
    adjusted_close: pd.DataFrame,
    master: pd.DataFrame,
    *,
    benchmark: str = "0050.TW",
) -> None:
    """Render Phase 1 diagnostics without importing concrete factor classes."""
    st.subheader("Factor Explorer · 單因子診斷")
    st.caption("Point-in-time 資料截斷｜月頻觀察｜Spearman Rank IC｜5／20／60 日衰減")
    st.warning(
        "目前股票池與產業分類使用現存官方清單，歷史結果仍有存活者偏誤與最新分類偏誤；"
        "此限制不會被回測績效掩蓋。"
    )
    if adjusted_close.empty:
        st.error("價格資料庫為空，無法執行因子診斷。")
        return

    specs = list_factors(market="TW")
    labels = {f"{spec.label} · {spec.name}": spec.name for spec in specs}
    controls = st.columns([2, 1, 1, 1])
    label_options = list(labels)
    default_index = next(
        (index for index, label in enumerate(label_options) if labels[label] == "mom_12_1"),
        0,
    )
    selected_label = controls[0].selectbox("因子", label_options, index=default_index)
    minimum_coverage = controls[1].slider(
        "最低覆蓋率", min_value=0.20, max_value=1.00, value=0.60, step=0.05
    )
    winsor_label = controls[2].selectbox("極值處理", ["1% / 99%", "MAD", "關閉"])
    standardize_label = controls[3].selectbox("標準化", ["Z-score", "Rank normal", "關閉"])
    winsor_method = cast(
        WinsorMethod,
        {"1% / 99%": "quantile", "MAD": "mad", "關閉": "none"}[winsor_label],
    )
    standardize_method = cast(
        StandardizeMethod,
        {"Z-score": "zscore", "Rank normal": "rank_normal", "關閉": "none"}[standardize_label],
    )
    factor_name = labels[selected_label]
    factor = get_factor(factor_name)
    st.caption(factor.spec.description)

    with st.spinner("正在建立 point-in-time 因子橫斷面與 IC…"):
        evaluation = _evaluate_cached(
            factor_name,
            f"{adjusted_close.index.max()}:{adjusted_close.shape}",
            adjusted_close,
            master,
            benchmark,
            float(minimum_coverage),
            winsor_method,
            standardize_method,
        )

    summary_frame = pd.DataFrame(
        [
            {
                "期間": f"{summary.horizon}D",
                "IC 平均": summary.mean,
                "IC 標準差": summary.standard_deviation,
                "IC IR": summary.information_ratio,
                "Newey-West t": summary.newey_west_t,
                "正 IC 比率": summary.positive_ratio,
                "樣本數": summary.observations,
                "資料警示": "疑似異常" if summary.suspicious else "",
            }
            for summary in evaluation.summaries
        ]
    )
    st.dataframe(
        summary_frame.style.format(
            {
                "IC 平均": "{:.3f}",
                "IC 標準差": "{:.3f}",
                "IC IR": "{:.2f}",
                "Newey-West t": "{:.2f}",
                "正 IC 比率": "{:.1%}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    if any(summary.suspicious for summary in evaluation.summaries):
        st.error("IC IR 高於 1.0，已標記為疑似資料或對齊問題；不得直接視為有效成果。")

    chart_columns = st.columns(2)
    if not evaluation.ic.empty:
        ic_long = evaluation.ic.rename(
            columns={f"forward_{h}d": f"未來 {h} 日" for h in (5, 20, 60)}
        ).melt(ignore_index=False, var_name="期間", value_name="IC")
        ic_figure = px.line(
            ic_long.reset_index(),
            x="date",
            y="IC",
            color="期間",
            color_discrete_sequence=PALETTE,
        )
        ic_figure.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
        ic_figure.update_layout(height=390, title="IC 時序")
        chart_columns[0].plotly_chart(ic_figure, width="stretch")

    coverage_figure = go.Figure(
        go.Scatter(
            x=evaluation.coverage.index,
            y=evaluation.coverage,
            mode="lines+markers",
            line={"color": PALETTE[1]},
            hovertemplate="%{x|%Y-%m-%d}<br>覆蓋率 %{y:.1%}<extra></extra>",
        )
    )
    coverage_figure.add_hline(y=minimum_coverage, line_dash="dot", line_color="#fb7185")
    coverage_figure.update_layout(height=390, title="覆蓋率時序", yaxis_tickformat=".0%")
    chart_columns[1].plotly_chart(coverage_figure, width="stretch")

    if evaluation.scores.empty:
        st.info("有效覆蓋率低於門檻，沒有可顯示的因子橫斷面。")
        return
    latest_date = pd.Timestamp(evaluation.scores.index[-1])
    latest = evaluation.scores.iloc[-1].dropna().rename("score")
    diagnostics_columns = st.columns(2)
    distribution = px.histogram(
        latest.rename("score").to_frame(),
        x="score",
        nbins=40,
        color_discrete_sequence=[PALETTE[0]],
        title=f"因子值分布 · {latest_date:%Y-%m-%d}",
    )
    diagnostics_columns[0].plotly_chart(distribution, width="stretch")

    industries = (
        master.drop_duplicates("Yahoo ticker", keep="last")
        .set_index("Yahoo ticker")["Industry"]
        .reindex(latest.index)
        .rename("industry")
    )
    exposure = (
        pd.concat([latest.rename("score"), industries], axis=1)
        .dropna()
        .groupby("industry")["score"]
        .mean()
        .sort_values()
        .tail(15)
    )
    industry_figure = px.bar(
        exposure.rename("平均分數").reset_index(),
        x="平均分數",
        y="industry",
        orientation="h",
        color="平均分數",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        title="產業平均因子分數（前 15）",
    )
    diagnostics_columns[1].plotly_chart(industry_figure, width="stretch")
