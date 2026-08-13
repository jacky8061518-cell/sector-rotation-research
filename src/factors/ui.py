"""Integrated Streamlit research workbench for registered factors."""

from __future__ import annotations

from typing import cast

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .backtest import BacktestResult, run_factor_backtest
from .config import EvaluationConfig, PipelineConfig, StandardizeMethod, WinsorMethod
from .context import DataContext, FactorDataStore
from .costs import CostConfig
from .evaluation import (
    EvaluationResult,
    evaluate_factor,
    evaluate_quantiles,
    factor_correlation,
    grouped_rank_ic,
)
from .pipeline import preprocess_factor
from .portfolio import PortfolioConfig, WeightMethod
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
        universe.loc[len(universe)] = [benchmark, "TW", "市場基準", False]
    return universe


def _signal_dates(index: pd.DatetimeIndex, observations: int = 24) -> tuple[pd.Timestamp, ...]:
    sessions = pd.Series(index=index, data=index)
    month_ends = sessions.groupby(index.to_period("M")).max()
    return tuple(pd.Timestamp(item) for item in month_ends.iloc[-observations:])


def _build_store(
    adjusted_close: pd.DataFrame,
    master: pd.DataFrame,
    benchmark: str,
    inst_flow: pd.DataFrame,
    revenue: pd.DataFrame,
    market_cap: pd.DataFrame,
    financials: pd.DataFrame,
) -> FactorDataStore:
    return FactorDataStore(
        pd.DataFrame(columns=["date", "ticker", "adj_close"]),
        _universe_from_master(master, benchmark),
        market_cap_data=market_cap,
        inst_flow_data=inst_flow,
        revenue_data=revenue,
        financial_data=financials,
        adjusted_close_wide=adjusted_close,
    )


def _available_specs(store: FactorDataStore, asof: pd.Timestamp | None = None) -> tuple:
    timestamp = pd.Timestamp(asof) if asof is not None else None

    def visible(frame: pd.DataFrame) -> bool:
        if frame.empty:
            return False
        if timestamp is None or "published_at" not in frame:
            return True
        return bool(pd.to_datetime(frame["published_at"], errors="coerce").le(timestamp).any())

    specs = []
    for spec in list_factors(market="TW"):
        if not store.supports(spec.requires):
            continue
        if any(item.startswith("revenue") for item in spec.requires) and not visible(store.revenue_data):
            continue
        if any(item.startswith("financials") for item in spec.requires) and not visible(store.financial_data):
            continue
        specs.append(spec)
    return tuple(specs)


@st.cache_data(show_spinner=False)
def _evaluate_cached(
    factor_name: str,
    data_version: str,
    _adjusted_close: pd.DataFrame,
    _master: pd.DataFrame,
    _inst_flow: pd.DataFrame,
    _revenue: pd.DataFrame,
    _market_cap: pd.DataFrame,
    _financials: pd.DataFrame,
    benchmark: str,
    minimum_coverage: float,
    winsor_method: WinsorMethod,
    standardize_method: StandardizeMethod,
    observations: int = 24,
) -> EvaluationResult:
    del data_version
    store = _build_store(
        _adjusted_close,
        _master,
        benchmark,
        _inst_flow,
        _revenue,
        _market_cap,
        _financials,
    )
    config = EvaluationConfig(
        pipeline=PipelineConfig(
            minimum_coverage=minimum_coverage,
            winsor_method=winsor_method,
            standardize_method=standardize_method,
        )
    )
    return evaluate_factor(
        get_factor(factor_name),
        lambda asof: DataContext(store, "TW", asof, benchmark),
        _adjusted_close,
        _signal_dates(pd.DatetimeIndex(_adjusted_close.index), observations),
        config,
    )


def _evaluation(
    factor_name: str,
    adjusted_close: pd.DataFrame,
    master: pd.DataFrame,
    sources: dict[str, pd.DataFrame],
    benchmark: str,
    minimum_coverage: float = 0.60,
    winsor_method: WinsorMethod = "quantile",
    standardize_method: StandardizeMethod = "zscore",
    observations: int = 24,
) -> EvaluationResult:
    version = f"{adjusted_close.index.max()}:{adjusted_close.shape}:" + ":".join(
        f"{name}={len(frame)}" for name, frame in sources.items()
    )
    return _evaluate_cached(
        factor_name,
        version,
        adjusted_close,
        master,
        sources["inst_flow"],
        sources["revenue"],
        sources["market_cap"],
        sources["financials"],
        benchmark,
        minimum_coverage,
        winsor_method,
        standardize_method,
        observations,
    )


def _summary_frame(evaluation: EvaluationResult) -> pd.DataFrame:
    return pd.DataFrame(
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


def _render_explorer(
    adjusted_close: pd.DataFrame,
    master: pd.DataFrame,
    sources: dict[str, pd.DataFrame],
    store: FactorDataStore,
    benchmark: str,
) -> None:
    st.subheader("Factor Explorer · 單因子診斷")
    st.caption("Point-in-time 資料截斷｜月頻觀察｜Rank IC｜分位數單調性｜5／20／60 日衰減")
    specs = _available_specs(store, adjusted_close.index.max())
    labels = {f"{spec.label} · {spec.name}": spec.name for spec in specs}
    controls = st.columns([2, 1, 1, 1])
    options = list(labels)
    default = next((i for i, label in enumerate(options) if labels[label] == "mom_12_1"), 0)
    selected = controls[0].selectbox("因子", options, index=default)
    coverage = controls[1].slider("最低覆蓋率", 0.20, 1.00, 0.60, 0.05)
    winsor_label = controls[2].selectbox("極值處理", ["1% / 99%", "MAD", "關閉"])
    standardize_label = controls[3].selectbox("標準化", ["Z-score", "Rank normal", "關閉"])
    winsor = cast(WinsorMethod, {"1% / 99%": "quantile", "MAD": "mad", "關閉": "none"}[winsor_label])
    standardize = cast(
        StandardizeMethod, {"Z-score": "zscore", "Rank normal": "rank_normal", "關閉": "none"}[standardize_label]
    )
    factor_name = labels[selected]
    st.caption(get_factor(factor_name).spec.description)
    evaluation = _evaluation(factor_name, adjusted_close, master, sources, benchmark, float(coverage), winsor, standardize)
    st.dataframe(
        _summary_frame(evaluation).style.format(
            {"IC 平均": "{:.3f}", "IC 標準差": "{:.3f}", "IC IR": "{:.2f}", "Newey-West t": "{:.2f}", "正 IC 比率": "{:.1%}"}
        ),
        hide_index=True,
        width="stretch",
    )
    if any(summary.suspicious for summary in evaluation.summaries):
        st.error("IC IR 高於 1.0，已標記為疑似資料或對齊問題，不直接視為有效成果。")
    quantiles = evaluate_quantiles(evaluation.scores, evaluation.forward_returns)
    chart_columns = st.columns(2)
    if not evaluation.ic.empty:
        ic_long = evaluation.ic.rename(columns={f"forward_{h}d": f"未來 {h} 日" for h in (5, 20, 60)}).melt(
            ignore_index=False, var_name="期間", value_name="IC"
        )
        figure = px.line(ic_long.reset_index(), x="date", y="IC", color="期間", color_discrete_sequence=PALETTE)
        figure.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
        figure.update_layout(height=370, title="IC 時序")
        chart_columns[0].plotly_chart(figure, width="stretch")
    if not quantiles.cumulative_returns.empty:
        q_long = quantiles.cumulative_returns.reset_index().melt(id_vars="date", var_name="分位", value_name="累積報酬")
        q_figure = px.line(q_long, x="date", y="累積報酬", color="分位", color_discrete_sequence=PALETTE)
        q_figure.update_layout(
            height=370, title=f"五分位未來 20 日累積報酬｜單調性 {quantiles.monotonicity:.2f}", yaxis_tickformat=".0%"
        )
        chart_columns[1].plotly_chart(q_figure, width="stretch")
    diagnostics = st.columns(3)
    coverage_figure = go.Figure(go.Scatter(x=evaluation.coverage.index, y=evaluation.coverage, mode="lines+markers"))
    coverage_figure.add_hline(y=coverage, line_dash="dot", line_color="#fb7185")
    coverage_figure.update_layout(height=330, title="覆蓋率時序", yaxis_tickformat=".0%")
    diagnostics[0].plotly_chart(coverage_figure, width="stretch")
    if evaluation.scores.empty:
        st.info("有效覆蓋率低於門檻，沒有可顯示的因子橫斷面。")
        return
    latest = evaluation.scores.iloc[-1].dropna().rename("score")
    diagnostics[1].plotly_chart(
        px.histogram(latest.to_frame(), x="score", nbins=40, title="最新因子值分布"), width="stretch"
    )
    industries = master.drop_duplicates("Yahoo ticker").set_index("Yahoo ticker")["Industry"].reindex(latest.index)
    exposure = (
        pd.concat([latest, industries.rename("industry")], axis=1)
        .dropna()
        .groupby("industry")["score"]
        .mean()
        .sort_values()
        .tail(15)
    )
    diagnostics[2].plotly_chart(
        px.bar(exposure.rename("平均分數").reset_index(), x="平均分數", y="industry", orientation="h", title="產業暴露"),
        width="stretch",
    )
    industry_ic = grouped_rank_ic(evaluation.scores, evaluation.forward_returns, industries)
    if not industry_ic.empty:
        stability = industry_ic.agg(["mean", "count"]).T.rename(columns={"mean": "20D IC 平均", "count": "觀察期數"})
        stability = stability[stability["觀察期數"].ge(3)].sort_values("20D IC 平均", ascending=False)
        with st.expander("產業穩定性切割（使用最新產業分類，已知限制）"):
            st.dataframe(stability.style.format({"20D IC 平均": "{:.3f}", "觀察期數": "{:.0f}"}), width="stretch")


def _render_zoo(
    adjusted_close: pd.DataFrame,
    master: pd.DataFrame,
    sources: dict[str, pd.DataFrame],
    store: FactorDataStore,
    benchmark: str,
) -> None:
    st.subheader("Factor Zoo · 因子總覽")
    specs = _available_specs(store, adjusted_close.index.max())
    comparison_defaults = [
        name
        for name in (
            "mom_12_1",
            "mom_6m",
            "vol_60d",
            "max_ret_5d",
            "flow_foreign_persist",
        )
        if any(spec.name == name for spec in specs)
    ]
    selected = st.multiselect("納入比較", [spec.name for spec in specs], default=comparison_defaults)
    if not selected:
        st.info("請至少選一個有資料支援的因子。")
        return
    if not st.button("執行因子比較", type="primary"):
        st.info("選擇因子後執行比較；結果會快取，切換頁面不需重算。")
        return
    results = {name: _evaluation(name, adjusted_close, master, sources, benchmark, observations=18) for name in selected}
    rows = []
    for name, result in results.items():
        summary = result.summaries[1]
        rows.append(
            {
                "因子": name,
                "20D IC": summary.mean,
                "IC IR": summary.information_ratio,
                "NW t": summary.newey_west_t,
                "正 IC 比率": summary.positive_ratio,
                "覆蓋率": result.coverage.mean(),
                "警示": "疑似異常" if summary.suspicious else "",
            }
        )
    ranking = pd.DataFrame(rows).sort_values("IC IR", ascending=False)
    st.dataframe(
        ranking.style.format(
            {"20D IC": "{:.3f}", "IC IR": "{:.2f}", "NW t": "{:.2f}", "正 IC 比率": "{:.1%}", "覆蓋率": "{:.1%}"}
        ),
        hide_index=True,
        width="stretch",
    )
    correlation = factor_correlation({name: result.scores for name, result in results.items()})
    if not correlation.empty:
        heatmaps = st.columns(2)
        heatmaps[0].plotly_chart(
            px.imshow(correlation, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f", title="因子相關矩陣"),
            width="stretch",
        )
        ic_correlation = pd.DataFrame({name: result.ic.get("forward_20d") for name, result in results.items()}).corr()
        heatmaps[1].plotly_chart(
            px.imshow(
                ic_correlation, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f", title="IC 相關矩陣"
            ),
            width="stretch",
        )


def _render_portfolio(
    adjusted_close: pd.DataFrame,
    master: pd.DataFrame,
    sources: dict[str, pd.DataFrame],
    store: FactorDataStore,
    benchmark: str,
) -> None:
    st.subheader("Portfolio Builder · 可交易時序回測")
    specs = _available_specs(store, adjusted_close.index.max())
    labels = {spec.label: spec.name for spec in specs}
    controls = st.columns(4)
    label_options = list(labels)
    default_index = next(
        (index for index, label_name in enumerate(label_options) if labels[label_name] == "mom_12_1"),
        0,
    )
    label = controls[0].selectbox("因子", label_options, index=default_index)
    long_short = controls[1].toggle("多空組合", value=False)
    method_label = controls[2].selectbox("權重", ["等權", "分數加權", "逆波動"])
    minimum_holdings = controls[3].number_input("每側最少持股", 5, 100, 20)
    factor_name = labels[label]
    evaluation = _evaluation(factor_name, adjusted_close, master, sources, benchmark, observations=36)
    method = cast(WeightMethod, {"等權": "equal", "分數加權": "score", "逆波動": "inverse_volatility"}[method_label])
    config = PortfolioConfig(long_short=long_short, weight_method=method, minimum_holdings=int(minimum_holdings))
    industry = store.universe_data.drop_duplicates("ticker").set_index("ticker")["industry"]
    result: BacktestResult = run_factor_backtest(
        evaluation.scores, adjusted_close, "TW", config, CostConfig(), industry=industry
    )
    if result.holdings.empty:
        st.info("目前樣本不足以滿足持股限制。")
        return
    metrics = result.metrics.rename(index={"gross": "未扣成本", "net": "扣除成本"}).reset_index(names="版本")
    st.dataframe(
        metrics.style.format(
            {column: "{:.2%}" for column in ["annual_return", "annual_volatility", "max_drawdown", "monthly_win_rate"]}
        ).format({"sharpe": "{:.2f}"}),
        hide_index=True,
        width="stretch",
    )
    equity = result.equity.reset_index().melt(id_vars="date", var_name="版本", value_name="淨值")
    st.plotly_chart(
        px.line(equity, x="date", y="淨值", color="版本", title="訊號後下一交易日收盤成交｜含／不含成本"), width="stretch"
    )
    downloads = st.columns(3)
    downloads[0].download_button("下載逐日報酬", result.returns.to_csv().encode(), "factor-returns.csv", "text/csv")
    downloads[1].download_button(
        "下載逐期持股", result.holdings.to_csv(index=False).encode(), "factor-holdings.csv", "text/csv"
    )
    downloads[2].download_button("下載成本明細", result.costs.to_csv(index=False).encode(), "factor-costs.csv", "text/csv")


def _render_snapshot(
    adjusted_close: pd.DataFrame,
    master: pd.DataFrame,
    store: FactorDataStore,
    benchmark: str,
) -> None:
    st.subheader("Cross-section Snapshot · 最新橫斷面")
    specs = _available_specs(store, adjusted_close.index.max())
    defaults = [name for name in ("mom_12_1", "vol_60d", "flow_foreign_persist") if any(spec.name == name for spec in specs)]
    selected = st.multiselect("合成因子", [spec.name for spec in specs], default=defaults)
    if not selected:
        st.info("請至少選一個因子。")
        return
    asof = pd.Timestamp(adjusted_close.index.max())
    ctx = DataContext(store, "TW", asof, benchmark)
    panels = []
    for name in selected:
        factor = get_factor(name)
        processed = preprocess_factor(
            factor.compute(ctx, asof), factor.spec, ctx.universe(), PipelineConfig(), industry=ctx.industry_map()
        )
        panels.append(processed.values.rename(name))
    scores = pd.concat(panels, axis=1)
    minimum_factors = max(1, int(len(selected) * 0.60 + 0.999))
    scores["綜合分數"] = scores.mean(axis=1).where(scores.notna().sum(axis=1).ge(minimum_factors))
    metadata = master.drop_duplicates("Yahoo ticker").set_index("Yahoo ticker")[["Name", "Market", "Industry"]]
    snapshot = metadata.join(scores).sort_values("綜合分數", ascending=False).reset_index(names="Ticker")
    industries = ["全部", *sorted(snapshot["Industry"].dropna().unique())]
    selected_industry = st.selectbox("產業篩選", industries)
    shown = snapshot if selected_industry == "全部" else snapshot[snapshot["Industry"].eq(selected_industry)]
    st.dataframe(shown, hide_index=True, width="stretch", height=620)
    st.download_button(
        "下載最新因子排名", shown.to_csv(index=False).encode("utf-8-sig"), f"factor-snapshot-{asof:%Y-%m-%d}.csv", "text/csv"
    )


def render_factor_lab(
    adjusted_close: pd.DataFrame,
    master: pd.DataFrame,
    *,
    benchmark: str = "0050.TW",
    inst_flow: pd.DataFrame | None = None,
    revenue: pd.DataFrame | None = None,
    market_cap: pd.DataFrame | None = None,
    financials: pd.DataFrame | None = None,
) -> None:
    """Render one integrated factor product inside the existing Streamlit app."""
    st.header("Factor Research Lab · 因子研究實驗室")
    st.warning("歷史股票池與產業分類目前仍含存活者偏誤及最新分類偏誤；正式結果不隱藏此限制。")
    if adjusted_close.empty:
        st.error("價格資料庫為空，無法執行因子研究。")
        return
    sources = {
        "inst_flow": inst_flow if inst_flow is not None else pd.DataFrame(),
        "revenue": revenue if revenue is not None else pd.DataFrame(),
        "market_cap": market_cap if market_cap is not None else pd.DataFrame(),
        "financials": financials if financials is not None else pd.DataFrame(),
    }
    store = _build_store(adjusted_close, master, benchmark, *sources.values())
    all_specs = list_factors(market="TW")
    available = _available_specs(store, adjusted_close.index.max())
    st.caption(f"已註冊 {len(all_specs)} 個因子｜目前資料可正式運算 {len(available)} 個｜未具 point-in-time 資料者自動停用")
    page = st.segmented_control(
        "因子工具", ["Explorer", "Factor Zoo", "Portfolio Builder", "最新橫斷面"], default="Explorer"
    )
    if page == "Factor Zoo":
        _render_zoo(adjusted_close, master, sources, store, benchmark)
    elif page == "Portfolio Builder":
        _render_portfolio(adjusted_close, master, sources, store, benchmark)
    elif page == "最新橫斷面":
        _render_snapshot(adjusted_close, master, store, benchmark)
    else:
        _render_explorer(adjusted_close, master, sources, store, benchmark)


def render_factor_explorer(adjusted_close: pd.DataFrame, master: pd.DataFrame, *, benchmark: str = "0050.TW") -> None:
    """Backward-compatible Phase 1 entry point."""
    render_factor_lab(adjusted_close, master, benchmark=benchmark)
