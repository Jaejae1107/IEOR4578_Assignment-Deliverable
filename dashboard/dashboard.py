"""
FortuneTellers - Model / Cluster Comparison Dashboard (Streamlit page).

This module is meant to be imported by ``app.py`` *and* runnable on its own::

    streamlit run dashboard/dashboard.py

It exposes:

- ``render(page_title)`` - the full Model / Cluster comparison page.
- ``inject_styles()``    - shared CSS theme, also called from app.py.
- ``hero_header(...)``   - gradient hero banner used at the top of every page.
- ``apply_chart_theme(fig)`` / ``add_forecast_connector(fig, history, forecast)``
  - shared Plotly helpers so the chart styling is consistent across pages.
- ``style_dataframe(df)`` - formatter used for highlighted comparison tables.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_THIS_FILE = Path(__file__).resolve()
_DASHBOARD_DIR = _THIS_FILE.parent
_PROJECT_ROOT = _DASHBOARD_DIR.parent
for p in (_DASHBOARD_DIR, _PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import agent_tools  # noqa: E402


# ===========================================================================
# Shared theme - colors, CSS and chart styling reused by every page.
# ===========================================================================
THEME = {
    "primary": "#2563eb",       # deep blue (actuals)
    "primary_dark": "#1e40af",
    "accent": "#f97316",        # warm orange (forecast)
    "bridge": "#94a3b8",        # slate gray (connector / no-sale weeks)
    "good": "#10b981",
    "warn": "#f59e0b",
    "danger": "#ef4444",
    "text": "#0f172a",
    "muted": "#64748b",
    "border": "#e2e8f0",
    "card_bg": "#ffffff",
    "page_bg": "#f8fafc",
}


_GLOBAL_CSS = f"""
<style>
/* ---------- Layout ---------- */
.main .block-container {{
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1380px;
}}
[data-testid="stHeader"] {{ background: transparent; }}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {{
    background-color: {THEME['page_bg']};
    border-right: 1px solid {THEME['border']};
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: {THEME['text']};
}}

/* ---------- Hero banner ---------- */
.ft-hero {{
    background: linear-gradient(135deg, {THEME['primary']} 0%, {THEME['primary_dark']} 100%);
    color: white;
    padding: 26px 30px;
    border-radius: 16px;
    margin-bottom: 22px;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.18);
}}
.ft-hero h1 {{
    color: white;
    margin: 0;
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}}
.ft-hero p {{
    color: rgba(255,255,255,0.88);
    margin: 6px 0 0 0;
    font-size: 1rem;
}}
.ft-hero .ft-hero-pill {{
    display: inline-block;
    background: rgba(255,255,255,0.18);
    color: white;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 8px;
}}

/* ---------- Section header ---------- */
.ft-section {{
    font-size: 1.25rem;
    font-weight: 700;
    color: {THEME['text']};
    margin-top: 1.6rem;
    margin-bottom: 0.6rem;
    padding-bottom: 0.45rem;
    border-bottom: 2px solid {THEME['border']};
    display: flex;
    align-items: baseline;
    gap: 10px;
}}
.ft-section .ft-section-num {{
    background: {THEME['primary']};
    color: white;
    border-radius: 8px;
    width: 28px;
    height: 28px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    font-weight: 700;
}}
.ft-section .ft-section-sub {{
    color: {THEME['muted']};
    font-size: 0.9rem;
    font-weight: 500;
    margin-left: auto;
}}

/* ---------- Metric cards ---------- */
[data-testid="stMetric"] {{
    background-color: {THEME['card_bg']};
    border: 1px solid {THEME['border']};
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    transition: box-shadow 120ms ease, transform 120ms ease;
}}
[data-testid="stMetric"]:hover {{
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
    transform: translateY(-1px);
}}
[data-testid="stMetric"] label {{
    color: {THEME['muted']};
    font-weight: 600;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
[data-testid="stMetricValue"] {{
    color: {THEME['text']};
    font-weight: 700;
}}

/* ---------- Chat ---------- */
[data-testid="stChatMessage"] {{
    border: 1px solid {THEME['border']};
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 12px;
    background-color: {THEME['card_bg']};
}}

/* ---------- Tables ---------- */
[data-testid="stDataFrame"] {{
    border-radius: 12px;
    overflow: hidden;
}}

/* ---------- Buttons ---------- */
.stButton > button {{
    border-radius: 10px;
    font-weight: 600;
}}
.stButton > button[kind="primary"] {{
    background: {THEME['primary']};
    border-color: {THEME['primary']};
}}
.stButton > button[kind="primary"]:hover {{
    background: {THEME['primary_dark']};
    border-color: {THEME['primary_dark']};
}}

/* ---------- Pill / badge utilities ---------- */
.ft-badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}}
.ft-badge-blue   {{ background: #dbeafe; color: #1e3a8a; }}
.ft-badge-amber  {{ background: #fef3c7; color: #92400e; }}
.ft-badge-green  {{ background: #d1fae5; color: #065f46; }}
.ft-badge-rose   {{ background: #ffe4e6; color: #9f1239; }}
.ft-badge-slate  {{ background: #e2e8f0; color: #1e293b; }}
</style>
"""


def inject_styles() -> None:
    """Inject the shared CSS theme into the current Streamlit page."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def hero_header(title: str, subtitle: str, eyebrow: Optional[str] = None) -> None:
    """Render the gradient hero banner used at the top of every page."""
    eyebrow_html = (
        f'<div class="ft-hero-pill">{eyebrow}</div>' if eyebrow else ""
    )
    st.markdown(
        f"""
        <div class="ft-hero">
            {eyebrow_html}
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(number: int, title: str, subtitle: Optional[str] = None) -> None:
    """Render a numbered, underlined section header."""
    sub_html = (
        f'<span class="ft-section-sub">{subtitle}</span>' if subtitle else ""
    )
    st.markdown(
        f"""
        <div class="ft-section">
            <span class="ft-section-num">{number}</span>
            <span>{title}</span>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def warning_badge(label: str) -> str:
    """Return an HTML pill matching the pipeline ``warning_flag`` value."""
    label_lower = (label or "").lower()
    cls = {
        "normal": "ft-badge-green",
        "medium": "ft-badge-amber",
        "review": "ft-badge-amber",
        "low": "ft-badge-rose",
    }.get(label_lower, "ft-badge-slate")
    return f'<span class="ft-badge {cls}">{label}</span>'


# ===========================================================================
# Chart helpers
# ===========================================================================
def apply_chart_theme(fig: go.Figure, *, height: int = 420) -> go.Figure:
    """Apply the shared Plotly look: clean grid, soft labels, sensible legend."""
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=56, b=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, system-ui, -apple-system, sans-serif", color=THEME["text"], size=12),
        title=dict(font=dict(size=15, color=THEME["text"]), x=0, xanchor="left"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0)",
        ),
        hoverlabel=dict(bgcolor="white", bordercolor=THEME["border"]),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="#eef2f7", zeroline=False,
        showline=True, linecolor=THEME["border"], ticks="outside", tickcolor=THEME["border"],
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#eef2f7", zeroline=True, zerolinecolor=THEME["border"],
        showline=False,
    )
    return fig


def add_forecast_connector(
    fig: go.Figure,
    history: pd.DataFrame,
    forecast: pd.DataFrame,
) -> None:
    """Draw a dotted slate-gray bridge from the last actual to the first forecast.

    For intermittent products the recent-history aggregation drops zero-sale
    weeks, which can leave a visible gap before the forecast starts.  This
    connector communicates that the gap is not a missing prediction - it is
    just the no-sale weeks between the last observed sale and the dataset's
    forecast cutoff.
    """
    if history is None or forecast is None or history.empty or forecast.empty:
        return
    last_actual_x = history["week_start"].iloc[-1]
    last_actual_y = float(history["sales"].iloc[-1])
    first_forecast_x = forecast["week_start"].iloc[0]
    first_forecast_y = float(forecast["forecast_sales"].iloc[0])

    fig.add_trace(
        go.Scatter(
            x=[last_actual_x, first_forecast_x],
            y=[last_actual_y, first_forecast_y],
            mode="lines",
            name="Bridge (no-sale weeks)",
            line=dict(color=THEME["bridge"], width=1.6, dash="dot"),
            hovertemplate=(
                "Bridge between last actual and first forecast<br>"
                "(weeks with zero sales are not shown in the actual line)"
                "<extra></extra>"
            ),
            showlegend=True,
        )
    )


def build_actual_vs_forecast_chart(
    product_id: str,
    country: str,
    history: pd.DataFrame,
    forecast: pd.DataFrame,
) -> go.Figure:
    """Compose the standard 'Recent actual vs Forecast' Plotly chart."""
    fig = go.Figure()

    if history is not None and not history.empty:
        history = history.copy()
        history["week_start"] = pd.to_datetime(history["week_start"])
        fig.add_trace(
            go.Scatter(
                x=history["week_start"],
                y=history["sales"],
                mode="lines+markers",
                name="Recent actual",
                line=dict(color=THEME["primary"], width=2.4),
                marker=dict(size=7, line=dict(color="white", width=1)),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Actual: %{y:,.2f}<extra></extra>",
            )
        )
    if forecast is not None and not forecast.empty:
        forecast = forecast.copy()
        forecast["week_start"] = pd.to_datetime(forecast["week_start"])
        fig.add_trace(
            go.Scatter(
                x=forecast["week_start"],
                y=forecast["forecast_sales"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color=THEME["accent"], width=2.4, dash="dash"),
                marker=dict(size=7, line=dict(color="white", width=1)),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Forecast: %{y:,.2f}<extra></extra>",
            )
        )

    add_forecast_connector(fig, history, forecast)

    fig.update_layout(
        title=f"{product_id} ({country}) - actual vs forecast weekly sales",
        xaxis_title="Week",
        yaxis_title="Weekly sales",
    )
    return apply_chart_theme(fig, height=420)


# ===========================================================================
# Cached artifact loaders
# ===========================================================================
@st.cache_data(show_spinner=False)
def _cached_features() -> pd.DataFrame:
    return agent_tools.get_clustered_features_df()


@st.cache_data(show_spinner=False)
def _cached_selection() -> pd.DataFrame:
    return agent_tools.get_selection_summary_df()


@st.cache_data(show_spinner=False)
def _cached_candidates() -> pd.DataFrame:
    return agent_tools.get_candidate_metrics_df()


@st.cache_data(show_spinner=False)
def _cached_metadata() -> Dict[str, Any]:
    return agent_tools.get_cluster_metadata()


@st.cache_data(show_spinner=False)
def _cached_products() -> List[str]:
    return agent_tools.list_available_products()


# ===========================================================================
# Small formatting utilities
# ===========================================================================
def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _format_mape(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _safe_value(row: pd.Series, col: str) -> Any:
    if col not in row.index:
        return None
    val = row[col]
    if isinstance(val, (np.floating, float)):
        return None if not np.isfinite(float(val)) else float(val)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if pd.isna(val):
        return None
    return val


# ===========================================================================
# Main page renderer
# ===========================================================================
def render(page_title: str = "Model / Cluster Comparison") -> None:
    """Render the full dashboard inside the current Streamlit context."""
    inject_styles()
    hero_header(
        title=page_title,
        subtitle=(
            "Explore product clusters, the saved best model per cluster, and "
            "weekly demand forecasts driven by the existing FortuneTellers pipeline."
        ),
        eyebrow="FortuneTellers · Dashboard",
    )

    # ---- Load artifacts up front so missing files are reported once. ----
    missing: List[str] = []
    try:
        features = _cached_features()
    except FileNotFoundError as exc:
        missing.append(str(exc))
        features = pd.DataFrame()
    try:
        selection = _cached_selection()
    except FileNotFoundError as exc:
        missing.append(str(exc))
        selection = pd.DataFrame()
    try:
        candidates = _cached_candidates()
    except FileNotFoundError as exc:
        missing.append(str(exc))
        candidates = pd.DataFrame()
    metadata = _cached_metadata()

    if missing:
        for msg in missing:
            st.error(msg)
        if features.empty:
            st.stop()

    # ---- Pipeline readiness banner ----
    pipeline_ready = agent_tools.is_pipeline_ready()
    if not pipeline_ready:
        st.warning(agent_tools.pipeline_setup_message())
        cached_entries = agent_tools.list_cached_forecasts()
        if cached_entries:
            cached_str = ", ".join(
                f"{e['product_id']} ({e['country']})" for e in cached_entries
            )
            st.info("Cached forecasts available without raw data: " + cached_str + ".")

    # ---- Sidebar controls ----
    with st.sidebar:
        st.markdown("### Filters")

        product_ids = features.index.astype(str).tolist() if not features.empty else []
        default_idx = product_ids.index("85123A") if "85123A" in product_ids else 0
        product_id = st.selectbox(
            "Product / SKU",
            product_ids,
            index=default_idx if product_ids else 0,
            help="Pick a StockCode from the clustered features artifact.",
        )

        countries = agent_tools.list_available_countries(product_id) or ["ALL"]
        country = st.selectbox("Country", countries, index=0)

        cluster_options: List[Any] = ["(use the product's cluster)"]
        if not features.empty and "cluster" in features.columns:
            cluster_options += sorted(features["cluster"].unique().tolist())
        cluster_choice = st.selectbox(
            "Cluster filter",
            cluster_options,
            index=0,
            help="Restrict the candidate-model table to one cluster, or follow the selected product.",
        )

        horizon = st.number_input(
            "Forecast horizon (weeks)",
            min_value=1, max_value=52, value=12, step=1,
        )

        candidate_filter: List[str] = []
        if not candidates.empty and "candidate" in candidates.columns:
            candidate_filter = st.multiselect(
                "Candidate model filter",
                sorted(candidates["candidate"].dropna().astype(str).unique().tolist()),
                default=[],
            )

        st.markdown("---")
        run_forecast = st.button("Run forecast", type="primary", use_container_width=True)

    if features.empty or product_id is None:
        st.warning("No clustered features available - cannot render the dashboard.")
        return

    # =======================================================================
    # 1. Product overview
    # =======================================================================
    section_header(1, "Product overview", "Cluster, sales footprint and customer base")

    product_row = features.loc[features.index.astype(str) == str(product_id)].iloc[0]

    description = "(no description available)"
    if pipeline_ready:
        try:
            pipeline = agent_tools.get_pipeline()
            prod_tx = pipeline.total_retail[
                pipeline.total_retail["StockCode"].astype(str) == str(product_id)
            ]
            if not prod_tx.empty:
                description = str(prod_tx["Description"].iloc[0])
        except Exception:
            pass
    else:
        for entry in agent_tools.list_cached_forecasts():
            if entry["product_id"].upper() == str(product_id).upper():
                try:
                    import json as _json
                    cached_payload = _json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
                    desc = cached_payload.get("product_description")
                    if desc:
                        description = str(desc)
                        break
                except Exception:
                    continue

    cluster_id = int(_safe_value(product_row, "cluster") or 0)
    cluster_label = str(_safe_value(product_row, "cluster_label") or "unknown")

    overview_cols = st.columns(5)
    overview_cols[0].metric("Product", str(product_id))
    overview_cols[1].metric("Cluster", f"{cluster_id} — {cluster_label}")
    overview_cols[2].metric(
        "Mean weekly sales", _format_number(_safe_value(product_row, "mean_weekly_sales"))
    )
    overview_cols[3].metric("Total sales", _format_number(_safe_value(product_row, "total_sales")))
    pct_zero = _safe_value(product_row, "pct_zero_weeks")
    overview_cols[4].metric(
        "Zero-sales weeks",
        f"{(pct_zero or 0) * 100:.1f}%" if pct_zero is not None else "n/a",
    )

    st.markdown(" ")
    detail_cols = st.columns(3)
    with detail_cols[0]:
        st.markdown("**Description**")
        st.write(description)
        st.markdown(f"**Country filter**  `{country}`")
    with detail_cols[1]:
        st.markdown(f"**Unique customers**  {int(_safe_value(product_row, 'n_unique_customers') or 0):,}")
        st.markdown(
            f"**Cancellation rate**  {(_safe_value(product_row, 'cancel_rate') or 0) * 100:.1f}%"
        )
    with detail_cols[2]:
        st.markdown(f"**Coefficient of variation**  {_format_number(_safe_value(product_row, 'cv'))}")
        st.markdown(f"**Q4 share of sales**  {(_safe_value(product_row, 'q4_pct') or 0) * 100:.1f}%")

    # =======================================================================
    # 2. Cluster information
    # =======================================================================
    section_header(2, "Cluster information", "Representative features for the product's cluster")

    cluster_size = int((features["cluster"] == cluster_id).sum())
    cluster_meta_cols = st.columns(4)
    cluster_meta_cols[0].metric("Cluster ID", cluster_id)
    cluster_meta_cols[1].metric("Cluster label", cluster_label)
    cluster_meta_cols[2].metric("Products in cluster", f"{cluster_size:,}")
    if metadata and "best_k" in metadata:
        cluster_meta_cols[3].metric("Best k (silhouette)", metadata.get("best_k"))

    cluster_features = features[features["cluster"] == cluster_id]
    if not cluster_features.empty:
        rep_features = (
            cluster_features[
                [
                    c for c in [
                        "mean_weekly_sales",
                        "cv",
                        "pct_zero_weeks",
                        "q4_pct",
                        "seasonal_conc",
                        "log_mean_price",
                        "cancel_rate",
                    ]
                    if c in cluster_features.columns
                ]
            ]
            .agg(["mean", "median"])
            .T
        )
        rep_features.columns = ["mean", "median"]
        st.markdown("**Representative cluster features**")
        st.dataframe(rep_features.round(4), use_container_width=True)

    if metadata.get("cluster_labels"):
        with st.expander("Cluster label dictionary (`cluster_metadata.json`)"):
            st.json(metadata.get("cluster_labels"))

    # =======================================================================
    # 3. Model selection summary
    # =======================================================================
    section_header(3, "Model selection summary", "Saved best model per cluster, with test/valid MAPE")

    if selection.empty:
        st.warning("`cluster_model_selection_summary.csv` is missing - skipping this section.")
    else:
        display_sel = selection.copy()
        for col in [
            "valid_mape_selected",
            "test_mape_selected",
            "test_mape_lgbm_baseline",
            "delta_selected_minus_lgbm",
        ]:
            if col in display_sel.columns:
                display_sel[col] = display_sel[col].astype(float).round(2)
        st.dataframe(display_sel, use_container_width=True, hide_index=True)

        cluster_sel = display_sel[display_sel["cluster"] == cluster_id]
        if not cluster_sel.empty:
            sel = cluster_sel.iloc[0]
            st.success(
                f"For cluster {cluster_id} ({cluster_label}) the saved best model is "
                f"**{sel.get('selected_model')}** with test MAPE "
                f"{_format_mape(sel.get('test_mape_selected'))} "
                f"(LGBM baseline {_format_mape(sel.get('test_mape_lgbm_baseline'))})."
            )

    # =======================================================================
    # 4. Forecast visualization
    # =======================================================================
    section_header(4, "Forecast visualization", f"Run the saved best model for {product_id} ({country})")

    payload: Optional[Dict[str, Any]] = None
    if run_forecast:
        with st.spinner("Running ForecastingPipeline.forecast_product..."):
            payload = agent_tools.run_forecast(
                product_id, country=country, horizon_weeks=int(horizon)
            )
    else:
        cached = agent_tools.get_last_forecast(product_id)
        if cached and cached.get("country") == country:
            payload = cached
        elif not pipeline_ready:
            payload = agent_tools.run_forecast(
                product_id, country=country, horizon_weeks=int(horizon)
            )

    if payload is None:
        st.info("Click **Run forecast** in the sidebar to generate a forecast for the selected product.")
    elif not payload.get("ok"):
        if not pipeline_ready:
            st.info(
                f"No cached forecast for **{product_id}** in **{country}**. "
                "Either restore the raw transaction data, or pick a product/country "
                "combo that has a saved JSON in `fortunetellers_artifacts/pipeline/`."
            )
        else:
            st.error(payload.get("error", "Forecast failed."))
    else:
        history = pd.DataFrame(payload.get("recent_12_weeks", []))
        forecast_df = pd.DataFrame(payload.get("forecast_12_weeks", []))

        kpi_cols = st.columns(4)
        summary = payload.get("forecast_summary", {})
        kpi_cols[0].metric("Mean forecast", _format_number(summary.get("mean_forecast_sales")))
        kpi_cols[1].metric("Peak forecast", _format_number(summary.get("peak_forecast_sales")))
        kpi_cols[2].metric("Total forecast", _format_number(summary.get("total_forecast_sales")))
        kpi_cols[3].metric("Selected MAPE", _format_mape(payload.get("test_mape_selected")))

        fig = build_actual_vs_forecast_chart(product_id, country, history, forecast_df)
        st.plotly_chart(fig, use_container_width=True)

        if not forecast_df.empty:
            st.markdown("**Forecast table**")
            st.dataframe(
                forecast_df.rename(
                    columns={
                        "week": "Week",
                        "week_start": "Week start",
                        "forecast_sales": "Forecast sales",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    # =======================================================================
    # 5. Candidate model comparison
    # =======================================================================
    section_header(5, "Candidate model comparison", "All candidates evaluated for this cluster")

    if candidates.empty:
        st.info("`cluster_candidate_metrics.csv` is missing - candidate comparison unavailable.")
    else:
        if cluster_choice == "(use the product's cluster)":
            cluster_filter = cluster_id
        else:
            cluster_filter = int(cluster_choice)

        cand_view = candidates[candidates["cluster"] == cluster_filter].copy()
        if candidate_filter:
            cand_view = cand_view[cand_view["candidate"].astype(str).isin(candidate_filter)]

        if cand_view.empty:
            st.info("No candidate metrics for this cluster / filter combination.")
        else:
            for col in ["valid_mape", "test_mape"]:
                if col in cand_view.columns:
                    cand_view[col] = cand_view[col].astype(float).round(2)
            sort_col = "valid_mape" if "valid_mape" in cand_view.columns else cand_view.columns[-1]
            cand_view = cand_view.sort_values(sort_col)

            selected_for_cluster: Optional[str] = None
            if not selection.empty:
                sel_row = selection[selection["cluster"] == cluster_filter]
                if not sel_row.empty:
                    selected_for_cluster = str(sel_row.iloc[0].get("selected_model"))

            def _highlight(row: pd.Series) -> List[str]:
                color = "background-color: #fef3c7; font-weight: 600;" if (
                    selected_for_cluster
                    and str(row.get("candidate")) == selected_for_cluster
                ) else ""
                return [color] * len(row)

            st.dataframe(
                cand_view.style.apply(_highlight, axis=1),
                use_container_width=True,
                hide_index=True,
            )
            if selected_for_cluster:
                st.caption(
                    f"Highlighted row = saved best model for cluster {cluster_filter}: "
                    f"**{selected_for_cluster}**."
                )


if __name__ == "__main__":
    st.set_page_config(
        page_title="FortuneTellers - Dashboard",
        page_icon=None,
        layout="wide",
    )
    render()
