"""
FortuneTellers - AI Forecasting Chatbot (Streamlit page).

Run with::

    streamlit run dashboard/app.py

The chatbot reuses the existing ``fortunetellers`` package and reads model
selection from the saved artifacts.  An LLM API key is *optional*: without
one the bot uses the deterministic regex parser in
``dashboard/agent_tools.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure both this folder and the project root are importable.
_THIS_FILE = Path(__file__).resolve()
_DASHBOARD_DIR = _THIS_FILE.parent
_PROJECT_ROOT = _DASHBOARD_DIR.parent
for p in (_DASHBOARD_DIR, _PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import agent_tools  # noqa: E402
from dashboard import (  # noqa: E402
    THEME,
    apply_chart_theme,
    build_actual_vs_forecast_chart,
    hero_header,
    inject_styles,
    section_header,
    warning_badge,
)


# ---------------------------------------------------------------------------
# Streamlit page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FortuneTellers - AI Forecasting Chatbot",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

hero_header(
    title="AI Forecasting Chatbot",
    subtitle=(
        'Ask in plain English — e.g. "Forecast product 85123A in the UK for the '
        "next 12 weeks.\" The bot finds the product's cluster, looks up the "
        "saved best model for that cluster, and runs the existing forecasting pipeline."
    ),
    eyebrow="FortuneTellers · Chatbot",
)


# ---------------------------------------------------------------------------
# Sidebar - settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Settings")

    api_key = st.text_input(
        "LLM API key (optional)",
        value=os.environ.get("ANTHROPIC_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        type="password",
        help=(
            "Optional. The chatbot works fully offline using a regex parser. "
            "An API key is only used to make replies sound more natural; "
            "model selection is always artifact-driven."
        ),
    )

    default_country = st.text_input(
        "Default country",
        value="United Kingdom",
        help="Used when the user does not specify a country in their query.",
    )

    default_horizon = st.number_input(
        "Default forecast horizon (weeks)",
        min_value=1,
        max_value=52,
        value=12,
        step=1,
    )

    st.markdown("---")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.session_state.pop("last_payload", None)
        st.rerun()

    st.markdown("---")
    with st.expander("How the model is chosen", expanded=False):
        st.markdown(
            "1. Parse the StockCode, country and horizon from the message.\n"
            "2. Look up the product's cluster in `product_features_clustered.csv`.\n"
            "3. Read the cluster's saved best model from `best_model_params.json`.\n"
            "4. Call `ForecastingPipeline.forecast_product(...)` with that model."
        )

    with st.expander("Try these examples", expanded=False):
        st.markdown(
            "- *Forecast product 85123A in the UK for the next 12 weeks*\n"
            "- *Give me a 20-week forecast for SKU 85123A in all countries*\n"
            "- *What is the forecast for 85123A in United Kingdom?*\n"
            "- *Show demand forecast for product 85123A*"
        )


# ---------------------------------------------------------------------------
# Pipeline-readiness banner
# ---------------------------------------------------------------------------
if not agent_tools.is_pipeline_ready():
    st.warning(agent_tools.pipeline_setup_message())
    cached = agent_tools.list_cached_forecasts()
    if cached:
        st.info(
            "Cached forecasts available without raw data: "
            + ", ".join(f"{e['product_id']} ({e['country']})" for e in cached)
            + "."
        )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I forecast weekly retail demand using your saved cluster models. "
                "Try: *Forecast product 85123A in the UK for the next 12 weeks*."
            ),
        }
    ]

if "last_payload" not in st.session_state:
    st.session_state.last_payload: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_mape(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _maybe_polish_with_llm(text: str, api_key: str) -> str:
    """If an API key is given, lightly rewrite the assistant text.

    Model *selection* is still always determined by the artifacts -
    the LLM only rewrites the prose for fluency.  We fall back silently
    on any error.
    """
    if not api_key or not text:
        return text
    try:
        import anthropic  # type: ignore

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Rewrite the following retail forecast briefing to sound natural "
                        "and concise for a business reader. Keep all numbers, model names "
                        "and cluster labels exactly as written. Do not invent new facts.\n\n"
                        f"{text}"
                    ),
                }
            ],
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text")) or text
    except Exception:
        return text


def _render_forecast_payload(payload: Dict[str, Any]) -> None:
    """Render metric cards, charts and tables for a forecast payload."""
    if not payload or not payload.get("ok"):
        return

    summary = payload.get("forecast_summary", {})
    statistics = payload.get("statistics", {}) or {}

    section_header(1, "Key forecast numbers", f"{payload.get('forecast_horizon_weeks', '?')}-week horizon")
    cols = st.columns(5)
    cols[0].metric("Mean forecast", _format_number(summary.get("mean_forecast_sales")))
    cols[1].metric("Peak forecast", _format_number(summary.get("peak_forecast_sales")))
    cols[2].metric("Total forecast", _format_number(summary.get("total_forecast_sales")))
    cols[3].metric("Selected MAPE", _format_mape(payload.get("test_mape_selected")))
    cols[4].metric("Recent 12-wk avg", _format_number(statistics.get("recent_12week_avg")))

    # ---- Combined history + forecast chart with bridge line ----
    section_header(2, "Recent actual vs forecast", "Dashed line bridges weeks with no observed sales")
    history = pd.DataFrame(payload.get("recent_12_weeks", []))
    forecast_df = pd.DataFrame(payload.get("forecast_12_weeks", []))
    fig = build_actual_vs_forecast_chart(
        payload.get("product_id", ""),
        payload.get("country", ""),
        history,
        forecast_df,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Recent history mini chart ----
    if not history.empty:
        section_header(3, "Recent weekly sales", "Last weeks the product actually sold")
        history2 = history.copy()
        history2["week_start"] = pd.to_datetime(history2["week_start"])
        hist_fig = go.Figure(
            go.Bar(
                x=history2["week_start"],
                y=history2["sales"],
                marker_color=THEME["primary"],
                marker_line_color=THEME["primary_dark"],
                marker_line_width=0.5,
                name="Recent",
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Actual: %{y:,.2f}<extra></extra>",
            )
        )
        hist_fig.update_layout(xaxis_title="Week", yaxis_title="Sales")
        apply_chart_theme(hist_fig, height=240)
        st.plotly_chart(hist_fig, use_container_width=True)

    # ---- Forecast table ----
    if not forecast_df.empty:
        section_header(4, "Forecast table", "Per-week forecasted sales")
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


def _format_assistant_reply(forecast: Dict[str, Any], assistant: Dict[str, Any]) -> str:
    """Build the markdown reply for the chat history bubble."""
    f = assistant["fields"]
    badge = warning_badge(str(f.get("warning_flag", "")))
    lines = [
        f"**{assistant['headline']}**",
        "",
        f"- Product: `{f['product_id']}` — {f['product_description']}",
        f"- Country: {f['country']}",
        f"- Cluster: {f['cluster']} — {f['cluster_label']}",
        f"- Selected model: **{f['selected_model']}**",
        f"- Test MAPE: {_format_mape(f['test_mape_selected'])}",
        f"- Forecast horizon: {f['forecast_horizon_weeks']} weeks",
        f"- Mean forecast sales: {_format_number(f['mean_forecast_sales'])}",
        f"- Peak forecast sales: {_format_number(f['peak_forecast_sales'])}",
        f"- Total forecast sales: {_format_number(f['total_forecast_sales'])}",
        f"- Warning level: {badge}",
        f"- Recommendation: {f['recommendation']}",
        "",
        "**Why this model?**",
        assistant["rationale"],
        "",
        "**About MAPE**",
        assistant["mape_explanation"],
    ]
    return "\n".join(lines)


def _handle_user_query(
    query: str, default_country: str, default_horizon: int, api_key: str
) -> Dict[str, Any]:
    """Run the full pipeline for one chat turn and return the payload."""
    parsed = agent_tools.parse_forecast_request(
        query, default_country=default_country, default_horizon=default_horizon
    )
    if not parsed["ok"]:
        return {"ok": False, "error": parsed["error"], "parsed": parsed}

    forecast = agent_tools.run_forecast(
        parsed["product_id"],
        country=parsed["country"],
        horizon_weeks=parsed["horizon_weeks"],
    )
    if not forecast.get("ok"):
        return {
            "ok": False,
            "error": forecast.get("error", "Unknown forecasting error."),
            "parsed": parsed,
        }

    best_model = agent_tools.get_best_model(parsed["product_id"])
    assistant = agent_tools.build_assistant_payload(parsed, forecast, best_model)
    reply_text = _format_assistant_reply(forecast, assistant)
    polished = _maybe_polish_with_llm(reply_text, api_key)

    return {
        "ok": True,
        "error": None,
        "parsed": parsed,
        "forecast": forecast,
        "best_model": best_model,
        "assistant": assistant,
        "reply_text": polished,
    }


# ---------------------------------------------------------------------------
# Render existing chat history
# ---------------------------------------------------------------------------
chat_col, _ = st.columns([1, 0.001])  # full-width chat region

with chat_col:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

# Render the most recent payload (charts + table) below the chat history.
if st.session_state.last_payload is not None and st.session_state.last_payload.get("ok"):
    with st.expander("Latest forecast details", expanded=True):
        _render_forecast_payload(st.session_state.last_payload["forecast"])


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
prompt = st.chat_input("Ask for a forecast — e.g. 'Forecast 85123A in the UK for 12 weeks'")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Looking up cluster, picking the saved best model, and forecasting..."):
            try:
                result = _handle_user_query(
                    prompt,
                    default_country=default_country,
                    default_horizon=int(default_horizon),
                    api_key=api_key,
                )
            except FileNotFoundError as exc:
                result = {"ok": False, "error": str(exc)}
            except Exception as exc:  # noqa: BLE001
                result = {"ok": False, "error": f"Unexpected error: {exc}"}

        if not result.get("ok"):
            err = result.get("error", "Something went wrong.")
            st.error(err)
            err_md = "**Error**\n\n```\n" + str(err) + "\n```"
            st.session_state.messages.append({"role": "assistant", "content": err_md})
        else:
            forecast = result["forecast"]
            if forecast.get("warning"):
                st.warning(forecast["warning"])
            st.markdown(result["reply_text"], unsafe_allow_html=True)
            _render_forecast_payload(forecast)
            st.session_state.messages.append(
                {"role": "assistant", "content": result["reply_text"]}
            )
            st.session_state.last_payload = result
