"""
dashboard/app.py
────────────────
Streamlit Monitoring Dashboard for the Self-Healing ML Serving Platform.

Read-only visualization layer over PostgreSQL telemetry tables:
  - Multi-tenant dropdown selector
  - Deployment timeline & status indicators
  - Feature PSI drift charts over time (with threshold reference lines)
  - Rolling accuracy/precision/recall/F1/ROC-AUC trends
  - Incident management table with expandable LLM explanations
  - Prediction volume & traffic flow charts

Usage:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import os

import pandas as pd
import streamlit as st

# ── Inject Streamlit Cloud Secrets into os.environ for pydantic-settings ────
try:
    for s_key in ("DATABASE_URL", "database_url", "API_BASE_URL", "api_base_url"):
        if s_key in st.secrets and s_key.upper() not in os.environ:
            os.environ[s_key.upper()] = str(st.secrets[s_key])
except Exception:
    pass

# ── Make project root importable ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.database import engine  # noqa: E402
from dashboard.queries import (  # noqa: E402
    get_accuracy_history,
    get_available_models,
    get_deployment_history,
    get_drift_history,
    get_incidents_history,
    get_model_summary,
    get_prediction_volume,
)

log = logging.getLogger("dashboard.app")

# ---------------------------------------------------------------------------
# Streamlit Page Config & Custom Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Self-Healing ML Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Enterprise CSS
st.markdown(
    """
    <style>
    /* Global Font & Spacing */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    
    /* Clean Cards */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #eaecf0;
        border-radius: 10px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(16,24,40,0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 500;
        color: #475467;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #101828;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Status Badges */
    .badge {
        font-size: 0.72rem;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-success {
        background-color: #ecfdf3;
        color: #027a48;
        border: 1px solid #abefc6;
    }
    .badge-warning {
        background-color: #fffaeb;
        color: #b54708;
        border: 1px solid #fedf89;
    }
    .badge-neutral {
        background-color: #f2f4f7;
        color: #344054;
        border: 1px solid #e4e7ec;
    }
    
    /* Header Styling */
    .app-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #101828;
        margin-bottom: 0.25rem;
    }
    .app-subheader {
        font-size: 0.95rem;
        color: #475467;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar & Tenant Selection
# ---------------------------------------------------------------------------
def render_sidebar() -> str:
    st.sidebar.title("Platform Control")
    st.sidebar.markdown("---")

    available_models = get_available_models(engine)
    selected_model = st.sidebar.selectbox(
        "Select Model Tenant",
        options=available_models,
        index=0 if available_models else 0,
        help="Switch between active machine learning tenants.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Live Telemetry")
    if st.sidebar.button("Refresh Telemetry Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.info("Data auto-refreshes every 5s on user interaction.")
    st.sidebar.markdown("---")
    st.sidebar.caption("Self-Healing ML Platform v1.0")

    return selected_model  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Overview Metrics Bar
# ---------------------------------------------------------------------------
def render_metrics_summary(summary: dict) -> None:
    version_val = summary["active_version"]
    status_str = summary["status"].upper()

    if status_str in ("DEPLOYED", "CURRENT", "ACTIVE"):
        badge_html = f'<span class="badge badge-success">{status_str}</span>'
    elif status_str == "CANARY":
        badge_html = f'<span class="badge badge-warning">CANARY {summary["traffic_percentage"]:.0f}%</span>'
    else:
        badge_html = f'<span class="badge badge-neutral">{status_str}</span>'

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Active Version</div>
                <div class="metric-value">{version_val} {badge_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Predictions</div>
                <div class="metric-value">{summary['total_predictions']:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Alerts (24h)</div>
                <div class="metric-value">{summary['alerts_24h']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Open Incidents</div>
                <div class="metric-value">{summary['open_incidents']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tab 1: Feature Drift Visualizations
# ---------------------------------------------------------------------------
def render_drift_tab(model_name: str) -> None:
    st.subheader(f"Feature & Output Drift Trends — {model_name}")

    drift_df = get_drift_history(engine, model_name)
    if drift_df.empty:
        st.info("No drift reports recorded yet for this model.")
        return

    # Filter features
    features = sorted(drift_df["feature_name"].unique())
    selected_features = st.multiselect(
        "Filter Features",
        options=features,
        default=features[:5] if len(features) >= 5 else features,
    )

    filtered_df = drift_df[drift_df["feature_name"].isin(selected_features)]
    if filtered_df.empty:
        st.warning("No data available for the selected features.")
        return

    # Pivot for line chart
    pivot_df = filtered_df.pivot_table(
        index="checked_at",
        columns="feature_name",
        values="psi_score",
        aggfunc="last",
    ).ffill()

    st.markdown("**Population Stability Index (PSI) over Time**")
    st.caption(
        "Thresholds: **PSI < 0.10** (Stable) | **0.10 ≤ PSI < 0.25** (Moderate Warning) | **PSI ≥ 0.25** (Critical Drift)"
    )

    st.line_chart(pivot_df, use_container_width=True)

    # Detailed Drift Table
    with st.expander("View Raw Drift Reports Table"):
        st.dataframe(
            drift_df.sort_values("checked_at", ascending=False),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Tab 2: Performance & Accuracy Metrics
# ---------------------------------------------------------------------------
def render_accuracy_tab(model_name: str) -> None:
    st.subheader(f"Rolling Model Accuracy & Performance — {model_name}")

    acc_df = get_accuracy_history(engine, model_name)
    if acc_df.empty:
        st.info("No accuracy reports recorded yet for this model.")
        return

    metrics_to_plot = ["f1", "precision", "recall", "roc_auc"]
    available_metrics = [m for m in metrics_to_plot if m in acc_df.columns]

    selected_metrics = st.multiselect(
        "Select Performance Metrics to Display",
        options=available_metrics,
        default=[m for m in ["f1", "precision", "roc_auc"] if m in available_metrics],
    )

    if not selected_metrics:
        st.warning("Please select at least one metric to display.")
        return

    pivot_acc = acc_df.set_index("checked_at")[selected_metrics].ffill()
    st.line_chart(pivot_acc, use_container_width=True)

    with st.expander("View Raw Accuracy Reports Table"):
        st.dataframe(
            acc_df.sort_values("checked_at", ascending=False),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Tab 3: Incidents & Diagnosis Layer
# ---------------------------------------------------------------------------
def render_incidents_tab(model_name: str) -> None:
    st.subheader(f"Incident Tickets & Diagnosis Agent — {model_name}")

    inc_df = get_incidents_history(engine, model_name)
    if inc_df.empty:
        st.success("No incidents logged for this tenant — model operates normally.")
        return

    st.dataframe(
        inc_df[
            [
                "id",
                "hypothesis",
                "confidence",
                "status",
                "action_taken",
                "created_at",
            ]
        ],
        use_container_width=True,
    )

    st.markdown("### Detailed Incident Reasoning & LLM Explanations")
    for idx, row in inc_df.iterrows():
        status_tag = (
            "[ACTIVE]"
            if row["status"] in ("investigating", "escalated")
            else "[RESOLVED]"
        )
        with st.expander(
            f"{status_tag} Incident #{row['id']} — {row['hypothesis']} (Confidence: {row['confidence']:.2f})"
        ):
            c1, c2 = st.columns(2)
            c1.write(f"**Status:** `{row['status']}`")
            c1.write(f"**Action Taken:** `{row['action_taken']}`")
            c2.write(f"**Model ID:** `{row['model_id']}`")
            c2.write(f"**Logged At:** `{row['created_at']}`")

            st.markdown("---")
            st.markdown("**LLM Human-Readable Explanation:**")
            explanation = row.get("llm_explanation")
            if explanation and pd.notna(explanation):
                st.info(explanation)
            else:
                st.caption(
                    "No LLM explanation generated (auto-resolved by rule-based agent)."
                )

            st.markdown("**Raw Evidence Context:**")
            evidence = row.get("evidence")
            if evidence:
                try:
                    if isinstance(evidence, str):
                        evidence_obj = json.loads(evidence)
                    else:
                        evidence_obj = evidence
                    st.json(evidence_obj)
                except Exception:
                    st.code(str(evidence))


# ---------------------------------------------------------------------------
# Tab 4: Deployments & Traffic Volume
# ---------------------------------------------------------------------------
def render_deployments_tab(model_name: str) -> None:
    st.subheader(f"Deployments & Prediction Traffic Volume — {model_name}")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Deployment History")
        dep_df = get_deployment_history(engine, model_name)
        if dep_df.empty:
            st.info("No deployment history found.")
        else:
            st.dataframe(dep_df, use_container_width=True)

    with c2:
        st.markdown("### Prediction Volume")
        vol_df = get_prediction_volume(engine, model_name)
        if vol_df.empty:
            st.info("No prediction traffic recorded yet.")
        else:
            # Resample traffic count
            vol_df["count"] = 1
            vol_df = vol_df.set_index("created_at")
            traffic_counts = vol_df["count"].resample("1h").sum()
            st.bar_chart(traffic_counts, use_container_width=True)


# ---------------------------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------------------------
def main() -> None:
    selected_model = render_sidebar()

    st.markdown(
        f'<div class="app-header">Self-Healing ML Platform — {selected_model}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="app-subheader">Continuous Telemetry, PSI Drift Detection & Autonomous LLM Operations</div>',
        unsafe_allow_html=True,
    )

    summary = get_model_summary(engine, selected_model)
    render_metrics_summary(summary)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Feature Drift",
            "Performance & Accuracy",
            "Incidents & Diagnosis",
            "Deployments & Traffic",
        ]
    )

    with tab1:
        render_drift_tab(selected_model)

    with tab2:
        render_accuracy_tab(selected_model)

    with tab3:
        render_incidents_tab(selected_model)

    with tab4:
        render_deployments_tab(selected_model)


if __name__ == "__main__":
    main()
