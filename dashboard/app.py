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
# Streamlit Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Self-Healing ML Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Sidebar & Tenant Selection
# ---------------------------------------------------------------------------
def render_sidebar() -> str:
    st.sidebar.title("🛡️ Platform Control")
    st.sidebar.markdown("---")

    available_models = get_available_models(engine)
    selected_model = st.sidebar.selectbox(
        "Select Model Tenant",
        options=available_models,
        index=0 if available_models else 0,
        help="Switch between active machine learning tenants.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Live Refresh")
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
    col1, col2, col3, col4 = st.columns(4)

    version_label = summary["active_version"]
    if summary["status"] == "canary":
        version_label += f" (Canary {summary['traffic_percentage']}%)"

    col1.metric("Active Version", version_label, summary["status"].upper())
    col2.metric("Total Predictions", f"{summary['total_predictions']:,}")
    col3.metric("Alerts (24h)", summary["alerts_24h"])
    col4.metric("Open Incidents", summary["open_incidents"])


# ---------------------------------------------------------------------------
# Tab 1: Feature Drift Visualizations
# ---------------------------------------------------------------------------
def render_drift_tab(model_name: str) -> None:
    st.subheader(f"📊 Feature & Output Drift Trends — {model_name}")

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
    st.subheader(f"📈 Rolling Model Accuracy & Performance — {model_name}")

    acc_df = get_accuracy_history(engine, model_name)
    if acc_df.empty:
        st.info("No accuracy reports recorded yet for this model.")
        return

    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    available_metrics = [m for m in metrics_to_plot if m in acc_df.columns]

    selected_metrics = st.multiselect(
        "Select Performance Metrics to Display",
        options=available_metrics,
        default=["f1", "accuracy", "roc_auc"],
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
    st.subheader(f"🚨 Incident Tickets & Diagnosis Agent — {model_name}")

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

    st.markdown("### 🔍 Detailed Incident Reasoning & LLM Explanations")
    for idx, row in inc_df.iterrows():
        status_icon = "🔴" if row["status"] in ("investigating", "escalated") else "🟢"
        with st.expander(
            f"{status_icon} Incident #{row['id']} — {row['hypothesis']} (Confidence: {row['confidence']:.2f})"
        ):
            c1, c2 = st.columns(2)
            c1.write(f"**Status:** `{row['status']}`")
            c1.write(f"**Action Taken:** `{row['action_taken']}`")
            c2.write(f"**Model ID:** `{row['model_id']}`")
            c2.write(f"**Logged At:** `{row['created_at']}`")

            st.markdown("---")
            st.markdown("**🤖 LLM Human-Readable Explanation:**")
            explanation = row.get("llm_explanation")
            if explanation and pd.notna(explanation):
                st.info(explanation)
            else:
                st.caption(
                    "No LLM explanation generated (auto-resolved by rule-based agent)."
                )

            st.markdown("**📋 Raw Evidence Context:**")
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
    st.subheader(f"🚀 Deployments & Prediction Traffic Volume — {model_name}")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 📋 Deployment History")
        dep_df = get_deployment_history(engine, model_name)
        if dep_df.empty:
            st.info("No deployment history found.")
        else:
            st.dataframe(dep_df, use_container_width=True)

    with c2:
        st.markdown("### 📊 Prediction Volume (Sanity Check)")
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

    st.title(f"🛡️ Self-Healing ML Platform — {selected_model}")
    st.markdown("---")

    summary = get_model_summary(engine, selected_model)
    render_metrics_summary(summary)

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Feature Drift",
            "📈 Performance & Accuracy",
            "🚨 Incidents & Diagnosis",
            "🚀 Deployments & Traffic",
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
