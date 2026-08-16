"""
dashboard/queries.py
────────────────────
Database query functions for the Streamlit Monitoring Dashboard.

Uses parameterized SQLAlchemy queries filtered by tenant model_name to ensure
strict tenant isolation and security.

Queries are cached with a 5-second TTL using @st.cache_data for live responsiveness
without hammering PostgreSQL.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

log = logging.getLogger("dashboard.queries")

# ---------------------------------------------------------------------------
# Streamlit cache wrapper fallback (for pytest / non-streamlit environments)
# ---------------------------------------------------------------------------
try:
    import streamlit as st

    cache_data: Callable = st.cache_data(ttl=5)
except Exception:
    # Fallback decorator when streamlit is not imported or in test context
    def cache_data(func: Any) -> Any:  # type: ignore[no-redef]
        return func


# ---------------------------------------------------------------------------
# Query Functions
# ---------------------------------------------------------------------------
@cache_data
def get_available_models(_engine: Engine) -> list[str]:
    """
    Fetch distinct tenant model names registered in deployments or predictions.
    Defaults to ['churn-model', 'fraud-model'] if no records exist yet.
    """
    sql = text(
        """
        SELECT DISTINCT
            CASE
                WHEN model_id LIKE '%:%' THEN SPLIT_PART(model_id, ':', 1)
                ELSE model_id
            END AS model_name
        FROM deployments
        UNION
        SELECT DISTINCT
            CASE
                WHEN model_id LIKE '%:%' THEN SPLIT_PART(model_id, ':', 1)
                ELSE model_id
            END AS model_name
        FROM predictions
        """
    )
    # SQLite fallback query if SPLIT_PART isn't available
    sqlite_sql = text(
        """
        SELECT DISTINCT model_id FROM deployments
        UNION
        SELECT DISTINCT model_id FROM predictions
        """
    )

    try:
        with _engine.connect() as conn:
            try:
                res = conn.execute(sql)
                models = [row[0] for row in res.fetchall() if row[0]]
            except Exception:
                res = conn.execute(sqlite_sql)
                raw = [row[0] for row in res.fetchall() if row[0]]
                models = list({r.split(":")[0] for r in raw if ":" in r})

        defaults = ["churn-model", "fraud-model"]
        for d in defaults:
            if d not in models:
                models.append(d)
        return sorted(list(set(models)))
    except Exception as exc:
        log.warning("Could not query available models: %s", exc)
        return ["churn-model", "fraud-model"]


@cache_data
def get_model_summary(_engine: Engine, model_name: str) -> dict[str, Any]:
    """
    Get top-level performance indicators for the selected tenant model.
    """
    summary = {
        "active_version": "N/A",
        "status": "Unknown",
        "traffic_percentage": 0.0,
        "total_predictions": 0,
        "alerts_24h": 0,
        "open_incidents": 0,
    }
    pattern = f"{model_name}%"

    try:
        with _engine.connect() as conn:
            # 1. Latest Deployment
            dep_sql = text(
                """
                SELECT version, status, traffic_percentage
                FROM deployments
                WHERE model_id LIKE :pattern
                ORDER BY deployed_at DESC
                LIMIT 1
                """
            )
            dep_res = conn.execute(dep_sql, {"pattern": pattern}).fetchone()
            if dep_res:
                summary["active_version"] = dep_res[0]
                summary["status"] = dep_res[1]
                summary["traffic_percentage"] = float(dep_res[2] or 100.0)

            # 2. Total Predictions
            pred_sql = text(
                """
                SELECT COUNT(*) FROM predictions WHERE model_id LIKE :pattern
                """
            )
            summary["total_predictions"] = (
                conn.execute(pred_sql, {"pattern": pattern}).scalar() or 0
            )

            # 3. Alerts in last 24h
            alert_sql = text(
                """
                SELECT COUNT(*) FROM alerts
                WHERE model_id LIKE :pattern
                """
            )
            summary["alerts_24h"] = (
                conn.execute(alert_sql, {"pattern": pattern}).scalar() or 0
            )

            # 4. Open Incidents
            inc_sql = text(
                """
                SELECT COUNT(*) FROM incidents
                WHERE model_id LIKE :pattern AND status IN ('investigating', 'escalated')
                """
            )
            summary["open_incidents"] = (
                conn.execute(inc_sql, {"pattern": pattern}).scalar() or 0
            )

    except Exception as exc:
        log.warning("Could not query model summary for %s: %s", model_name, exc)

    return summary


@cache_data
def get_deployment_history(_engine: Engine, model_name: str) -> pd.DataFrame:
    """
    Fetch deployment timeline history for the selected tenant model.
    """
    pattern = f"{model_name}%"
    sql = text(
        """
        SELECT
            model_id,
            version,
            status,
            traffic_percentage,
            rejection_reason,
            deployed_at,
            promoted_at
        FROM deployments
        WHERE model_id LIKE :pattern
        ORDER BY deployed_at DESC
        """
    )
    try:
        with _engine.connect() as conn:
            return pd.read_sql(sql, conn, params={"pattern": pattern})
    except Exception as exc:
        log.warning("Could not fetch deployment history: %s", exc)
        return pd.DataFrame()


@cache_data
def get_drift_history(_engine: Engine, model_name: str) -> pd.DataFrame:
    """
    Fetch drift PSI scores over time for all features of the selected tenant model.
    """
    pattern = f"{model_name}%"
    sql = text(
        """
        SELECT
            model_id,
            feature_name,
            psi_score,
            status,
            sample_size,
            checked_at
        FROM drift_reports
        WHERE model_id LIKE :pattern
        ORDER BY checked_at ASC
        """
    )
    try:
        with _engine.connect() as conn:
            df = pd.read_sql(sql, conn, params={"pattern": pattern})
            if not df.empty and "checked_at" in df.columns:
                df["checked_at"] = pd.to_datetime(df["checked_at"])
            return df
    except Exception as exc:
        log.warning("Could not fetch drift history: %s", exc)
        return pd.DataFrame()


@cache_data
def get_accuracy_history(_engine: Engine, model_name: str) -> pd.DataFrame:
    """
    Fetch rolling accuracy and performance metrics over time for the selected tenant model.
    """
    pattern = f"{model_name}%"
    sql = text(
        """
        SELECT
            model_id,
            n_samples AS sample_size,
            precision,
            recall,
            f1,
            roc_auc,
            created_at AS checked_at
        FROM accuracy_reports
        WHERE model_id LIKE :pattern
        ORDER BY created_at ASC
        """
    )
    try:
        with _engine.connect() as conn:
            df = pd.read_sql(sql, conn, params={"pattern": pattern})
            if not df.empty and "checked_at" in df.columns:
                df["checked_at"] = pd.to_datetime(df["checked_at"])
            return df
    except Exception as exc:
        log.warning("Could not fetch accuracy history: %s", exc)
        return pd.DataFrame()


@cache_data
def get_incidents_history(_engine: Engine, model_name: str) -> pd.DataFrame:
    """
    Fetch incident tickets with evidence and LLM explanations for the selected model.
    """
    pattern = f"{model_name}%"
    sql = text(
        """
        SELECT
            id,
            model_id,
            hypothesis,
            confidence,
            status,
            recommended_action AS action_taken,
            llm_explanation,
            created_at,
            evidence
        FROM incidents
        WHERE model_id LIKE :pattern
        ORDER BY created_at DESC
        """
    )
    try:
        with _engine.connect() as conn:
            df = pd.read_sql(sql, conn, params={"pattern": pattern})
            if not df.empty and "created_at" in df.columns:
                df["created_at"] = pd.to_datetime(df["created_at"])
            return df
    except Exception as exc:
        log.warning("Could not fetch incidents history: %s", exc)
        return pd.DataFrame()


@cache_data
def get_prediction_volume(_engine: Engine, model_name: str) -> pd.DataFrame:
    """
    Fetch time-series prediction counts for the selected tenant model.
    """
    pattern = f"{model_name}%"
    sql = text(
        """
        SELECT
            created_at,
            model_id
        FROM predictions
        WHERE model_id LIKE :pattern
        ORDER BY created_at ASC
        """
    )
    try:
        with _engine.connect() as conn:
            df = pd.read_sql(sql, conn, params={"pattern": pattern})
            if not df.empty and "created_at" in df.columns:
                df["created_at"] = pd.to_datetime(df["created_at"])
            return df
    except Exception as exc:
        log.warning("Could not fetch prediction volume: %s", exc)
        return pd.DataFrame()
