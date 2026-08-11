"""
tests/test_dashboard.py
────────────────────────
Smoke test suite for the Streamlit Monitoring Dashboard queries and empty-state safety.
"""

from __future__ import annotations

import pandas as pd

from dashboard.queries import (
    get_accuracy_history,
    get_available_models,
    get_deployment_history,
    get_drift_history,
    get_incidents_history,
    get_model_summary,
    get_prediction_volume,
)


class TestDashboardQueriesSmoke:
    def test_get_available_models_returns_list(self, test_engine):
        models = get_available_models(test_engine)
        assert isinstance(models, list)
        assert "churn-model" in models
        assert "fraud-model" in models

    def test_get_model_summary_returns_valid_structure(self, test_engine):
        summary = get_model_summary(test_engine, "churn-model")
        assert isinstance(summary, dict)
        assert "active_version" in summary
        assert "total_predictions" in summary
        assert "alerts_24h" in summary
        assert "open_incidents" in summary

    def test_get_drift_history_handles_empty(self, test_engine):
        df = get_drift_history(test_engine, "churn-model")
        assert isinstance(df, pd.DataFrame)

    def test_get_accuracy_history_handles_empty(self, test_engine):
        df = get_accuracy_history(test_engine, "churn-model")
        assert isinstance(df, pd.DataFrame)

    def test_get_incidents_history_handles_empty(self, test_engine):
        df = get_incidents_history(test_engine, "churn-model")
        assert isinstance(df, pd.DataFrame)

    def test_get_deployment_history_handles_empty(self, test_engine):
        df = get_deployment_history(test_engine, "churn-model")
        assert isinstance(df, pd.DataFrame)

    def test_get_prediction_volume_handles_empty(self, test_engine):
        df = get_prediction_volume(test_engine, "churn-model")
        assert isinstance(df, pd.DataFrame)
