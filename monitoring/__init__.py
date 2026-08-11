"""
monitoring
──────────
Drift detection, accuracy tracking, and alerting package.

Modules:
  psi            — Population Stability Index (PSI) engine from scratch
  db_models      — ORM models for `drift_reports`, `accuracy_reports`, `alerts`
  check_drift    — Feature & prediction drift checking script
  check_accuracy — Rolling accuracy evaluation script
"""

from monitoring.check_accuracy import run_accuracy_check
from monitoring.check_drift import run_drift_check
from monitoring.db_models import AccuracyReport, Alert, DriftReport
from monitoring.psi import (
    MIN_SAMPLE_SIZE,
    PSI_MODERATE_THRESHOLD,
    PSI_STABLE_THRESHOLD,
    calculate_categorical_psi,
    calculate_numeric_psi,
    classify_psi,
)

__all__ = [
    "AccuracyReport",
    "Alert",
    "DriftReport",
    "MIN_SAMPLE_SIZE",
    "PSI_MODERATE_THRESHOLD",
    "PSI_STABLE_THRESHOLD",
    "calculate_categorical_psi",
    "calculate_numeric_psi",
    "classify_psi",
    "run_accuracy_check",
    "run_drift_check",
]
