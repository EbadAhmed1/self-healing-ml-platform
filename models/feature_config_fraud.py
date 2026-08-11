"""
models/feature_config_fraud.py
──────────────────────────────
Single source of truth for column names and roles in the fraud-model pipeline.

Tenant #2: Fraud Detection Model (fraud-model)
Demonstrates multi-tenant support across the self-healing platform.
"""

from __future__ import annotations

# Raw target column
TARGET_COL = "is_fraud"

# Transaction ID column
ID_COL = "transaction_id"

# Numeric features
NUMERIC_FEATURES: list[str] = [
    "transaction_amount",
    "location_distance_km",
    "account_age_days",
    "velocity_1h_count",
]

# Categorical features
CATEGORICAL_FEATURES: list[str] = [
    "merchant_category",
    "device_type",
    "is_international",
    "is_flagged_ip",
]

ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ALL_FEATURES_SET: set[str] = set(ALL_FEATURES)

# Model registry constants
MODEL_NAME = "fraud-model"
CURRENT_VERSION_FILENAME = "current.json"
CANARY_FILENAME = "canary.json"
ARTIFACT_FILENAME = "pipeline.joblib"
METADATA_FILENAME = "metadata.json"
