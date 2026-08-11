"""
models/feature_config.py
────────────────────────
Single source of truth for all column names and their roles in the
churn-model pipeline.

WHY THIS FILE EXISTS:
  Both the training script and the FastAPI input schema need to agree on
  exactly which features exist, their types, and how they are named.
  If this list lives in two places they will silently drift apart.
  Any future change (add a feature, rename a column) happens ONCE here.
"""

# ---------------------------------------------------------------------------
# Raw target column (dropped from features before training)
# ---------------------------------------------------------------------------
TARGET_COL = "Churn"

# ---------------------------------------------------------------------------
# Customer ID column (dropped — not a predictive feature)
# ---------------------------------------------------------------------------
ID_COL = "customerID"

# ---------------------------------------------------------------------------
# Numeric features
# NaN strategy: MEDIAN imputation — robust to the right-skewed billing
# distributions common in telecom data (heavy long-tail of long-tenure
# customers with high TotalCharges).
# ---------------------------------------------------------------------------
NUMERIC_FEATURES: list[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",  # arrives as object dtype due to spaces; cast to float first
]

# ---------------------------------------------------------------------------
# Categorical features
# NaN strategy: constant "missing" — preserves the absence of data as a
# potentially meaningful signal rather than collapsing it into a mode.
# Unknown values at inference: OrdinalEncoder with unknown_value=-1 maps
# unseen categories to a dedicated -1 bucket rather than crashing or
# silently mispredicting (see train.py for encoder config).
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "SeniorCitizen",  # 0/1 int stored as categorical for consistency
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# All feature columns (order must match ColumnTransformer output for the
# Pydantic schema — numeric first, then categorical, matching the transformer).
ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Convenience set for fast membership checks
ALL_FEATURES_SET: set[str] = set(ALL_FEATURES)

# ---------------------------------------------------------------------------
# Model registry constants
# ---------------------------------------------------------------------------
MODEL_NAME = "churn-model"
CURRENT_VERSION_FILENAME = "current.json"
CANARY_FILENAME = "canary.json"
ARTIFACT_FILENAME = "pipeline.joblib"
METADATA_FILENAME = "metadata.json"
