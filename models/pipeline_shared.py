"""
models/pipeline_shared.py
─────────────────────────
Shared machine learning training pipeline utilities.

Provides reusable pipeline construction, data preprocessing, evaluation,
and training distribution snapshot computation used across ALL model tenants
(churn-model, fraud-model, etc.).

Guarantees 100% feature consistency, imputer logic, and evaluation metrics across
all model versions.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from models.feature_config import (
    CATEGORICAL_FEATURES as CHURN_CATEGORICAL,
    ID_COL,
    NUMERIC_FEATURES as CHURN_NUMERIC,
    TARGET_COL,
)

log = logging.getLogger("pipeline_shared")

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply dtype fixes that are properties of the raw data format.
    Safe before train/test splitting as no statistics are fit.
    """
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if TARGET_COL in df.columns:
        if df[TARGET_COL].dtype == object or df[TARGET_COL].dtype == str:
            df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

    df = df.drop(columns=[ID_COL], errors="ignore")

    return df


# ---------------------------------------------------------------------------
# Pipeline Construction
# ---------------------------------------------------------------------------
def build_pipeline(
    numeric_features: list[str] = CHURN_NUMERIC,
    categorical_features: list[str] = CHURN_CATEGORICAL,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """
    Build the scikit-learn Pipeline (ColumnTransformer + LogisticRegression).
    Dynamically accepts numeric and categorical feature lists for any tenant model.
    """
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=random_state,
                    class_weight="balanced",
                    C=1.0,
                ),
            ),
        ]
    )

    return pipeline


# ---------------------------------------------------------------------------
# Pipeline Evaluation
# ---------------------------------------------------------------------------
def evaluate_pipeline(
    pipeline: Pipeline, X_eval: pd.DataFrame, y_eval: pd.Series
) -> dict:
    """
    Evaluate fitted pipeline on held-out evaluation set.
    """
    y_pred = pipeline.predict(X_eval)
    y_prob = pipeline.predict_proba(X_eval)[:, 1]

    metrics = {
        "precision": round(float(precision_score(y_eval, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_eval, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_eval, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_eval, y_prob)), 4),
    }

    log.info("Evaluation metrics: %s", metrics)
    log.debug(
        "\nClassification report:\n%s",
        classification_report(y_eval, y_pred),
    )
    return metrics


# ---------------------------------------------------------------------------
# Training Statistics Snapshot (PSI Baseline)
# ---------------------------------------------------------------------------
def compute_training_snapshot(
    X_train: pd.DataFrame,
    pipeline: Pipeline | None = None,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> dict:
    """
    Compute distribution statistics for baseline PSI comparison in monitoring.
    Accepts custom numeric and categorical feature lists for tenant isolation.
    """
    num_feats = (
        numeric_features
        if numeric_features is not None
        else [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    )
    cat_feats = (
        categorical_features
        if categorical_features is not None
        else [c for c in X_train.columns if c not in num_feats]
    )

    snapshot: dict = {"numeric": {}, "categorical": {}, "prediction": {}}

    for col in num_feats:
        if col in X_train.columns:
            s = pd.to_numeric(X_train[col], errors="coerce").dropna()
            if len(s) > 0:
                deciles = np.percentile(s, np.linspace(10, 90, 9)).round(4).tolist()
                snapshot["numeric"][col] = {
                    "mean": round(float(s.mean()), 4),
                    "std": round(float(s.std()), 4),
                    "median": round(float(s.median()), 4),
                    "p25": round(float(s.quantile(0.25)), 4),
                    "p75": round(float(s.quantile(0.75)), 4),
                    "deciles": deciles,
                    "n_missing": int(X_train[col].isna().sum()),
                }

    for col in cat_feats:
        if col in X_train.columns:
            s = X_train[col].astype(str)
            vc = s.value_counts().to_dict()
            snapshot["categorical"][col] = {
                "value_counts": {str(k): int(v) for k, v in vc.items()},
                "n_missing": int(X_train[col].isna().sum()),
                "n_unique": int(s.nunique()),
            }

    if pipeline is not None and len(X_train) > 0:
        try:
            probs = pipeline.predict_proba(X_train)[:, 1]
            preds = pipeline.predict(X_train)
            deciles = np.percentile(probs, np.linspace(10, 90, 9)).round(4).tolist()
            snapshot["prediction"] = {
                "mean_confidence": round(float(probs.mean()), 4),
                "deciles": deciles,
                "positive_class_ratio": round(float((preds == 1).mean()), 4),
            }
        except Exception as exc:
            log.warning("Could not compute prediction snapshot: %s", exc)

    return snapshot
