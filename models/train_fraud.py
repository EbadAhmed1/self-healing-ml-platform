"""
models/train_fraud.py
─────────────────────
Training script for Tenant #2 (fraud-model: Fraud Detection).

Reuses the shared pipeline module (models/pipeline_shared.py) to demonstrate
100% code reuse for model pipeline construction, evaluation, and snapshot creation.

Usage:
    python models/train_fraud.py [--version v1]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Make project root importable ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.db_models import Deployment  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from models.feature_config_fraud import (  # noqa: E402
    ALL_FEATURES,
    ARTIFACT_FILENAME,
    CATEGORICAL_FEATURES,
    CURRENT_VERSION_FILENAME,
    ID_COL,
    METADATA_FILENAME,
    MODEL_NAME,
    NUMERIC_FEATURES,
    TARGET_COL,
)
from models.pipeline_shared import (  # noqa: E402
    build_pipeline,
    compute_training_snapshot,
    evaluate_pipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("train_fraud")

ARTIFACTS_ROOT = PROJECT_ROOT / "models" / "artifacts"


# ---------------------------------------------------------------------------
# Synthetic Dataset Generator for Fraud Model
# ---------------------------------------------------------------------------
def generate_fraud_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic fraud transaction dataset."""
    rng = np.random.default_rng(seed)

    amounts = rng.exponential(scale=75.0, size=n).round(2)
    distances = rng.exponential(scale=15.0, size=n).round(2)
    account_ages = rng.integers(1, 1000, size=n)
    velocities = rng.poisson(lam=1.5, size=n)

    categories = rng.choice(
        ["retail", "travel", "gaming", "crypto", "electronics"], size=n
    )
    devices = rng.choice(["mobile_ios", "mobile_android", "web_desktop"], size=n)
    internationals = rng.choice([0, 1], size=n, p=[0.85, 0.15])
    flagged_ips = rng.choice([0, 1], size=n, p=[0.92, 0.08])

    # Target calculation logic
    fraud_prob = (
        0.05
        + (amounts > 300) * 0.25
        + (distances > 50) * 0.20
        + (velocities > 4) * 0.25
        + (flagged_ips == 1) * 0.30
    )
    fraud_prob = np.clip(fraud_prob, 0.01, 0.95)
    is_fraud = rng.binomial(1, fraud_prob)

    df = pd.DataFrame(
        {
            ID_COL: [f"TX{i:05d}" for i in range(n)],
            "transaction_amount": amounts,
            "location_distance_km": distances,
            "account_age_days": account_ages,
            "velocity_1h_count": velocities,
            "merchant_category": categories,
            "device_type": devices,
            "is_international": internationals,
            "is_flagged_ip": flagged_ips,
            TARGET_COL: is_fraud,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------
def train_fraud_model(version: str = "v1") -> None:
    log.info("═" * 60)
    log.info(
        "Starting training for Tenant #2 — model=%s version=%s", MODEL_NAME, version
    )
    log.info("═" * 60)

    # 1. Generate / load data
    df_raw = generate_fraud_dataset(n=600, seed=42)
    log.info("Generated %d synthetic fraud records.", len(df_raw))

    X = df_raw[ALL_FEATURES]
    y = df_raw[TARGET_COL]

    # 2. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Build & fit pipeline (using shared module)
    pipeline = build_pipeline(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        random_state=42,
    )
    pipeline.fit(X_train, y_train)

    # 4. Evaluate metrics
    metrics = evaluate_pipeline(pipeline, X_test, y_test)

    # 5. Compute snapshot baseline
    snapshot = compute_training_snapshot(
        X_train,
        pipeline,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )

    # 6. Save artifacts
    artifact_dir = ARTIFACTS_ROOT / MODEL_NAME / version
    artifact_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, artifact_dir / ARTIFACT_FILENAME)

    metadata = {
        "model_name": MODEL_NAME,
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "eval_metrics": metrics,
        "training_data_snapshot": snapshot,
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
    }
    with open(artifact_dir / METADATA_FILENAME, "w") as f:
        json.dump(metadata, f, indent=2)

    pointer_path = ARTIFACTS_ROOT / MODEL_NAME / CURRENT_VERSION_FILENAME
    with open(pointer_path, "w") as f:
        json.dump({"version": version}, f, indent=2)

    log.info("Saved artifact and pointer file for %s:%s", MODEL_NAME, version)

    # 7. Record deployment in DB
    try:
        db = SessionLocal()
        try:
            model_id = f"{MODEL_NAME}:{version}"
            d = Deployment(
                model_id=model_id,
                version=version,
                git_commit="unknown",
                eval_metrics=metrics,
                is_current=True,
                status="deployed",
                traffic_percentage=100.0,
                deployed_at=datetime.now(timezone.utc),
            )
            db.add(d)
            db.commit()
            log.info("Recorded deployment in DB: %s", model_id)
        except Exception:
            db.rollback()
            log.warning("Could not record deployment in DB (non-fatal)")
        finally:
            db.close()
    except Exception:
        log.warning("Database unavailable for recording deployment (non-fatal)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fraud Model (Tenant #2)")
    parser.add_argument("--version", type=str, default="v1", help="Model version tag")
    args = parser.parse_args()

    train_fraud_model(version=args.version)
