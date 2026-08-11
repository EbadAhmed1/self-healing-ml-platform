"""
tests/test_training.py
───────────────────────
Tests for the training script end-to-end behaviour.

These tests run the full training pipeline on a tiny 50-row CSV so they
complete in < 2 seconds. They verify:
  - The script runs without raising exceptions
  - The artifact (pipeline.joblib) is created on disk
  - The metadata JSON contains the required keys
  - The training data snapshot has the expected structure
"""

from __future__ import annotations

import json

import pytest


class TestTrainingEndToEnd:
    def test_train_runs_without_error(self, tiny_csv, tmp_path, monkeypatch):
        """Training script must complete without raising on a valid dataset."""
        monkeypatch.setattr("models.train.ARTIFACTS_ROOT", tmp_path / "artifacts")
        from models.train import train

        train(data_path=tiny_csv, version="test-v1")

    def test_artifact_file_created(self, tiny_csv, tmp_path, monkeypatch):
        """pipeline.joblib must exist after training."""
        from models.train import train

        # Redirect artifacts to tmp_path so tests don't pollute the real registry
        monkeypatch.setattr("models.train.ARTIFACTS_ROOT", tmp_path / "artifacts")
        train(data_path=tiny_csv, version="test-v1")

        artifact = (
            tmp_path / "artifacts" / "churn-model" / "test-v1" / "pipeline.joblib"
        )
        assert artifact.exists(), f"Expected artifact at {artifact}"

    def test_metadata_json_created(self, tiny_csv, tmp_path, monkeypatch):
        """metadata.json must exist after training."""
        monkeypatch.setattr("models.train.ARTIFACTS_ROOT", tmp_path / "artifacts")
        from models.train import train

        train(data_path=tiny_csv, version="test-v1")

        metadata_path = (
            tmp_path / "artifacts" / "churn-model" / "test-v1" / "metadata.json"
        )
        assert metadata_path.exists(), f"Expected metadata at {metadata_path}"

    def test_metadata_has_required_keys(self, tiny_csv, tmp_path, monkeypatch):
        """metadata.json must contain all keys consumed by the serving layer."""
        monkeypatch.setattr("models.train.ARTIFACTS_ROOT", tmp_path / "artifacts")
        from models.train import train

        train(data_path=tiny_csv, version="test-v1")

        metadata_path = (
            tmp_path / "artifacts" / "churn-model" / "test-v1" / "metadata.json"
        )
        with open(metadata_path) as f:
            meta = json.load(f)

        required_keys = {
            "model_name",
            "version",
            "trained_at",
            "git_commit",
            "eval_metrics",
            "training_data_snapshot",
            "features",
            "n_train",
            "n_test",
        }
        missing = required_keys - set(meta.keys())
        assert not missing, f"Missing metadata keys: {missing}"

    def test_eval_metrics_has_required_fields(self, tiny_csv, tmp_path, monkeypatch):
        """eval_metrics must contain precision, recall, f1, roc_auc."""
        monkeypatch.setattr("models.train.ARTIFACTS_ROOT", tmp_path / "artifacts")
        from models.train import train

        train(data_path=tiny_csv, version="test-v1")

        metadata_path = (
            tmp_path / "artifacts" / "churn-model" / "test-v1" / "metadata.json"
        )
        with open(metadata_path) as f:
            meta = json.load(f)

        required_metric_keys = {"precision", "recall", "f1", "roc_auc"}
        missing = required_metric_keys - set(meta["eval_metrics"].keys())
        assert not missing, f"Missing metric keys: {missing}"

    def test_training_snapshot_has_numeric_and_categorical(
        self, tiny_csv, tmp_path, monkeypatch
    ):
        """The drift-detection snapshot must have 'numeric' and 'categorical' keys."""
        monkeypatch.setattr("models.train.ARTIFACTS_ROOT", tmp_path / "artifacts")
        from models.train import train

        train(data_path=tiny_csv, version="test-v1")

        metadata_path = (
            tmp_path / "artifacts" / "churn-model" / "test-v1" / "metadata.json"
        )
        with open(metadata_path) as f:
            meta = json.load(f)

        snapshot = meta["training_data_snapshot"]
        assert "numeric" in snapshot
        assert "categorical" in snapshot

    def test_current_pointer_updated(self, tiny_csv, tmp_path, monkeypatch):
        """current.json pointer must be written after training."""
        monkeypatch.setattr("models.train.ARTIFACTS_ROOT", tmp_path / "artifacts")
        from models.train import train

        train(data_path=tiny_csv, version="test-v1")

        pointer_path = tmp_path / "artifacts" / "churn-model" / "current.json"
        assert pointer_path.exists()

        with open(pointer_path) as f:
            pointer = json.load(f)
        assert pointer["version"] == "test-v1"

    def test_missing_data_file_raises_file_not_found(self, tmp_path):
        """Training script must raise FileNotFoundError (not a cryptic KeyError)
        when the input CSV is missing."""
        from models.train import train

        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            train(data_path=tmp_path / "nonexistent.csv", version="test-v1")

    def test_pipeline_can_predict_after_training(self, tiny_csv, tmp_path, monkeypatch):
        """The saved pipeline must be loadable and produce predictions."""
        import joblib

        monkeypatch.setattr("models.train.ARTIFACTS_ROOT", tmp_path / "artifacts")
        from models.train import train

        train(data_path=tiny_csv, version="test-v1")

        pipeline_path = (
            tmp_path / "artifacts" / "churn-model" / "test-v1" / "pipeline.joblib"
        )
        pipeline = joblib.load(pipeline_path)

        # Build a single-row DataFrame matching feature order
        import pandas as pd
        from models.feature_config import (
            ALL_FEATURES,
            CATEGORICAL_FEATURES,
            NUMERIC_FEATURES,
        )

        row = {col: 0 for col in NUMERIC_FEATURES}
        row.update({col: "No" for col in CATEGORICAL_FEATURES})
        row["gender"] = "Male"
        row["InternetService"] = "DSL"
        row["Contract"] = "Month-to-month"
        row["PaymentMethod"] = "Mailed check"
        row["MultipleLines"] = "No phone service"
        row["OnlineSecurity"] = "No internet service"
        row["OnlineBackup"] = "No internet service"
        row["DeviceProtection"] = "No internet service"
        row["TechSupport"] = "No internet service"
        row["StreamingTV"] = "No internet service"
        row["StreamingMovies"] = "No internet service"

        X = pd.DataFrame([row])[ALL_FEATURES]
        pred = pipeline.predict(X)
        prob = pipeline.predict_proba(X)

        assert len(pred) == 1
        assert pred[0] in (0, 1)
        assert prob.shape == (1, 2)
        assert abs(prob[0].sum() - 1.0) < 1e-6
