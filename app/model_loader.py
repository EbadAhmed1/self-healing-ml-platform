"""
app/model_loader.py
────────────────────
Singleton and multi-tenant model loader with canary traffic splitting.

SUPPORTED TENANTS:
  - churn-model (Phase 1)
  - fraud-model (Phase 7)
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import joblib

from models.feature_config import (
    ARTIFACT_FILENAME,
    CANARY_FILENAME,
    CURRENT_VERSION_FILENAME,
    MODEL_NAME,
)

log = logging.getLogger(__name__)

# Default model cache (churn-model)
_pipeline: Any = None
_model_id: str = ""

# Canary cache for default model
_canary_pipeline: Any = None
_canary_model_id: str = ""
_canary_traffic_pct: float = 0.0

# Multi-tenant cache map: model_name -> (pipeline, model_id)
_tenant_pipelines: dict[str, tuple[Any, str]] = {}


def download_from_hf_hub(model_name: str, registry_path: Path) -> None:
    """Download model version pointer and artifact from Hugging Face Hub if configured."""
    from app.config import get_settings

    settings = get_settings()
    if not settings.use_hf_hub or not settings.hf_repo_id:
        return

    try:
        import shutil
        from huggingface_hub import hf_hub_download

        token = settings.hf_hub_token or None
        log.info(
            "Fetching pointer for %s from HF Hub: %s", model_name, settings.hf_repo_id
        )

        pointer_file = hf_hub_download(
            repo_id=settings.hf_repo_id,
            filename=f"{model_name}/{CURRENT_VERSION_FILENAME}",
            token=token,
        )
        target_dir = registry_path / model_name
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(pointer_file, target_dir / CURRENT_VERSION_FILENAME)

        with open(pointer_file) as f:
            pointer = json.load(f)
        version = pointer["version"]

        artifact_file = hf_hub_download(
            repo_id=settings.hf_repo_id,
            filename=f"{model_name}/{version}/{ARTIFACT_FILENAME}",
            token=token,
        )
        version_dir = target_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(artifact_file, version_dir / ARTIFACT_FILENAME)
        log.info("Successfully fetched %s:%s from HF Hub", model_name, version)
    except Exception as exc:
        log.warning(
            "Hugging Face Hub fetch failed (%s) — falling back to local files", exc
        )


def load_model(registry_path: Path, model_name: str = MODEL_NAME) -> tuple[Any, str]:
    """
    Load main model and optional canary model for a given model_name into memory.
    """
    global _pipeline, _model_id, _canary_pipeline, _canary_model_id, _canary_traffic_pct

    # Attempt Hugging Face Hub download if configured
    download_from_hf_hub(model_name, registry_path)

    pointer_path = registry_path / model_name / CURRENT_VERSION_FILENAME
    if not pointer_path.exists():
        raise RuntimeError(
            f"\n\nModel version pointer not found at: {pointer_path}\n"
            f"Run training script first for model '{model_name}'.\n"
        )

    with open(pointer_path) as f:
        pointer = json.load(f)
    version = pointer["version"]

    artifact_path = registry_path / model_name / version / ARTIFACT_FILENAME
    if not artifact_path.exists():
        raise RuntimeError(
            f"\n\nModel artifact not found at: {artifact_path}\n"
            f"The pointer file says version='{version}' but the .joblib file is missing.\n"
        )

    pipeline = joblib.load(artifact_path)
    model_id = f"{model_name}:{version}"
    log.info("Model loaded: %s from %s", model_id, artifact_path)

    _tenant_pipelines[model_name] = (pipeline, model_id)

    if model_name == MODEL_NAME:
        _pipeline = pipeline
        _model_id = model_id

        # Check for active Canary deployment on default model
        canary_pointer_path = registry_path / MODEL_NAME / CANARY_FILENAME
        if canary_pointer_path.exists():
            try:
                with open(canary_pointer_path) as f:
                    c_pointer = json.load(f)

                if c_pointer.get("status") == "active":
                    c_version = c_pointer["canary_version"]
                    c_artifact_path = (
                        registry_path / MODEL_NAME / c_version / ARTIFACT_FILENAME
                    )
                    if c_artifact_path.exists():
                        _canary_pipeline = joblib.load(c_artifact_path)
                        _canary_model_id = f"{MODEL_NAME}:{c_version}"
                        _canary_traffic_pct = float(
                            c_pointer.get("traffic_percentage", 10.0)
                        )
                        log.info(
                            "Canary model loaded: %s with %.1f%% traffic allocation.",
                            _canary_model_id,
                            _canary_traffic_pct,
                        )
            except Exception as exc:
                log.warning("Failed to load canary pointer: %s", exc)

    return pipeline, model_id


def load_all_models(registry_path: Path) -> dict[str, tuple[Any, str]]:
    """
    Startup loader to load all available tenant models in the registry.
    """
    load_model(registry_path, "churn-model")
    try:
        load_model(registry_path, "fraud-model")
    except Exception as exc:
        log.info("Optional tenant 'fraud-model' not loaded: %s", exc)
    return _tenant_pipelines


def get_pipeline() -> Any:
    """Return main pipeline (churn-model default)."""
    if _pipeline is None:
        raise RuntimeError("Model pipeline has not been loaded.")
    return _pipeline


def get_model_id() -> str:
    """Return main model_id string (churn-model default)."""
    return _model_id


def get_pipeline_for_request() -> tuple[Any, str]:
    """Select pipeline for churn-model (supports canary routing)."""
    if _pipeline is None:
        raise RuntimeError("Model pipeline has not been loaded.")

    if _canary_pipeline is not None and _canary_traffic_pct > 0:
        roll = random.uniform(0.0, 100.0)
        if roll < _canary_traffic_pct:
            return _canary_pipeline, _canary_model_id

    return _pipeline, _model_id


def get_pipeline_for_tenant(model_name: str) -> tuple[Any, str]:
    """
    Retrieve loaded pipeline and model_id for any registered tenant model name.
    """
    if model_name not in _tenant_pipelines:
        raise RuntimeError(f"Model '{model_name}' has not been loaded into memory.")
    return _tenant_pipelines[model_name]
