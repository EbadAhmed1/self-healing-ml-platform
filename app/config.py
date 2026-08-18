"""
app/config.py
─────────────
Centralised settings loaded from the .env file via pydantic-settings.

All environment variables are read ONCE here and injected everywhere else
through dependency injection — no os.getenv() calls scattered across the
codebase.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore unknown env vars (LLM keys etc added later)
        protected_namespaces=(),  # suppress warning for model_registry_path field
    )

    # Database
    database_url: str = "postgresql://mluser:mlpassword@localhost:5432/mlplatform"

    # Model registry
    model_registry_path: Path = Path("./models/artifacts")

    # Logging
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # LLM / Diagnosis Agent (Phase 5)
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    # Hugging Face Hub Model Registry (Phase 9 Cloud Registry)
    use_hf_hub: bool = False
    hf_repo_id: str = ""
    hf_hub_token: str = ""

    # Alerts & Webhooks (Phase 9)
    slack_webhook_url: str = ""
    api_base_url: str = "http://localhost:8000"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton.  Use as a FastAPI dependency."""
    return Settings()
