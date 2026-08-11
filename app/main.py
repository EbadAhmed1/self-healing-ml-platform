"""
app/main.py
───────────
FastAPI application entry point.

STARTUP BEHAVIOUR (lifespan):
  The model artifact is loaded synchronously during lifespan startup.
  If the artifact is missing, RuntimeError is raised BEFORE uvicorn marks
  the app as healthy — the process exits with a clear error, preventing the
  silent-failure mode where the app starts but every prediction returns 500.

ROUTES:
  GET  /health                → liveness + model status
  POST /predict/churn-model   → churn prediction (see routers/churn.py)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app import model_loader
from app.routers import churn

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load the model artifact — fail-fast if missing.
    Shutdown: nothing to clean up in Phase 1.
    """
    log.info("Starting up — loading model artifact…")
    registry_path = settings.model_registry_path
    # Resolve relative paths against CWD (where uvicorn is launched from)
    if not registry_path.is_absolute():
        from pathlib import Path

        registry_path = Path.cwd() / registry_path

    model_loader.load_all_models(
        registry_path
    )  # raises RuntimeError if primary artifact missing
    log.info(
        "Startup complete. Serving models: %s",
        list(model_loader._tenant_pipelines.keys()),
    )
    yield
    log.info("Shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Self-Healing ML Serving Platform",
    description=(
        "Multi-Tenant ML serving platform supporting churn-model and fraud-model. "
        "Namespaced endpoints, versioned model artifacts, Postgres prediction logging."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
from app.routers import fraud  # noqa: E402

app.include_router(churn.router)
app.include_router(fraud.router)


@app.get("/health", tags=["ops"], summary="Liveness check")
def health():
    """Returns 200 with model status. Safe to call before the model is loaded."""
    from app.schemas import HealthResponse

    loaded = model_loader._pipeline is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_id=model_loader.get_model_id() if loaded else None,
    )
