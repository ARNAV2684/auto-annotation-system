"""FastAPI application entry point.

Run locally::

    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

* ``CORS`` is opened for the Vite dev origin (``http://localhost:5173`` by default
  or whatever ``FRONTEND_ORIGIN`` env var sets).
* The ``lifespan`` hook loads SAM 2 + GroundingDINO **once** at startup so the
  first inference request is fast. CI mocks this away via dependency overrides.
* Interactive docs land at ``/docs`` (Swagger UI) and ``/redoc``.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from utils.config import REPO_ROOT

logger = logging.getLogger(__name__)

load_dotenv(REPO_ROOT / ".env")

_FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
_LOAD_MODELS_ON_STARTUP = os.getenv("LOAD_MODELS_ON_STARTUP", "true").lower() not in {
    "0", "false", "no",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the GPU models once at app startup; release on shutdown."""
    if _LOAD_MODELS_ON_STARTUP:
        # Import lazily so test setups that override the dependency never trigger
        # torch / sam2 / transformers imports.
        from app.services.model_registry import init_model_registry

        logger.info("Initialising model registry ...")
        init_model_registry()
    else:
        logger.info("LOAD_MODELS_ON_STARTUP=false — skipping model load.")
    yield
    logger.info("FastAPI shutdown complete.")


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    application = FastAPI(
        title="Auto Annotation System API",
        version="0.4.0",
        summary="GroundingDINO + SAM 2 backend for drone-image annotation.",
        description=(
            "REST API documented in `docs/api_contract.md`. Single-user, "
            "localhost-only for v1; CORS allows the Vite dev server origin."
        ),
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=[_FRONTEND_ORIGIN],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(router)

    @application.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "name": "auto-annotation-system",
            "docs": "/docs",
            "api_health": "/api/health",
        }

    return application


app = create_app()
