"""
Self-Correcting RAG with Hallucination Detection — FastAPI Entrypoint.

This module bootstraps the FastAPI application, registers routers,
and configures CORS middleware.
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import debug, health, ingest, query
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan — runs on startup and shutdown."""
    logger.info(
        "Starting %s v%s",
        settings.PROJECT_NAME,
        settings.VERSION,
    )
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Self-Correcting RAG with Hallucination Detection",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow the Next.js frontend during local development
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(health.router, tags=["health"])
app.include_router(ingest.router, prefix="/api/ingest", tags=["ingestion"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingestion"], include_in_schema=False)
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(debug.router, prefix="/debug", tags=["debug"])



