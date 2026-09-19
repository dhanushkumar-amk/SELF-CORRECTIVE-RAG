"""
Self-Correcting RAG with Hallucination Detection — FastAPI Entrypoint.

This module bootstraps the FastAPI application, registers routers,
and configures CORS middleware.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, ingest, query
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Self-Correcting RAG with Hallucination Detection",
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
app.include_router(ingest.router, prefix="/ingest", tags=["ingestion"])
app.include_router(query.router, prefix="/query", tags=["query"])


@app.on_event("startup")
async def startup_event() -> None:
    logger.info(
        "Starting %s v%s",
        settings.PROJECT_NAME,
        settings.VERSION,
    )
