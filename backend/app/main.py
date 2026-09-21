"""
Self-Correcting RAG with Hallucination Detection — FastAPI Entrypoint.

This module bootstraps the FastAPI application, registers API routers,
and configures CORS middleware.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import debug, health, ingest, query
from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval import get_bm25_index

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan — runs on startup and shutdown."""
    logger.info(
        "Starting %s v%s (Environment: %s)",
        settings.PROJECT_NAME,
        settings.VERSION,
        settings.ENVIRONMENT,
    )
    # Initialize in-memory BM25 index on startup from stored READY chunks
    try:
        bm25_idx = get_bm25_index()
        logger.info(
            "BM25 index initialized on startup (%d chunks indexed in %.4fs)",
            bm25_idx.chunk_count,
            bm25_idx.build_time_seconds,
        )
    except Exception as exc:
        logger.warning("Failed to initialize BM25 index on startup: %s", exc)

    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-grade Self-Correcting RAG with Hallucination Detection & SSE Streaming",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS Middleware
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Core Production API Surface (/api/v1)
# ---------------------------------------------------------------------------
# Health endpoint
app.include_router(health.router, tags=["Health"])

# RAG Query & SSE Streaming Endpoint
app.include_router(query.router, prefix="/api/v1/query", tags=["Query Engine"])
app.include_router(query.router, prefix="/api/query", tags=["Query Engine"], include_in_schema=False)
app.include_router(query.router, prefix="/query", tags=["Query Engine"], include_in_schema=False)

# Document Ingestion & Lifecycle Management
app.include_router(ingest.router, prefix="/api/v1/documents", tags=["Document Storage"])
app.include_router(ingest.router, prefix="/api/ingest", tags=["Document Storage"], include_in_schema=False)
app.include_router(ingest.router, prefix="/ingest", tags=["Document Storage"], include_in_schema=False)

# Debug & Verification Endpoints (development only in schema)
app.include_router(
    debug.router,
    prefix="/debug",
    tags=["Debug & Diagnostics"],
    include_in_schema=(settings.ENVIRONMENT == "development"),
)
