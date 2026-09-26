"""
RAG Query Endpoints for Phase 42 & 43: Non-Streaming & SSE Real-Time Streaming API.

Architecture & Design Decisions:
1. Unified API Entrypoint (`POST /api/v1/query`):
   - Accepts `QueryRequest` payload containing `query`, optional `document_id` filter, and `stream` boolean flag.
   - Non-Streaming Mode (`stream=False`): Executes full compiled LangGraph state machine, returning a single `QueryResponse` JSON object.
   - SSE Streaming Mode (`stream=True`): Returns a `StreamingResponse` with `media_type="text/event-stream"`, pushing real-time progress events as graph node transitions occur.

2. Document Scoping:
   - If `document_id` is supplied, hybrid retrieval is scoped strictly to vectors matching that parent document UUID.
"""

from __future__ import annotations

import time
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.query_stream import stream_query_execution
from app.core.config import settings
from app.core.logging import get_logger
from app.core.rate_limit import rate_limit_query
from app.graph.graph import app_graph
from app.graph.state import RAGState
from app.models.api_models import ErrorResponse, QueryRequest, QueryResponse

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid query input"},
        422: {"model": ErrorResponse, "description": "Request validation failed"},
        429: {"model": ErrorResponse, "description": "Per-client rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Unexpected pipeline execution error"},
        503: {"model": ErrorResponse, "description": "LLM generation unavailable"},
    },
    status_code=status.HTTP_200_OK,
    summary="Execute a self-correcting RAG query (supports JSON & SSE streaming)",
    dependencies=[Depends(rate_limit_query)],
)
async def query_rag(request: QueryRequest) -> QueryResponse | StreamingResponse:
    """Execute natural language query through the self-correcting RAG state machine.

    Args:
        request: QueryRequest payload (`query`, `document_id`, `stream`).

    Returns:
        QueryResponse JSON object if `stream=False`, or StreamingResponse (text/event-stream) if `stream=True`.
    """
    if not request.query or not request.query.strip():
        logger.warning("Query endpoint rejected: Empty or whitespace query.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty or whitespace-only.",
        )

    # 1. SSE Real-Time Streaming Mode
    if request.stream:
        logger.info(
            "Executing SSE streaming query ('%s', document_id=%s)...",
            request.query[:50],
            request.document_id,
        )
        return StreamingResponse(
            stream_query_execution(
                query=request.query,
                document_id=request.document_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # 2. Non-Streaming JSON Response Mode
    logger.info(
        "Executing non-streaming query ('%s', document_id=%s)...",
        request.query[:50],
        request.document_id,
    )
    start_t = time.perf_counter()

    initial_state: RAGState = {
        "query": request.query,
        "document_id": request.document_id,
        "retry_count": 0,
        "max_retries": settings.CORRECTION_MAX_RETRIES,
    }

    # Exceptions propagate to the centralized handlers in app.core.exceptions:
    # GenerationError -> 503, unexpected failures -> generic 500 (no leakage).
    final_state = app_graph.invoke(initial_state)
    duration_ms = (time.perf_counter() - start_t) * 1000.0

    return QueryResponse(
        query=request.query,
        final_status=final_state.get("final_status", "unverifiable"),
        final_answer_text=final_state.get("final_answer_text", ""),
        claims=final_state.get("claims", []),
        retry_count=final_state.get("retry_count", 0),
        retrieved_chunks=final_state.get("retrieved_chunks", []),
        latency_ms=round(duration_ms, 2),
    )
