"""
Correlation ID & Request Timing Middleware for Phase 45.

Architecture & Design Decisions:
1. Distributed Request Tracing (`X-Correlation-ID`):
   - Generates a unique UUID4 correlation ID for every incoming HTTP request if not present.
   - Propagates `X-Correlation-ID` to response headers for end-to-end client observability.

2. Performance Monitoring (`X-Process-Time-Ms`):
   - Measures exact server-side wall-clock request duration in milliseconds.
   - Attaches `X-Process-Time-Ms` to response headers.
"""

from __future__ import annotations

import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.core.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "CorrelationIdAndTimingMiddleware",
]


class CorrelationIdAndTimingMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for request correlation ID tracing and execution timing."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.perf_counter()

        # 1. Extract or generate unique UUID4 correlation ID
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        request.state.correlation_id = correlation_id

        # 2. Process request
        response: Response = await call_next(request)

        # 3. Calculate latency duration in milliseconds
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # 4. Attach telemetry headers
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"

        logger.info(
            "%s %s -> HTTP %d (Duration: %.2f ms, CID: %s)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            correlation_id[:8],
        )

        return response
