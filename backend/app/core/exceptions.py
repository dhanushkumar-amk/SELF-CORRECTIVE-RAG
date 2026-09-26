"""
Centralized Exception Handlers for Phase 45: API Error Standardization.

Architecture & Design Decisions:
1. Standardized Error Contract (`ErrorResponse`):
   Guarantees that ALL exception pathways across the API (validation errors, HTTP exceptions,
   LLM generation errors, ingestion errors, rate limiting, and unhandled 500s)
   return a consistent `{ "error": str, "detail": str, "status_code": int }` JSON body.

2. Explicit Exception -> HTTP Status Mapping:
   Every project-specific exception type maps to a deliberate status code rather than
   defaulting to 500:

   +----------------------------+---------+---------------------------+
   | Exception                  | Status  | Error Code                |
   +============================+=========+===========================+
   | HTTPException              | passthru| HTTP_ERROR                |
   | RequestValidationError     | 422     | VALIDATION_ERROR          |
   | PDFExtractionError         | 400     | PDF_EXTRACTION_ERROR      |
   | ValueError                 | 400     | BAD_REQUEST               |
   | RateLimitExceeded          | 429     | RATE_LIMIT_EXCEEDED       |
   | PermanentIngestionError    | 422     | INGESTION_PERMANENT_ERROR |
   | TransientIngestionError    | 503     | INGESTION_TRANSIENT_ERROR |
   | GenerationError            | 503     | GENERATION_ERROR          |
   | Exception (unhandled)      | 500     | INTERNAL_SERVER_ERROR     |
   +----------------------------+---------+---------------------------+

   Document-not-found is surfaced as HTTPException(404) by the routes themselves.

3. Logging & Tracing:
   Logs exception details with severity levels appropriate for debugging WITHOUT leaking
   internal stack traces or exception messages to API clients. The unhandled 500 path
   returns a generic message by design.
"""

from __future__ import annotations

from math import ceil

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.core.rate_limit import RateLimitExceeded
from app.ingestion.exceptions import PermanentIngestionError, TransientIngestionError
from app.ingestion.pdf_extractor import PDFExtractionError
from app.models.api_models import ErrorResponse
from app.models.schemas import GenerationError

logger = get_logger(__name__)

__all__ = [
    "register_exception_handlers",
]


def register_exception_handlers(app: FastAPI) -> None:
    """Register centralized FastAPI exception handlers to enforce standard ErrorResponse output.

    Args:
        app: Bootstrapped FastAPI application instance.
    """

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        logger.warning(
            "HTTPException [%d] on %s %s: %s",
            exc.status_code,
            request.method,
            request.url.path,
            exc.detail,
        )
        payload = ErrorResponse(
            error="HTTP_ERROR",
            detail=str(exc.detail),
            status_code=exc.status_code,
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        err_msg = "; ".join(f"{e.get('loc', [])}: {e.get('msg', '')}" for e in exc.errors())
        logger.warning(
            "RequestValidationError [422] on %s %s: %s",
            request.method,
            request.url.path,
            err_msg,
        )
        payload = ErrorResponse(
            error="VALIDATION_ERROR",
            detail=f"Request validation failed: {err_msg}",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=payload.model_dump())

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        retry_after = max(1, ceil(exc.retry_after))
        logger.warning(
            "RateLimitExceeded [429] on %s %s (client='%s', limit=%d/min).",
            request.method,
            request.url.path,
            exc.client_key,
            exc.limit,
        )
        payload = ErrorResponse(
            error="RATE_LIMIT_EXCEEDED",
            detail=(
                f"Rate limit exceeded: maximum {exc.limit} queries per minute per client. "
                f"Retry after {retry_after} seconds."
            ),
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        )
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=payload.model_dump(),
            headers={"Retry-After": str(retry_after)},
        )

    @app.exception_handler(PermanentIngestionError)
    async def permanent_ingestion_error_handler(request: Request, exc: PermanentIngestionError) -> JSONResponse:
        logger.error(
            "PermanentIngestionError [422] on %s %s: %s",
            request.method,
            request.url.path,
            exc,
        )
        payload = ErrorResponse(
            error="INGESTION_PERMANENT_ERROR",
            detail=str(exc),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=payload.model_dump())

    @app.exception_handler(TransientIngestionError)
    async def transient_ingestion_error_handler(request: Request, exc: TransientIngestionError) -> JSONResponse:
        logger.error(
            "TransientIngestionError [503] on %s %s: %s",
            request.method,
            request.url.path,
            exc,
        )
        payload = ErrorResponse(
            error="INGESTION_TRANSIENT_ERROR",
            detail=str(exc),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload.model_dump())

    @app.exception_handler(GenerationError)
    async def generation_error_handler(request: Request, exc: GenerationError) -> JSONResponse:
        logger.error(
            "GenerationError [503] on %s %s: %s",
            request.method,
            request.url.path,
            exc.message,
        )
        payload = ErrorResponse(
            error="GENERATION_ERROR",
            detail=exc.message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload.model_dump())

    @app.exception_handler(PDFExtractionError)
    async def pdf_extraction_error_handler(request: Request, exc: PDFExtractionError) -> JSONResponse:
        logger.warning(
            "PDFExtractionError [400] on %s %s: %s",
            request.method,
            request.url.path,
            exc,
        )
        payload = ErrorResponse(
            error="PDF_EXTRACTION_ERROR",
            detail=str(exc),
            status_code=status.HTTP_400_BAD_REQUEST,
        )
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=payload.model_dump())

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        logger.warning(
            "ValueError [400] on %s %s: %s",
            request.method,
            request.url.path,
            exc,
        )
        payload = ErrorResponse(
            error="BAD_REQUEST",
            detail=str(exc),
            status_code=status.HTTP_400_BAD_REQUEST,
        )
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "Unhandled Internal Error [500] on %s %s: %s",
            request.method,
            request.url.path,
            exc,
            exc_info=True,
        )
        payload = ErrorResponse(
            error="INTERNAL_SERVER_ERROR",
            detail="An unexpected internal server error occurred.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=payload.model_dump(),
        )
