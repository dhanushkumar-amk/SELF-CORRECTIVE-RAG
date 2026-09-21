"""
Centralized Exception Handlers for Phase 45: API Error Standardization.

Architecture & Design Decisions:
1. Standardized Error Contract (`ErrorResponse`):
   Guarantees that ALL exception pathways across the API (validation errors, HTTP exceptions,
   LLM generation errors, PDF extraction errors, value errors, and unhandled 500s)
   return a consistent `{ "error": str, "detail": str, "status_code": int }` JSON body.

2. Logging & Tracing:
   Logs exception details with severity levels appropriate for debugging without leaking internal stack traces to clients.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.ingestion.pdf_extractor import PDFExtractionError
from app.models.schemas import ErrorResponse, GenerationError

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

    @app.exception_handler(GenerationError)
    async def generation_error_handler(request: Request, exc: GenerationError) -> JSONResponse:
        logger.error(
            "GenerationError [502] on %s %s: %s",
            request.method,
            request.url.path,
            exc.message,
        )
        payload = ErrorResponse(
            error="GENERATION_ERROR",
            detail=exc.message,
            status_code=status.HTTP_502_BAD_GATEWAY,
        )
        return JSONResponse(status_code=status.HTTP_502_BAD_GATEWAY, content=payload.model_dump())

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
