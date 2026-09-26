"""
Tests for Phase 45: Centralized API Error Handling, Request IDs, and Rate Limiting.

Covers:
1. Every custom exception type maps to its documented HTTP status code.
2. Unhandled exceptions never leak tracebacks or internal messages to clients.
3. The correlation/request ID appears in BOTH logs and response headers.
4. Rate limiting triggers with a 429 ErrorResponse after the configured threshold.
"""

import logging
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.rate_limit import RateLimitExceeded, reset_rate_limiters
from app.ingestion.exceptions import PermanentIngestionError, TransientIngestionError
from app.ingestion.pdf_extractor import PDFExtractionError
from app.main import app
from app.models.schemas import GenerationError

# raise_server_exceptions=False: Starlette's ServerErrorMiddleware re-raises the
# original exception after sending the 500 response (by design), so tests must
# opt out to assert on the delivered JSON body instead.
client = TestClient(app, raise_server_exceptions=False)

# ────────────────────────────────────────────────────────────────────────────
# 1. Explicit Exception -> HTTP Status Mapping
# ────────────────────────────────────────────────────────────────────────────


def _make_probe_app() -> FastAPI:
    """Build a minimal FastAPI app with the centralized handlers and probe routes."""
    probe_app = FastAPI()
    register_exception_handlers(probe_app)

    @probe_app.get("/probe/{exc_type}")
    async def probe(exc_type: str) -> dict:
        if exc_type == "permanent_ingestion":
            raise PermanentIngestionError("Corrupted PDF stream detected", stage="extracting")
        if exc_type == "transient_ingestion":
            raise TransientIngestionError("Pinecone connection timeout", stage="upserting")
        if exc_type == "generation":
            raise GenerationError("Unable to generate a reliable answer, please try rephrasing your question.")
        if exc_type == "pdf_extraction":
            raise PDFExtractionError("Password-protected PDF")
        if exc_type == "value":
            raise ValueError("Invalid input value")
        if exc_type == "rate_limit":
            raise RateLimitExceeded(limit=10, retry_after=12.3, client_key="127.0.0.1")
        raise RuntimeError(SECRET_INTERNAL_TOKEN)

    return probe_app


probe_client = TestClient(_make_probe_app(), raise_server_exceptions=False)

EXCEPTION_STATUS_MAP = {
    "permanent_ingestion": (422, "INGESTION_PERMANENT_ERROR"),
    "transient_ingestion": (503, "INGESTION_TRANSIENT_ERROR"),
    "generation": (503, "GENERATION_ERROR"),
    "pdf_extraction": (400, "PDF_EXTRACTION_ERROR"),
    "value": (400, "BAD_REQUEST"),
    "rate_limit": (429, "RATE_LIMIT_EXCEEDED"),
}


@pytest.mark.parametrize("exc_type,expected", EXCEPTION_STATUS_MAP.items())
def test_custom_exception_maps_to_documented_status(exc_type, expected):
    """Verify each custom exception maps to its documented HTTP status + error code."""
    expected_status, expected_error = expected
    res = probe_client.get(f"/probe/{exc_type}")
    assert res.status_code == expected_status
    body = res.json()
    assert body["error"] == expected_error
    assert body["status_code"] == expected_status
    assert isinstance(body["detail"], str) and body["detail"]


def test_rate_limit_response_carries_retry_after_header():
    """Verify the 429 rate-limit response includes a Retry-After header."""
    res = probe_client.get("/probe/rate_limit")
    assert res.status_code == 429
    assert res.headers["Retry-After"] == "13"


def test_unhandled_exception_does_not_leak_traceback_or_internals():
    """Verify unhandled exceptions return the generic 500 shape without leaking details."""
    res = probe_client.get("/probe/unhandled")
    assert res.status_code == 500
    body = res.json()
    assert body["error"] == "INTERNAL_SERVER_ERROR"
    assert body["detail"] == "An unexpected internal server error occurred."
    assert SECRET_INTERNAL_TOKEN not in res.text
    assert "Traceback" not in res.text


def test_unhandled_exception_on_real_query_endpoint_does_not_leak():
    """End-to-end: a crash inside the RAG graph returns generic 500, no internals."""
    with patch(
        "app.api.routes.query.app_graph.invoke",
        side_effect=RuntimeError(SECRET_INTERNAL_TOKEN),
    ):
        res = client.post("/api/v1/query", json={"query": "Test query", "stream": False})

    assert res.status_code == 500
    body = res.json()
    assert body["error"] == "INTERNAL_SERVER_ERROR"
    assert body["detail"] == "An unexpected internal server error occurred."
    assert SECRET_INTERNAL_TOKEN not in res.text


def test_generation_error_on_real_query_endpoint_returns_503():
    """End-to-end: GenerationError propagates from the route to the 503 handler."""
    with patch(
        "app.api.routes.query.app_graph.invoke",
        side_effect=GenerationError("LLM providers unavailable"),
    ):
        res = client.post("/api/v1/query", json={"query": "Test query", "stream": False})

    assert res.status_code == 503
    body = res.json()
    assert body["error"] == "GENERATION_ERROR"
    assert body["status_code"] == 503


# ────────────────────────────────────────────────────────────────────────────
# 2. Request ID in Logs AND Response Headers
# ────────────────────────────────────────────────────────────────────────────

def test_request_id_appears_in_both_logs_and_response_headers(caplog):
    """Verify the X-Correlation-ID is echoed in headers and written to request logs."""
    incoming_cid = "req-abc123456789"
    with caplog.at_level(logging.INFO, logger="app.core.middleware"):
        res = client.get("/health", headers={"X-Correlation-ID": incoming_cid})

    assert res.status_code == 200
    assert res.headers["X-Correlation-ID"] == incoming_cid

    assert incoming_cid[:8] in caplog.text
    assert "/health" in caplog.text
    assert "HTTP 200" in caplog.text


# ────────────────────────────────────────────────────────────────────────────
# 3. Rate Limiting
# ────────────────────────────────────────────────────────────────────────────

def test_rate_limiting_triggers_after_configured_threshold(monkeypatch):
    """Verify the 4th request within a minute is rejected with 429 + Retry-After."""
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_QUERY_PER_MINUTE", 3)
    reset_rate_limiters()

    # First 3 requests pass the limiter (and then fail body validation -> 422).
    for _ in range(3):
        res = client.post("/api/v1/query", json={"query": "   ", "stream": False})
        assert res.status_code == 422, res.text

    # 4th request is rejected by the rate limiter before validation.
    res = client.post("/api/v1/query", json={"query": "   ", "stream": False})
    assert res.status_code == 429
    body = res.json()
    assert body["error"] == "RATE_LIMIT_EXCEEDED"
    assert body["status_code"] == 429
    assert "3 queries per minute" in body["detail"]
    assert int(res.headers["Retry-After"]) >= 1

    # Non-limited endpoints are unaffected.
    res = client.get("/api/v1/documents")
    assert res.status_code == 200
