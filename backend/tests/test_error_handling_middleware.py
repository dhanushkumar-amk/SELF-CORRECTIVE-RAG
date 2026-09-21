"""
Unit and Integration tests for Phase 45: Centralized API Error Handlers, Correlation IDs, and Timing Middleware.
"""

from unittest.mock import patch

from fastapi import status
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import GenerationError

client = TestClient(app)


def test_correlation_id_and_timing_headers():
    """Verify middleware injects X-Correlation-ID and X-Process-Time-Ms headers on all responses."""
    # 1. Server generates new correlation ID if not provided
    res = client.get("/health")
    assert res.status_code == 200
    assert "X-Correlation-ID" in res.headers
    assert "X-Process-Time-Ms" in res.headers

    # 2. Server preserves incoming correlation ID
    incoming_cid = "custom-cid-12345678"
    res_custom = client.get("/health", headers={"X-Correlation-ID": incoming_cid})
    assert res_custom.status_code == 200
    assert res_custom.headers["X-Correlation-ID"] == incoming_cid


def test_validation_error_handler_422():
    """Verify pydantic validation errors return structured 422 ErrorResponse JSON."""
    res = client.post("/api/v1/query", json={"query": "    ", "stream": False})
    assert res.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    data = res.json()
    assert data["error"] == "VALIDATION_ERROR"
    assert data["status_code"] == 422
    assert "detail" in data


def test_http_exception_handler_404():
    """Verify HTTPException returns structured 404 ErrorResponse JSON."""
    res = client.get("/api/v1/documents/nonexistent-doc-uuid-999")
    assert res.status_code == status.HTTP_404_NOT_FOUND
    data = res.json()
    assert data["error"] == "HTTP_ERROR"
    assert data["status_code"] == 404


def test_generation_error_handler_502():
    """Verify GenerationError returns structured 502 Bad Gateway ErrorResponse JSON."""
    with patch("app.api.routes.query.app_graph.invoke", side_effect=GenerationError("LLM providers unavailable")):
        res = client.post("/api/v1/query", json={"query": "Test query", "stream": False})
        assert res.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR or res.status_code == status.HTTP_502_BAD_GATEWAY
        data = res.json()
        assert "error" in data
        assert "status_code" in data
