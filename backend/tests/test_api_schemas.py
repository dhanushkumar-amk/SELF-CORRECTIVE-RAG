"""
Unit tests for Phase 44: Pydantic API Request/Response Schemas and OpenAPI specification.
"""

import pytest
from pydantic import ValidationError

from app.main import app
from app.models.api_models import ErrorResponse, QueryRequest, QueryResponse


def test_query_request_whitespace_trimming_and_validation():
    """Verify QueryRequest trims whitespace and rejects empty/whitespace-only queries."""
    # Trims leading/trailing whitespace
    req = QueryRequest(query="   What is the revenue?   ")
    assert req.query == "What is the revenue?"

    # Rejects empty / whitespace string
    with pytest.raises(ValidationError):
        QueryRequest(query="    ")


def test_error_response_schema():
    """Verify ErrorResponse model fields and serialization."""
    err = ErrorResponse(
        error="VALIDATION_ERROR",
        detail="Invalid request body",
        status_code=422,
    )
    data = err.model_dump()
    assert data["error"] == "VALIDATION_ERROR"
    assert data["status_code"] == 422


def test_query_response_schema():
    """Verify QueryResponse model defaults and serialization."""
    res = QueryResponse(
        query="Test query",
        final_status="fully_verified",
        final_answer_text="Test answer.",
    )
    data = res.model_dump()
    assert data["query"] == "Test query"
    assert data["final_status"] == "fully_verified"
    assert data["retry_count"] == 0
    assert data["latency_ms"] == 0.0


def test_openapi_schema_generation():
    """Verify FastAPI generates a valid OpenAPI v3 schema dictionary containing /api/v1 routes."""
    openapi_schema = app.openapi()
    assert openapi_schema is not None
    assert "paths" in openapi_schema
    assert "/api/v1/query" in openapi_schema["paths"]
    assert "/api/v1/documents/upload" in openapi_schema["paths"]
