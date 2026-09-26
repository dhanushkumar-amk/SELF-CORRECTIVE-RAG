"""
Tests for Phase 44: Consolidated Public API Models (app/models/api_models.py).

Covers:
1. All public API models live in the dedicated api_models module (separate from
   internal domain models in schemas.py).
2. Uniform PascalCase naming conventions.
3. Realistic `examples` on key models validate and appear in the OpenAPI schema.
4. Versioned routes respond under /api/v1/...
"""

import re

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.models import api_models, schemas
from app.models.api_models import (
    DocumentUploadResponse,
    ErrorResponse,
    QueryRequest,
    QueryResponse,
)

client = TestClient(app)

API_MODEL_NAMES = [
    "HealthResponse",
    "ErrorResponse",
    "QueryRequest",
    "QueryResponse",
    "DocumentUploadResponse",
    "DocumentMetadata",
    "DocumentExtractionResponse",
    "ChunkListResponse",
    "DocumentListResponse",
    "IngestionResult",
    "DocumentStatusResponse",
    "DocumentVerificationResponse",
]


def test_api_models_use_pascal_case_naming():
    """Verify every public API model class follows PascalCase naming."""
    for name in API_MODEL_NAMES:
        assert re.fullmatch(r"[A-Z][A-Za-z0-9]*", name), f"{name} is not PascalCase"


def test_api_models_live_in_dedicated_module():
    """Verify public API models are defined in api_models.py, NOT schemas.py."""
    for name in API_MODEL_NAMES:
        cls = getattr(api_models, name)
        assert cls.__module__ == "app.models.api_models", f"{name} lives in {cls.__module__}"
        assert not hasattr(schemas, name), f"{name} still leaks into internal schemas.py"


def test_query_request_example_validates():
    """Verify the QueryRequest OpenAPI example is a valid payload."""
    example = QueryRequest.model_json_schema()["examples"][0]
    req = QueryRequest(**example)
    assert req.query == example["query"]
    assert req.stream is True


def test_query_response_example_validates():
    """Verify the QueryResponse OpenAPI example is a valid payload."""
    example = QueryResponse.model_json_schema()["examples"][0]
    res = QueryResponse(**example)
    assert res.final_status == "fully_verified"
    assert len(res.claims) == 1
    assert res.claims[0].verification_status is not None


def test_error_response_example_validates():
    """Verify the ErrorResponse OpenAPI example is a valid payload."""
    example = ErrorResponse.model_json_schema()["examples"][0]
    err = ErrorResponse(**example)
    assert err.error == "RATE_LIMIT_EXCEEDED"
    assert err.status_code == 429


def test_document_upload_response_example_validates():
    """Verify the DocumentUploadResponse OpenAPI example is a valid payload."""
    example = DocumentUploadResponse.model_json_schema()["examples"][0]
    upload = DocumentUploadResponse(**example)
    assert upload.status.value == "uploaded"
    assert upload.size_bytes > 0


def test_query_request_rejects_whitespace_only():
    """Verify QueryRequest validation rejects empty/whitespace queries."""
    with pytest.raises(ValidationError):
        QueryRequest(query="    ")


def test_versioned_routes_respond():
    """Verify the /api/v1/ surface responds with the documented shapes."""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

    res = client.get("/api/v1/documents")
    assert res.status_code == 200
    data = res.json()
    assert "documents" in data
    assert "total" in data

    res = client.post("/api/v1/query", json={"query": "   ", "stream": False})
    assert res.status_code == 422
    body = res.json()
    assert body["error"] == "VALIDATION_ERROR"
    assert body["status_code"] == 422


def test_openapi_documents_versioned_routes_with_examples():
    """Verify the generated OpenAPI schema exposes versioned routes and example payloads."""
    openapi_schema = app.openapi()
    paths = openapi_schema["paths"]

    assert "/api/v1/query" in paths
    assert "/api/v1/documents" in paths
    assert "/api/v1/documents/upload" in paths

    query_operation = paths["/api/v1/query"]["post"]
    request_schema = query_operation["requestBody"]["content"]["application/json"]["schema"]
    # The path operation references the model by $ref; the examples are
    # attached to the component schema in #/components/schemas/QueryRequest.
    schema_name = request_schema["$ref"].split("/")[-1]
    component_schema = openapi_schema["components"]["schemas"][schema_name]
    assert "examples" in component_schema
