"""
Unit and Integration tests for Phase 42: POST /api/v1/query FastAPI route.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import Claim, GeneratedAnswer, RerankResult, RetrievalResult, VerificationStatus

client = TestClient(app)


def test_query_endpoint_empty_query_rejected():
    """Verify POST /api/v1/query returns HTTP 422/400 Bad Request when query is empty."""
    response = client.post("/api/v1/query", json={"query": "", "stream": False})
    assert response.status_code in (400, 422)


def test_query_endpoint_non_streaming_success():
    """Verify POST /api/v1/query with stream=False returns structured QueryResponse JSON."""
    chunk = RetrievalResult(
        chunk_id="chunk-test-1",
        score=0.9,
        metadata={"source_text": "SentenceTransformers embeddings are 384 dimensional.", "document_id": "doc-1", "page_number": 1},
    )

    answer = GeneratedAnswer(
        claims=[Claim(claim_text="SentenceTransformers embeddings are 384 dimensional.", source_chunk_id="chunk-test-1")],
        insufficient_information=False,
    )

    with patch("app.graph.graph.hybrid_search", return_value=[chunk]), \
         patch("app.graph.graph.select_relevant_chunks", return_value=RerankResult(chunks=[chunk], relevant_count=1)), \
         patch("app.graph.graph.generate_answer_with_citations", return_value=answer):

        response = client.post(
            "/api/v1/query",
            json={"query": "What dimension are sentence transformer embeddings?", "stream": False},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What dimension are sentence transformer embeddings?"
        assert data["final_status"] in ("verified", "fully_verified")
        assert "SentenceTransformers embeddings are 384 dimensional." in data["final_answer_text"]
        assert len(data["claims"]) == 1
        assert data["claims"][0]["verification_status"] == VerificationStatus.ENTAILED.value
        assert data["retry_count"] == 0
        assert len(data["retrieved_chunks"]) == 1


def test_query_endpoint_document_id_scoping():
    """Verify document_id parameter is passed into hybrid_search filter."""
    with patch("app.graph.graph.hybrid_search", return_value=[]) as mock_hybrid, \
         patch("app.graph.graph.select_relevant_chunks", return_value=RerankResult(chunks=[], relevant_count=0)):

        response = client.post(
            "/api/v1/query",
            json={"query": "Financial query", "document_id": "doc-uuid-1234", "stream": False},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["final_status"] == "unverifiable"
        mock_hybrid.assert_called_once_with(query="Financial query", filter={"document_id": "doc-uuid-1234"})


def test_query_endpoint_no_relevant_documents():
    """Verify query with no context chunks returns unverifiable status."""
    with patch("app.graph.graph.hybrid_search", return_value=[]), \
         patch("app.graph.graph.select_relevant_chunks", return_value=RerankResult(chunks=[], relevant_count=0)):

        response = client.post(
            "/api/v1/query",
            json={"query": "Query with zero matching documents", "stream": False},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["final_status"] == "unverifiable"
        assert len(data["claims"]) == 0
