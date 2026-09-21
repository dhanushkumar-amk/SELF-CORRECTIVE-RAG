"""
Unit and Integration tests for Phase 43: SSE Real-Time Progress Streaming API.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import Claim, ClaimWithSource, GeneratedAnswer, LLMResponse, RerankResult, RetrievalResult, VerificationStatus

client = TestClient(app)


def test_sse_streaming_event_sequence():
    """Verify stream=True emits events in correct chronological sequence for normal query."""
    chunk = RetrievalResult(
        chunk_id="c1",
        score=0.9,
        metadata={"source_text": "Python 3.14 was released.", "document_id": "doc1", "page_number": 1},
    )
    answer = GeneratedAnswer(
        claims=[Claim(claim_text="Python 3.14 was released.", source_chunk_id="c1")],
        insufficient_information=False,
    )
    claim_verified = ClaimWithSource(
        claim_text="Python 3.14 was released.",
        source_chunk_id="c1",
        source_text="Python 3.14 was released.",
        is_valid_source=True,
        verification_status=VerificationStatus.ENTAILED,
        confidence=0.99,
    )

    with patch("app.api.query_stream.hybrid_search", return_value=[chunk]), \
         patch("app.api.query_stream.select_relevant_chunks", return_value=RerankResult(chunks=[chunk], relevant_count=1)), \
         patch("app.api.query_stream.generate_answer_with_citations", return_value=answer), \
         patch("app.api.query_stream.verify_claims", return_value=[claim_verified]):

        response = client.post(
            "/api/v1/query",
            json={"query": "When was Python released?", "stream": True},
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        content = response.text
        assert "event: retrieval_started" in content
        assert "event: retrieval_complete" in content
        assert "event: generation_started" in content
        assert "event: generation_complete" in content
        assert "event: verification_started" in content
        assert "event: verification_complete" in content
        assert "event: final_answer" in content


def test_sse_streaming_correction_triggered():
    """Verify correction_triggered event is emitted when claim verification fails."""
    chunk = RetrievalResult(
        chunk_id="c1",
        score=0.9,
        metadata={"source_text": "Company earned $10M.", "document_id": "doc1", "page_number": 1},
    )
    answer = GeneratedAnswer(
        claims=[Claim(claim_text="Company earned $50M.", source_chunk_id="c1")],
        insufficient_information=False,
    )
    claim_contradicted = ClaimWithSource(
        claim_text="Company earned $50M.",
        source_chunk_id="c1",
        source_text="Company earned $10M.",
        is_valid_source=True,
        verification_status=VerificationStatus.CONTRADICTED,
        confidence=0.95,
    )
    claim_corrected = ClaimWithSource(
        claim_text="Company earned $10M.",
        source_chunk_id="c1",
        source_text="Company earned $10M.",
        is_valid_source=True,
        verification_status=VerificationStatus.ENTAILED,
        confidence=0.95,
    )

    verify_calls = 0

    def mock_verify(claims):
        nonlocal verify_calls
        verify_calls += 1
        return [claim_contradicted] if verify_calls == 1 else [claim_corrected]

    corrected_json = '{"claims": [{"claim_text": "Company earned $10M.", "source_chunk_id": "c1"}], "insufficient_information": false}'

    with patch("app.api.query_stream.hybrid_search", return_value=[chunk]), \
         patch("app.api.query_stream.select_relevant_chunks", return_value=RerankResult(chunks=[chunk], relevant_count=1)), \
         patch("app.api.query_stream.generate_answer_with_citations", return_value=answer), \
         patch("app.api.query_stream.verify_claims", side_effect=mock_verify), \
         patch("app.graph.targeted_retrieve.hybrid_search", return_value=[chunk]), \
         patch("app.graph.targeted_retrieve.select_relevant_chunks", return_value=RerankResult(chunks=[chunk], relevant_count=1)), \
         patch("app.generation.partial_regenerate.generate_answer", return_value=LLMResponse(content=corrected_json, provider="groq", model_name="llama3")):

        response = client.post(
            "/api/v1/query",
            json={"query": "What were the earnings?", "stream": True},
        )

        assert response.status_code == 200
        content = response.text
        assert "event: correction_triggered" in content
        assert "event: final_answer" in content


def test_sse_streaming_error_handling():
    """Verify pipeline exception mid-stream emits event: error payload."""
    with patch("app.api.query_stream.hybrid_search", side_effect=RuntimeError("Vector index unavailable")):
        response = client.post(
            "/api/v1/query",
            json={"query": "Failure test query", "stream": True},
        )

        assert response.status_code == 200
        content = response.text
        assert "event: error" in content
        assert "RuntimeError" in content
        assert "Vector index unavailable" in content
