"""
Unit test for Phase 41: Retry Cap Enforcement & Loop Termination Safety.
"""

from unittest.mock import patch

from app.core.config import settings
from app.graph.graph import app_graph
from app.graph.state import RAGState
from app.models.schemas import ClaimWithSource, GeneratedAnswer, RetrievalResult, VerificationStatus


def test_retry_cap_prevents_infinite_loop_on_persistent_failure():
    """Verify that graph execution strictly terminates at max_retries even if claims persistently fail verification."""
    persistent_failed_claim = ClaimWithSource(
        claim_text="Unresolvable persistent hallucination statement.",
        source_chunk_id="chunk-1",
        is_valid_source=True,
        verification_status=VerificationStatus.CONTRADICTED,
        confidence=0.95,
    )

    chunk = RetrievalResult(
        chunk_id="chunk-1",
        score=0.9,
        metadata={"source_text": "Real source context text", "document_id": "doc-1", "page_number": 1},
    )

    initial_state: RAGState = {
        "query": "Test query for persistent failure",
        "retrieved_chunks": [chunk],
        "generated_answer": GeneratedAnswer(claims=[], raw_response="Initial answer"),
        "claims": [persistent_failed_claim],
        "retry_count": 0,
        "max_retries": settings.CORRECTION_MAX_RETRIES,
        "final_status": "pending",
    }

    # Mock nodes so verify_node ALWAYS returns the failed claim
    with patch("app.graph.graph.hybrid_search") as mock_hybrid, \
         patch("app.graph.graph.select_relevant_chunks") as mock_rerank, \
         patch("app.graph.graph.verify_claims") as mock_verify, \
         patch("app.graph.targeted_retrieve.hybrid_search") as mock_targeted_hybrid, \
         patch("app.graph.targeted_retrieve.select_relevant_chunks") as mock_targeted_rerank, \
         patch("app.graph.regenerate.regenerate_failed_claims") as mock_regen:

        mock_hybrid.return_value = [chunk]
        mock_rerank.return_value = type("RerankRes", (), {"chunks": [chunk]})()

        mock_targeted_hybrid.return_value = [chunk]
        mock_targeted_rerank.return_value = type("RerankRes", (), {"chunks": [chunk]})()

        # Always return the CONTRADICTED claim to simulate persistent failure across retries
        mock_verify.return_value = [persistent_failed_claim]
        mock_regen.return_value = []

        # Execute compiled graph
        final_state = app_graph.invoke(initial_state)

        # Assert graph terminated cleanly at max_retries
        assert final_state["retry_count"] == settings.CORRECTION_MAX_RETRIES
        assert final_state["final_status"] == "unverifiable"
        assert "final_answer_text" in final_state
        assert final_state["final_answer_text"] == "Unresolvable persistent hallucination statement."

        # Verify router did not loop beyond max_retries
        assert mock_verify.call_count <= settings.CORRECTION_MAX_RETRIES + 2
