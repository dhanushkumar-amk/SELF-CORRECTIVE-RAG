"""
Unit tests for Phase 39: Targeted Re-Retrieval logic, chunk merging/deduplication, and nothing-new-found logging.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.graph.state import RAGState
from app.graph.targeted_retrieve import targeted_retrieve_node
from app.models.schemas import ClaimWithSource, RerankResult, RetrievalResult, VerificationStatus


def test_targeted_retrieve_uses_claim_text_as_query():
    """Verify targeted_retrieve_node queries using failed claim.claim_text rather than original user query."""
    failed_claim = ClaimWithSource(
        claim_text="The company expanded to 5 new markets in 2024.",
        source_chunk_id="chunk-old-1",
        is_valid_source=True,
        verification_status=VerificationStatus.CONTRADICTED,
        confidence=0.88,
    )

    original_chunk = RetrievalResult(
        chunk_id="chunk-old-1",
        score=0.9,
        metadata={"source_text": "Original context text", "document_id": "doc-1", "page_number": 1},
    )

    new_retrieved_chunk = RetrievalResult(
        chunk_id="chunk-new-2",
        score=0.95,
        metadata={"source_text": "Company expanded to 5 markets in 2024.", "document_id": "doc-1", "page_number": 3},
    )

    state: RAGState = {
        "query": "What is the company's financial and global status?",
        "retrieved_chunks": [original_chunk],
        "generated_answer": None,
        "claims": [failed_claim],
        "retry_count": 0,
        "max_retries": 2,
        "final_status": "pending",
    }

    with patch("app.graph.targeted_retrieve.hybrid_search") as mock_hybrid, patch(
        "app.graph.targeted_retrieve.select_relevant_chunks"
    ) as mock_rerank:

        mock_hybrid.return_value = [new_retrieved_chunk]
        mock_rerank.return_value = RerankResult(chunks=[new_retrieved_chunk], relevant_count=1)

        result = targeted_retrieve_node(state)

        # Assert search was called with claim_text, NOT state["query"]
        mock_hybrid.assert_called_once_with(query="The company expanded to 5 new markets in 2024.")
        mock_rerank.assert_called_once_with(
            query="The company expanded to 5 new markets in 2024.",
            candidates=[new_retrieved_chunk],
        )

        assert result["retry_count"] == 1
        assert len(result["retrieved_chunks"]) == 2
        assert result["retrieved_chunks"][0].chunk_id == "chunk-old-1"
        assert result["retrieved_chunks"][1].chunk_id == "chunk-new-2"


def test_targeted_retrieve_deduplication():
    """Verify that merging deduplicates chunks by chunk_id."""
    failed_claim = ClaimWithSource(
        claim_text="Revenue grew by 20 percent.",
        source_chunk_id="chunk-1",
        is_valid_source=True,
        verification_status=VerificationStatus.NEUTRAL,
        confidence=0.4,
    )

    existing_chunk1 = RetrievalResult(chunk_id="chunk-1", score=0.8, metadata={})
    existing_chunk2 = RetrievalResult(chunk_id="chunk-2", score=0.75, metadata={})

    # Targeted retrieval returns chunk-2 (already present) and chunk-3 (new)
    retrieved_chunk2 = RetrievalResult(chunk_id="chunk-2", score=0.75, metadata={})
    retrieved_chunk3 = RetrievalResult(chunk_id="chunk-3", score=0.9, metadata={})

    state: RAGState = {
        "query": "What was the growth rate?",
        "retrieved_chunks": [existing_chunk1, existing_chunk2],
        "generated_answer": None,
        "claims": [failed_claim],
        "retry_count": 0,
        "max_retries": 2,
        "final_status": "pending",
    }

    with patch("app.graph.targeted_retrieve.hybrid_search") as mock_hybrid, patch(
        "app.graph.targeted_retrieve.select_relevant_chunks"
    ) as mock_rerank:

        mock_hybrid.return_value = [retrieved_chunk2, retrieved_chunk3]
        mock_rerank.return_value = RerankResult(chunks=[retrieved_chunk2, retrieved_chunk3], relevant_count=2)

        result = targeted_retrieve_node(state)

        # Chunks should be: [chunk-1, chunk-2, chunk-3] (no double chunk-2)
        merged = result["retrieved_chunks"]
        chunk_ids = [c.chunk_id for c in merged]

        assert chunk_ids == ["chunk-1", "chunk-2", "chunk-3"]
        assert len(merged) == 3


def test_targeted_retrieve_nothing_new_found_logging(caplog):
    """Verify 'nothing new found' warning log triggers when targeted retrieval returns only already-retrieved chunks."""
    failed_claim = ClaimWithSource(
        claim_text="Unmatched assertion.",
        source_chunk_id="chunk-1",
        is_valid_source=True,
        verification_status=VerificationStatus.CONTRADICTED,
        confidence=0.9,
    )

    existing_chunk = RetrievalResult(chunk_id="chunk-1", score=0.8, metadata={})

    state: RAGState = {
        "query": "Some query",
        "retrieved_chunks": [existing_chunk],
        "generated_answer": None,
        "claims": [failed_claim],
        "retry_count": 0,
        "max_retries": 2,
        "final_status": "pending",
    }

    with patch("app.graph.targeted_retrieve.hybrid_search") as mock_hybrid, patch(
        "app.graph.targeted_retrieve.select_relevant_chunks"
    ) as mock_rerank:

        mock_hybrid.return_value = [existing_chunk]
        mock_rerank.return_value = RerankResult(chunks=[existing_chunk], relevant_count=1)

        with caplog.at_level("WARNING"):
            result = targeted_retrieve_node(state)

        assert len(result["retrieved_chunks"]) == 1
        assert "yielded NO new context chunks" in caplog.text
