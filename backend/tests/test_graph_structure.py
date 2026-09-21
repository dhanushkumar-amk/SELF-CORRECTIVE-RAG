"""
Unit tests for LangGraph state machine structure and end-to-end execution flow (Phase 37).
"""

from unittest.mock import MagicMock, patch
import pytest

from app.graph.graph import app_graph, build_rag_graph, correction_router
from app.graph.state import RAGState
from app.models.schemas import (
    Claim,
    ClaimWithSource,
    GeneratedAnswer,
    RerankResult,
    RetrievalResult,
    VerificationStatus,
)


def test_graph_compiles_without_errors():
    """Test that build_rag_graph() constructs and compiles a valid LangGraph state graph."""
    graph = build_rag_graph()
    assert graph is not None


def test_correction_router_decisions():
    """Test correction_router logic for finalized vs retry routing."""
    # Case 1: All claims verified -> route to finalize
    verified_claim = ClaimWithSource(
        claim_text="Verified statement.",
        source_chunk_id="chunk_1",
        source_text="Source text.",
        is_valid_source=True,
        verification_status=VerificationStatus.ENTAILED,
        confidence=0.95,
    )
    state_all_good: RAGState = {
        "claims": [verified_claim],
        "retry_count": 0,
        "max_retries": 2,
    }
    assert correction_router(state_all_good) == "finalize"

    # Case 2: Contradicted claim present & retries remaining -> route to targeted_retrieve
    contradicted_claim = ClaimWithSource(
        claim_text="Contradicted statement.",
        source_chunk_id="chunk_1",
        source_text="Source text.",
        is_valid_source=True,
        verification_status=VerificationStatus.CONTRADICTED,
        confidence=0.98,
    )
    state_failed: RAGState = {
        "claims": [contradicted_claim],
        "retry_count": 0,
        "max_retries": 2,
    }
    assert correction_router(state_failed) == "targeted_retrieve"

    # Case 3: Retries exhausted -> route to finalize even if unverified
    state_exhausted: RAGState = {
        "claims": [contradicted_claim],
        "retry_count": 2,
        "max_retries": 2,
    }
    assert correction_router(state_exhausted) == "finalize"


def test_graph_end_to_end_mocked_execution_flow():
    """Test executing a query through app_graph.invoke() and verifying state accumulation across nodes."""
    sample_chunk = RetrievalResult(
        chunk_id="c1",
        score=0.9,
        metadata={
            "source_text": "SentenceTransformers models produce 384 dimensional vectors.",
            "page_number": 1,
            "filename": "test.pdf",
        },
    )

    sample_answer = GeneratedAnswer(
        claims=[Claim(claim_text="SentenceTransformers models produce 384 dimensional vectors.", source_chunk_id="c1")],
        insufficient_information=False,
    )

    with patch("app.graph.graph.hybrid_search", return_value=[sample_chunk]), \
         patch("app.graph.graph.select_relevant_chunks", return_value=RerankResult(chunks=[sample_chunk], total_candidates=1, relevant_count=1)), \
         patch("app.graph.graph.generate_answer_with_citations", return_value=sample_answer):

        initial_state: RAGState = {
            "query": "What dimension vectors do sentence transformers produce?",
            "max_retries": 2,
        }

        final_state = app_graph.invoke(initial_state)

        assert final_state["query"] == "What dimension vectors do sentence transformers produce?"
        assert len(final_state["retrieved_chunks"]) == 1
        assert final_state["generated_answer"] is not None
        assert len(final_state["claims"]) == 1
        assert final_state["claims"][0].verification_status == VerificationStatus.ENTAILED
        assert final_state["final_status"] == "verified"
