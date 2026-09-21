"""
Unit tests for Phase 41: Finalize Node state synthesis and status labeling.
"""

from app.graph.finalize import finalize_node
from app.graph.state import RAGState
from app.models.schemas import ClaimWithSource, VerificationStatus


def test_finalize_node_fully_verified():
    """Verify finalize_node assigns 'fully_verified' when all claims pass NLI verification."""
    claim1 = ClaimWithSource(
        claim_text="Acme revenue reached $10M.",
        source_chunk_id="chunk-1",
        is_valid_source=True,
        verification_status=VerificationStatus.ENTAILED,
        confidence=0.95,
    )
    claim2 = ClaimWithSource(
        claim_text="Acme operates in 3 continents.",
        source_chunk_id="chunk-2",
        is_valid_source=True,
        verification_status=VerificationStatus.ENTAILED,
        confidence=0.92,
    )

    state: RAGState = {
        "claims": [claim1, claim2],
        "retry_count": 0,
        "max_retries": 2,
    }

    result = finalize_node(state)

    assert result["final_status"] == "fully_verified"
    assert result["final_answer_text"] == "Acme revenue reached $10M. Acme operates in 3 continents."


def test_finalize_node_partially_verified():
    """Verify finalize_node assigns 'partially_verified' when retries exhaust with mixed claims."""
    verified_claim = ClaimWithSource(
        claim_text="Acme revenue was $10M.",
        source_chunk_id="chunk-1",
        is_valid_source=True,
        verification_status=VerificationStatus.ENTAILED,
        confidence=0.95,
    )
    failed_claim = ClaimWithSource(
        claim_text="Acme acquired Beta for $5B.",
        source_chunk_id="chunk-2",
        is_valid_source=True,
        verification_status=VerificationStatus.CONTRADICTED,
        confidence=0.90,
    )

    state: RAGState = {
        "claims": [verified_claim, failed_claim],
        "retry_count": 2,
        "max_retries": 2,
    }

    result = finalize_node(state)

    assert result["final_status"] == "partially_verified"
    assert result["final_answer_text"] == "Acme revenue was $10M. Acme acquired Beta for $5B."


def test_finalize_node_unverifiable():
    """Verify finalize_node assigns 'unverifiable' when 0 claims pass verification or list is empty."""
    empty_state: RAGState = {
        "claims": [],
        "retry_count": 0,
        "max_retries": 2,
    }

    result = finalize_node(empty_state)

    assert result["final_status"] == "unverifiable"
    assert "No valid factual statements" in result["final_answer_text"]
