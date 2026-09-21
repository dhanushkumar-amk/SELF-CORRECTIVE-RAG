"""
Unit tests for Phase 38: Conditional Routing logic and get_failed_claims().
"""

import pytest

from app.graph.routing import correction_router, get_failed_claims
from app.graph.state import RAGState
from app.models.schemas import ClaimWithSource, VerificationStatus


def test_get_failed_claims_filtering():
    """Verify get_failed_claims() extracts only claims whose final status is not 'verified'."""
    verified_claim = ClaimWithSource(
        claim_text="The company made $10M in revenue.",
        source_chunk_id="chunk-1",
        is_valid_source=True,
        verification_status=VerificationStatus.ENTAILED,
        confidence=0.95,
    )
    needs_review_claim = ClaimWithSource(
        claim_text="The company expanded into 5 countries.",
        source_chunk_id="chunk-2",
        is_valid_source=True,
        verification_status=VerificationStatus.NEUTRAL,
        confidence=0.50,
    )
    contradicted_claim = ClaimWithSource(
        claim_text="The company lost $50M in revenue.",
        source_chunk_id="chunk-3",
        is_valid_source=True,
        verification_status=VerificationStatus.CONTRADICTED,
        confidence=0.90,
    )
    unverifiable_claim = ClaimWithSource(
        claim_text="The CEO was born on Mars.",
        source_chunk_id="non-existent-chunk",
        is_valid_source=False,
        verification_status=VerificationStatus.UNVERIFIABLE,
        confidence=0.0,
    )

    claims = [
        verified_claim,
        needs_review_claim,
        contradicted_claim,
        unverifiable_claim,
    ]

    failed = get_failed_claims(claims)

    assert len(failed) == 3
    assert verified_claim not in failed
    assert needs_review_claim in failed
    assert contradicted_claim in failed
    assert unverifiable_claim in failed


def test_correction_router_all_verified():
    """Test that correction_router routes to 'finalize' when all claims are verified."""
    verified_claim = ClaimWithSource(
        claim_text="Valid statement.",
        source_chunk_id="chunk-1",
        is_valid_source=True,
        verification_status=VerificationStatus.ENTAILED,
        confidence=0.99,
    )

    state: RAGState = {
        "query": "Test query",
        "retrieved_chunks": [],
        "generated_answer": None,
        "claims": [verified_claim],
        "retry_count": 0,
        "max_retries": 2,
        "final_status": "pending",
    }

    route = correction_router(state)
    assert route == "finalize"


@pytest.mark.parametrize(
    "status,is_valid,conf",
    [
        (VerificationStatus.NEUTRAL, True, 0.50),        # needs_review
        (VerificationStatus.CONTRADICTED, True, 0.90),    # contradicted
        (VerificationStatus.UNVERIFIABLE, False, 0.0),    # unverifiable
        (VerificationStatus.ENTAILED, True, 0.40),        # needs_review (below threshold)
    ],
)
def test_correction_router_failed_claim_with_retries(status, is_valid, conf):
    """Test routing to 'targeted_retrieve' when a failed claim is present and retry_count < max_retries."""
    claim = ClaimWithSource(
        claim_text="Problematic claim",
        source_chunk_id="chunk-1",
        is_valid_source=is_valid,
        verification_status=status,
        confidence=conf,
    )

    state: RAGState = {
        "query": "Test query",
        "retrieved_chunks": [],
        "generated_answer": None,
        "claims": [claim],
        "retry_count": 0,
        "max_retries": 2,
        "final_status": "pending",
    }

    route = correction_router(state)
    assert route == "targeted_retrieve"


def test_correction_router_retries_exhausted():
    """Test routing to 'finalize' when retry_count >= max_retries even with failed claims."""
    failed_claim = ClaimWithSource(
        claim_text="False claim",
        source_chunk_id="chunk-1",
        is_valid_source=True,
        verification_status=VerificationStatus.CONTRADICTED,
        confidence=0.95,
    )

    state: RAGState = {
        "query": "Test query",
        "retrieved_chunks": [],
        "generated_answer": None,
        "claims": [failed_claim],
        "retry_count": 2,
        "max_retries": 2,
        "final_status": "pending",
    }

    route = correction_router(state)
    assert route == "finalize"
