"""
Unit tests for claim confidence scoring, empirical thresholding, and final status labels (Phase 36).
"""

import pytest

from app.models.schemas import ClaimWithSource, VerificationStatus
from app.verification.claim_verifier import get_claim_final_status


def test_get_claim_final_status_verified():
    claim = ClaimWithSource(
        claim_text="SentenceTransformers produces 384d vectors.",
        source_chunk_id="chunk_1",
        source_text="Context text.",
        is_valid_source=True,
        verification_status=VerificationStatus.ENTAILED,
        confidence=0.96,
    )
    status = get_claim_final_status(claim, threshold=0.85)
    assert status == "verified"


def test_get_claim_final_status_needs_review_borderline_entailed():
    claim = ClaimWithSource(
        claim_text="Borderline entailed statement.",
        source_chunk_id="chunk_1",
        source_text="Context text.",
        is_valid_source=True,
        verification_status=VerificationStatus.ENTAILED,
        confidence=0.72,  # Below 0.85 threshold
    )
    status = get_claim_final_status(claim, threshold=0.85)
    assert status == "needs_review"


def test_get_claim_final_status_needs_review_neutral():
    claim = ClaimWithSource(
        claim_text="Ungrounded neutral claim statement.",
        source_chunk_id="chunk_1",
        source_text="Context text.",
        is_valid_source=True,
        verification_status=VerificationStatus.NEUTRAL,
        confidence=0.91,
    )
    status = get_claim_final_status(claim, threshold=0.85)
    assert status == "needs_review"


def test_get_claim_final_status_contradicted():
    claim = ClaimWithSource(
        claim_text="Contradicted statement.",
        source_chunk_id="chunk_1",
        source_text="Context text.",
        is_valid_source=True,
        verification_status=VerificationStatus.CONTRADICTED,
        confidence=0.99,
    )
    status = get_claim_final_status(claim, threshold=0.85)
    assert status == "contradicted"


def test_get_claim_final_status_unverifiable():
    claim = ClaimWithSource(
        claim_text="Hallucinated statement with fake chunk ID.",
        source_chunk_id="fake_chunk",
        source_text=None,
        is_valid_source=False,
        verification_status=VerificationStatus.UNVERIFIABLE,
        confidence=0.0,
    )
    status = get_claim_final_status(claim, threshold=0.85)
    assert status == "unverifiable"
