"""
Unit tests for finalized Claim models and VerificationStatus enum (Phase 32).
"""

import uuid
import pytest

from app.models.schemas import ClaimWithSource, NLIScore, VerificationStatus


def test_verification_status_enum_members():
    assert VerificationStatus.PENDING.value == "pending"
    assert VerificationStatus.ENTAILED.value == "entailed"
    assert VerificationStatus.CONTRADICTED.value == "contradicted"
    assert VerificationStatus.NEUTRAL.value == "neutral"
    assert VerificationStatus.UNVERIFIABLE.value == "unverifiable"


def test_claim_with_source_model_defaults_and_uuid_generation():
    claim = ClaimWithSource(
        claim_text="SentenceTransformers models generate 384-dimensional embeddings.",
        source_chunk_id="chunk_101",
        source_text="Full chunk text context.",
    )

    # Verify claim_id is auto-generated UUID4
    assert claim.claim_id is not None
    parsed_uuid = uuid.UUID(claim.claim_id)
    assert parsed_uuid.version == 4

    # Verify default fields
    assert claim.is_valid_source is True
    assert claim.verification_status is None
    assert claim.confidence is None
    assert claim.page_number is None


def test_nli_score_schema_validation():
    score = NLIScore(
        contradiction=0.001,
        entailment=0.995,
        neutral=0.004,
        predicted_label=VerificationStatus.ENTAILED,
    )
    assert score.predicted_label == VerificationStatus.ENTAILED
    assert score.entailment == 0.995
