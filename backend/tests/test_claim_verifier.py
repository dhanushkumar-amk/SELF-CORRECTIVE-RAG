"""
Unit tests for end-to-end claim verification and NLI classification (Phase 35).
"""

from unittest.mock import MagicMock, patch
import pytest

from app.models.schemas import ClaimWithSource, VerificationStatus
from app.verification.claim_verifier import verify_claims


def test_verify_claims_entailed():
    """Test that a claim grounded in its source chunk is classified as ENTAILED."""
    claims = [
        ClaimWithSource(
            claim_text="SentenceTransformers models map text to a 384 dimensional vector space.",
            source_chunk_id="chunk_101",
            source_text="SentenceTransformers all-MiniLM-L6-v2 maps sentences & paragraphs to a 384 dimensional dense vector space.",
            is_valid_source=True,
        )
    ]

    results = verify_claims(claims)

    assert len(results) == 1
    assert results[0].verification_status == VerificationStatus.ENTAILED
    assert results[0].confidence is not None
    assert results[0].confidence > 0.80


def test_verify_claims_contradicted():
    """Test that a deliberately wrong claim contradicting the source chunk is classified as CONTRADICTED."""
    claims = [
        ClaimWithSource(
            claim_text="SentenceTransformers models generate 4096-dimensional vectors for MySQL databases.",
            source_chunk_id="chunk_101",
            source_text="SentenceTransformers all-MiniLM-L6-v2 maps sentences & paragraphs to a 384 dimensional dense vector space.",
            is_valid_source=True,
        )
    ]

    results = verify_claims(claims)

    assert len(results) == 1
    assert results[0].verification_status == VerificationStatus.CONTRADICTED
    assert results[0].confidence is not None
    assert results[0].confidence > 0.80


def test_verify_claims_unverifiable_skips_nli():
    """Test that claims with is_valid_source=False skip NLI inference and are set to UNVERIFIABLE."""
    claims = [
        ClaimWithSource(
            claim_text="Hallucinated claim with fake chunk ID.",
            source_chunk_id="fake_chunk_99",
            source_text=None,
            is_valid_source=False,
        )
    ]

    with patch("app.verification.claim_verifier.score_entailment_batch") as mock_score_batch:
        results = verify_claims(claims)

        # score_entailment_batch should NOT be called since valid pairs list was empty
        mock_score_batch.assert_not_called()

        assert len(results) == 1
        assert results[0].verification_status == VerificationStatus.UNVERIFIABLE
        assert results[0].confidence == 0.0


def test_verify_claims_batch_reattachment():
    """Test batch verification reattaches NLI results to the exact corresponding claim_id."""
    c1 = ClaimWithSource(
        claim_text="The sky is blue.",
        source_chunk_id="c1",
        source_text="The sky is blue and clear.",
        is_valid_source=True,
    )
    c2 = ClaimWithSource(
        claim_text="The sky is red.",
        source_chunk_id="c2",
        source_text="The sky is blue and clear.",
        is_valid_source=True,
    )
    c3 = ClaimWithSource(
        claim_text="Unverifiable claim.",
        source_chunk_id="fake",
        source_text=None,
        is_valid_source=False,
    )

    results = verify_claims([c1, c2, c3])

    assert len(results) == 3
    assert results[0].claim_id == c1.claim_id
    assert results[0].verification_status == VerificationStatus.ENTAILED

    assert results[1].claim_id == c2.claim_id
    assert results[1].verification_status == VerificationStatus.CONTRADICTED

    assert results[2].claim_id == c3.claim_id
    assert results[2].verification_status == VerificationStatus.UNVERIFIABLE
