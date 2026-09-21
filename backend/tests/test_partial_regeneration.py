"""
Unit tests for Phase 40: Targeted Partial Regeneration and Claim Merging.
"""

from unittest.mock import patch

from app.generation.partial_regenerate import (
    merge_corrected_claims,
    regenerate_failed_claims,
)
from app.models.schemas import Claim, ClaimWithSource, LLMResponse, RetrievalResult, VerificationStatus


def test_regenerate_failed_claims_calls_llm_and_parses():
    """Verify regenerate_failed_claims formats prompt, invokes LLM, and parses corrected claims."""
    failed_claim = ClaimWithSource(
        claim_text="Acme acquired Beta for $500M.",
        source_chunk_id="chunk-1",
        is_valid_source=True,
        verification_status=VerificationStatus.CONTRADICTED,
        confidence=0.9,
    )

    enriched_chunk = RetrievalResult(
        chunk_id="chunk-2",
        score=0.95,
        metadata={"source_text": "Acme acquired Beta AI Inc for $200 million in cash.", "document_id": "doc-1", "page_number": 1},
    )

    mock_json = '{"claims": [{"claim_text": "Acme acquired Beta AI Inc for $200 million.", "source_chunk_id": "chunk-2"}], "insufficient_information": false}'

    with patch("app.generation.partial_regenerate.generate_answer") as mock_gen:
        mock_gen.return_value = LLMResponse(
            content=mock_json,
            provider="groq",
            model_name="llama3",
        )

        corrected = regenerate_failed_claims(
            original_query="What was the deal size for Beta AI?",
            failed_claims=[failed_claim],
            enriched_chunks=[enriched_chunk],
        )

        assert len(corrected) == 1
        assert corrected[0].claim_text == "Acme acquired Beta AI Inc for $200 million."
        assert corrected[0].source_chunk_id == "chunk-2"
        mock_gen.assert_called_once()


def test_merge_corrected_claims_preserves_verified_claims():
    """Verify merge_corrected_claims replaces failed claims by ID and leaves verified claims untouched byte-for-byte."""
    verified_claim = ClaimWithSource(
        claim_text="Acme was founded in 2010.",
        source_chunk_id="chunk-1",
        is_valid_source=True,
        verification_status=VerificationStatus.ENTAILED,
        confidence=0.98,
    )

    failed_claim = ClaimWithSource(
        claim_text="Acme has 50,000 employees.",
        source_chunk_id="chunk-2",
        is_valid_source=True,
        verification_status=VerificationStatus.CONTRADICTED,
        confidence=0.91,
    )

    corrected_claim = ClaimWithSource(
        claim_text="Acme has 5,000 employees worldwide.",
        source_chunk_id="chunk-3",
        is_valid_source=True,
    )

    original_claims = [verified_claim, failed_claim]

    merged = merge_corrected_claims(
        original_claims=original_claims,
        failed_claim_ids=[failed_claim.claim_id],
        corrected_claims=[corrected_claim],
    )

    assert len(merged) == 2
    # Verified claim preserved byte-for-byte (same claim_id, same confidence)
    assert merged[0] is verified_claim
    assert merged[0].claim_id == verified_claim.claim_id
    assert merged[0].verification_status == VerificationStatus.ENTAILED
    assert merged[0].confidence == 0.98

    # Failed claim replaced by corrected_claim
    assert merged[1] is corrected_claim
    assert merged[1].claim_text == "Acme has 5,000 employees worldwide."
    assert merged[1].source_chunk_id == "chunk-3"
