"""
Unit tests for premise-hypothesis pairing logic and premise truncation (Phase 34).
"""

import pytest

from app.models.schemas import ClaimWithSource
from app.verification.pairing import VerificationPair, build_verification_pairs, truncate_premise


def test_build_verification_pairs_excludes_invalid_sources():
    claims = [
        ClaimWithSource(
            claim_text="Valid claim sentence.",
            source_chunk_id="chunk_1",
            source_text="Valid chunk context text.",
            is_valid_source=True,
        ),
        ClaimWithSource(
            claim_text="Invalid claim sentence.",
            source_chunk_id="fake_chunk_99",
            source_text=None,
            is_valid_source=False,
        ),
        ClaimWithSource(
            claim_text="Empty text claim sentence.",
            source_chunk_id="chunk_2",
            source_text="   ",
            is_valid_source=True,
        ),
    ]

    pairs = build_verification_pairs(claims)

    assert len(pairs) == 1
    assert pairs[0].claim_id == claims[0].claim_id
    assert pairs[0].premise == "Valid chunk context text."
    assert pairs[0].hypothesis == "Valid claim sentence."


def test_claim_id_traceability_preserved():
    c1 = ClaimWithSource(claim_text="First claim", source_chunk_id="c1", source_text="Context 1")
    c2 = ClaimWithSource(claim_text="Second claim", source_chunk_id="c2", source_text="Context 2")

    pairs = build_verification_pairs([c1, c2])

    assert len(pairs) == 2
    assert pairs[0].claim_id == c1.claim_id
    assert pairs[1].claim_id == c2.claim_id


def test_premise_truncation_on_oversized_text():
    # Build text longer than max_chars with sentence boundaries
    s1 = "This is sentence one containing important background information."
    s2 = "This is sentence two containing additional contextual details."
    s3 = "This is sentence three which exceeds the max character threshold."

    full_text = f"{s1} {s2} {s3}"
    max_len = len(s1) + len(s2) + 2  # Allows s1 + s2, excludes s3

    truncated = truncate_premise(full_text, max_chars=max_len)

    assert s1 in truncated
    assert s2 in truncated
    assert s3 not in truncated
    assert len(truncated) <= max_len
