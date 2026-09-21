"""
Verification & Claim Processing module package exports (Phase 30-36).
"""

from app.models.schemas import ClaimWithSource
from app.verification.claim_mapper import map_claims_to_chunks
from app.verification.claim_splitter import (
    ensure_atomic_claims,
    is_atomic_claim,
    split_into_sentences,
)

__all__ = [
    "ClaimWithSource",
    "ensure_atomic_claims",
    "is_atomic_claim",
    "map_claims_to_chunks",
    "split_into_sentences",
]
