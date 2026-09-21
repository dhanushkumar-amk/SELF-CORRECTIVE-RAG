"""
Verification & Claim Processing module package exports (Phase 30-36).
"""

from app.models.schemas import ClaimWithSource, NLIScore, VerificationStatus
from app.verification.claim_mapper import map_claims_to_chunks
from app.verification.claim_splitter import (
    ensure_atomic_claims,
    is_atomic_claim,
    split_into_sentences,
)
from app.verification.claim_verifier import verify_claims
from app.verification.nli_verifier import (
    NLIVerifier,
    get_nli_verifier,
    score_entailment,
    score_entailment_batch,
)
from app.verification.pairing import (
    VerificationPair,
    build_verification_pairs,
    truncate_premise,
)

__all__ = [
    "ClaimWithSource",
    "NLIScore",
    "NLIVerifier",
    "VerificationPair",
    "VerificationStatus",
    "build_verification_pairs",
    "ensure_atomic_claims",
    "get_nli_verifier",
    "is_atomic_claim",
    "map_claims_to_chunks",
    "score_entailment",
    "score_entailment_batch",
    "split_into_sentences",
    "truncate_premise",
    "verify_claims",
]
