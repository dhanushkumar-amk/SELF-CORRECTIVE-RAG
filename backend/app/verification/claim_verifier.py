"""
End-to-End Claim Verification & NLI Classification Module for Phase 35.

Architecture & Design Decisions:
1. Batched NLI Inference:
   Calls `build_verification_pairs()` to construct pairs, then passes all valid pairs to
   `score_entailment_batch()` in a single batch pass for optimal performance.

2. Result Reattachment via UUID claim_id:
   Uses `VerificationPair.claim_id` to map each `NLIScore` back onto the exact originating claim
   without relying on index positions or string matching.

3. Treatment of UNVERIFIABLE Claims:
   Claims with `is_valid_source=False` or empty `source_text` skip NLI inference entirely.
   They are directly assigned `verification_status = VerificationStatus.UNVERIFIABLE` and `confidence = 0.0`.

4. Treatment of NEUTRAL Predictions:
   In RAG hallucination detection, an unsupported assertion (neither confirmed nor denied by source text)
   is an ungrounded hallucination risk. Therefore, `NEUTRAL` claims are flagged as unsupported,
   and will trigger self-correction in Phase 39 alongside `CONTRADICTED` claims.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.models.schemas import ClaimWithSource, NLIScore, VerificationStatus
from app.verification.nli_verifier import score_entailment_batch
from app.verification.pairing import build_verification_pairs

logger = get_logger(__name__)

__all__ = [
    "verify_claims",
]


def verify_claims(claims: list[ClaimWithSource]) -> list[ClaimWithSource]:
    """Verify a list of ClaimWithSource objects against their source chunk premises using NLI.

    Populates `verification_status` and `confidence` fields on each claim in-place and returns the list.

    Args:
        claims: List of input ClaimWithSource objects.

    Returns:
        List of updated ClaimWithSource objects with populated verification_status and confidence.
    """
    if not claims:
        logger.debug("verify_claims called with empty claim list.")
        return []

    # 1. Build verification pairs for claims with valid sources
    pairs = build_verification_pairs(claims)

    # 2. Execute batched NLI inference for valid pairs
    score_map: dict[str, NLIScore] = {}
    if pairs:
        nli_inputs = [(p.premise, p.hypothesis) for p in pairs]
        logger.info("Executing batched NLI verification across %d pairs...", len(nli_inputs))
        nli_scores = score_entailment_batch(nli_inputs)

        # 3. Build claim_id -> NLIScore lookup map
        for i, pair in enumerate(pairs):
            score_map[pair.claim_id] = nli_scores[i]

    # 4. Update each claim with NLI verification status and confidence score
    for claim in claims:
        if not claim.is_valid_source or not claim.source_text or not claim.source_text.strip():
            claim.verification_status = VerificationStatus.UNVERIFIABLE
            claim.confidence = 0.0
            logger.warning("Claim '%s' [ID: %s] marked UNVERIFIABLE (invalid source).", claim.claim_text[:40], claim.claim_id)
        elif claim.claim_id in score_map:
            score = score_map[claim.claim_id]
            claim.verification_status = score.predicted_label
            # Confidence is highest probability among entailment, contradiction, and neutral
            claim.confidence = max(score.entailment, score.contradiction, score.neutral)
            logger.info(
                "Claim '%s' verified as %s (confidence: %.4f).",
                claim.claim_text[:40],
                claim.verification_status.value.upper(),
                claim.confidence,
            )

    # 5. Log summary metrics
    status_counts = {
        VerificationStatus.ENTAILED: sum(1 for c in claims if c.verification_status == VerificationStatus.ENTAILED),
        VerificationStatus.CONTRADICTED: sum(1 for c in claims if c.verification_status == VerificationStatus.CONTRADICTED),
        VerificationStatus.NEUTRAL: sum(1 for c in claims if c.verification_status == VerificationStatus.NEUTRAL),
        VerificationStatus.UNVERIFIABLE: sum(1 for c in claims if c.verification_status == VerificationStatus.UNVERIFIABLE),
    }

    logger.info(
        "Claim verification completed: %d entailed, %d contradicted, %d neutral, %d unverifiable.",
        status_counts[VerificationStatus.ENTAILED],
        status_counts[VerificationStatus.CONTRADICTED],
        status_counts[VerificationStatus.NEUTRAL],
        status_counts[VerificationStatus.UNVERIFIABLE],
    )

    return claims
