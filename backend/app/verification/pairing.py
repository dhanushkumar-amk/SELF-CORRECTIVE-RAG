r"""
Premise-Hypothesis Pairing Construction & Truncation Module for Phase 34.

Architecture & Design Decisions:
1. Traceable VerificationPair Structure:
   Returns `VerificationPair` objects binding `claim_id` with `premise` and `hypothesis`.
   This ensures 100% deterministic reattachment of NLI scores back to the originating claims
   after batched inference, preventing result mix-ups.

2. Invalid Source Exclusion:
   Claims with `is_valid_source=False` or empty `source_text` are excluded from pairing
   and never sent to the NLI model. They are assigned `VerificationStatus.UNVERIFIABLE` directly
   in `claim_verifier.py`.

3. Sentence-Boundary-Aware Premise Truncation:
   The NLI model (`nli-deberta-v3-base`) has a max sequence length of 512 tokens (~1500-2000 characters).
   If a source chunk context exceeds `max_premise_chars` (default 1500), `truncate_premise()` splits
   the text on sentence boundaries (`r"(?<=[.!?])\s+"`) to preserve complete sentences rather than
   cutting off mid-word or mid-sentence.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.models.schemas import ClaimWithSource

logger = get_logger(__name__)

__all__ = [
    "VerificationPair",
    "build_verification_pairs",
    "truncate_premise",
]


class VerificationPair(BaseModel):
    """Pairing container coupling source chunk premise with claim hypothesis for NLI inference."""

    claim_id: str = Field(description="Originating claim UUID for deterministic batch result reattachment")
    premise: str = Field(description="Sanitized and truncated source chunk text context")
    hypothesis: str = Field(description="Atomic claim statement text")


def truncate_premise(text: str, max_chars: int = 1500) -> str:
    """Truncate premise text on sentence boundaries to fit within NLI model max sequence length.

    Args:
        text: Raw source chunk text context.
        max_chars: Maximum character limit (default 1500 chars).

    Returns:
        Sentence-boundary-aware truncated premise string.
    """
    if not text or len(text) <= max_chars:
        return text or ""

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    accumulated: list[str] = []
    current_length = 0

    for sentence in sentences:
        s_len = len(sentence)
        if current_length + s_len + 1 <= max_chars:
            accumulated.append(sentence)
            current_length += s_len + 1
        else:
            break

    if accumulated:
        return " ".join(accumulated)

    # Fallback to hard character truncation if single first sentence exceeds limit
    return text[:max_chars].strip()


def build_verification_pairs(
    claims: list[ClaimWithSource],
    max_premise_chars: int = 1500,
) -> list[VerificationPair]:
    """Construct NLI verification pairs from a list of ClaimWithSource objects.

    Excludes invalid/unresolvable source claims (where `is_valid_source` is False).

    Args:
        claims: List of input ClaimWithSource objects.
        max_premise_chars: Maximum character limit for premise truncation.

    Returns:
        List of VerificationPair objects containing claim_id, premise, and hypothesis.
    """
    pairs: list[VerificationPair] = []

    for claim in claims:
        # Exclude claims with invalid or missing source context
        if not claim.is_valid_source or not claim.source_text or not claim.source_text.strip():
            logger.debug(
                "Skipping pairing for claim '%s' (claim_id: %s) because is_valid_source=%s or source_text is empty.",
                claim.claim_text[:40],
                claim.claim_id,
                claim.is_valid_source,
            )
            continue

        premise = truncate_premise(claim.source_text, max_chars=max_premise_chars)
        hypothesis = claim.claim_text.strip()

        pairs.append(
            VerificationPair(
                claim_id=claim.claim_id,
                premise=premise,
                hypothesis=hypothesis,
            )
        )

    logger.debug(
        "Built %d verification pairs from %d input claims (%d excluded as unverifiable).",
        len(pairs),
        len(claims),
        len(claims) - len(pairs),
    )

    return pairs
