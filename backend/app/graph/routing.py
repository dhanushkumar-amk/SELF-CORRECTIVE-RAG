"""
Conditional Routing Module for Phase 38: Self-Correction State Machine Edges.

Architecture & Design Decisions:
1. Failed Claim Identification (`get_failed_claims`):
   - Isolates non-verified claims ("needs_review", "contradicted", "unverifiable") from the verified ones.
   - Prevents re-retrieving or re-generating context for claims that already passed NLI verification.

2. Routing Decision Logic (`correction_router`):
   - Evaluates system state post `verify_node`.
   - If ALL claims are "verified", or if retries are exhausted (`retry_count >= max_retries`),
     routes directly to `finalize`.
   - If ANY claim failed verification and retries remain (`retry_count < max_retries`),
     routes to `targeted_retrieve`.

3. Treatment of UNVERIFIABLE Claims:
   - Routed to `targeted_retrieve_node` alongside CONTRADICTED and NEEDS_REVIEW claims.
   - Rationale: UNVERIFIABLE claims cite invalid/hallucinated chunk IDs. Using the claim's text
     as a search query gives the retriever a chance to locate valid, grounding context chunks in the index.
     If no chunks exist, targeted retrieval returns 0 new chunks, and retry limits cap the loop.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.graph.state import RAGState
from app.models.schemas import ClaimWithSource
from app.verification.claim_verifier import get_claim_final_status

logger = get_logger(__name__)

__all__ = [
    "correction_router",
    "get_failed_claims",
]


def get_failed_claims(claims: list[ClaimWithSource]) -> list[ClaimWithSource]:
    """Filter claims to return only those requiring self-correction re-retrieval.

    A claim requires correction if its final status is NOT "verified"
    (i.e. "needs_review", "contradicted", "unverifiable", or "pending").

    Args:
        claims: List of processed ClaimWithSource objects.

    Returns:
        List of non-verified ClaimWithSource objects.
    """
    if not claims:
        return []

    return [claim for claim in claims if get_claim_final_status(claim) != "verified"]


def correction_router(state: RAGState) -> str:
    """LangGraph Conditional Edge function executed after `verify_node`.

    Determines whether the execution graph transitions to `finalize` or `targeted_retrieve`.

    Decision Rules:
    - If `claims` is empty or ALL claims are "verified": route to "finalize".
    - If `retry_count >= max_retries`: route to "finalize" (retry cap safety).
    - If ANY claim failed verification and retries remain: route to "targeted_retrieve".

    Args:
        state: Current LangGraph state dictionary (RAGState).

    Returns:
        Graph route target string: "finalize" or "targeted_retrieve".
    """
    claims = state.get("claims", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    failed_claims = get_failed_claims(claims)
    all_verified = bool(claims) and len(failed_claims) == 0

    if all_verified:
        logger.info(
            "Correction Router decision: 'finalize' (all %d claims verified successfully).",
            len(claims),
        )
        return "finalize"

    if retry_count >= max_retries:
        logger.info(
            "Correction Router decision: 'finalize' (retry count %d reached max_retries %d; %d failed claims remaining).",
            retry_count,
            max_retries,
            len(failed_claims),
        )
        return "finalize"

    logger.info(
        "Correction Router decision: 'targeted_retrieve' (%d/%d claims failed verification; retry %d/%d).",
        len(failed_claims),
        len(claims),
        retry_count + 1,
        max_retries,
    )
    return "targeted_retrieve"
