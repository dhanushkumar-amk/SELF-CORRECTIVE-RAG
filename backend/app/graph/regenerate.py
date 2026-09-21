"""
Regeneration Node for Phase 40: Targeted Partial Re-Generation State Machine Node.

Architecture & Design Decisions:
1. `regenerate_node`:
   - Receives state post `targeted_retrieve`.
   - Executes targeted re-generation only for failed claims.
   - Maps corrected claims against enriched chunks via `map_claims_to_chunks()`.
   - Merges corrected claims back into `state.claims` via `merge_corrected_claims()`.
   - Returns updated `claims` dictionary to trigger `verify_node` re-verification.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.generation.partial_regenerate import (
    merge_corrected_claims,
    regenerate_failed_claims,
)
from app.graph.routing import get_failed_claims
from app.graph.state import RAGState
from app.models.schemas import GeneratedAnswer
from app.verification.claim_mapper import map_claims_to_chunks

logger = get_logger(__name__)

__all__ = [
    "regenerate_node",
]


def regenerate_node(state: RAGState) -> dict[str, Any]:
    """LangGraph Node: Execute partial re-generation for failed claims and update state.

    Args:
        state: Current LangGraph state dictionary (RAGState).

    Returns:
        Dictionary with updated `claims` list.
    """
    original_claims = state.get("claims", [])
    chunks = state.get("retrieved_chunks", [])
    query = state.get("query", "")

    failed_claims = get_failed_claims(original_claims)
    logger.info("--- LANGGRAPH NODE: REGENERATE (%d failed claims) ---", len(failed_claims))

    if not failed_claims or not chunks:
        logger.info("Regenerate node called with 0 failed claims or 0 chunks. Skipping.")
        return {}

    failed_ids = [c.claim_id for c in failed_claims]

    # 1. Targeted partial re-generation
    corrected_raw_claims = regenerate_failed_claims(
        original_query=query,
        failed_claims=failed_claims,
        enriched_chunks=chunks,
    )

    if not corrected_raw_claims:
        logger.warning("Partial re-generation yielded 0 replacement claims.")
        return {}

    # 2. Map corrected claims to enriched context chunks
    temp_answer = GeneratedAnswer(claims=corrected_raw_claims)
    corrected_mapped_claims = map_claims_to_chunks(temp_answer, chunks)

    # 3. Positional merge with untouched verified claims
    merged_claims = merge_corrected_claims(
        original_claims=original_claims,
        failed_claim_ids=failed_ids,
        corrected_claims=corrected_mapped_claims,
    )

    logger.info(
        "Regenerate node complete: %d claims in merged list (%d corrected, %d preserved).",
        len(merged_claims),
        len(corrected_mapped_claims),
        len(merged_claims) - len(corrected_mapped_claims),
    )

    return {"claims": merged_claims}
