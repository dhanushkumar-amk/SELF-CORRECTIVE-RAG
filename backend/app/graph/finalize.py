"""
Finalize Node for Phase 41: Aggregate Status & Answer Synthesis.

Architecture & Design Decisions:
1. System Final Status Calculation (`finalize_node`):
   - Evaluates the terminal claim verification state post retry loop.
   - Status options:
     - "fully_verified": ALL claims are verified with high NLI confidence.
     - "partially_verified": Mix of verified and unverified/contradicted claims (retry limit reached).
     - "unverifiable": 0 claims verified or missing context/citations.

2. Transparent Final Answer Synthesis:
   - Assembles `final_answer_text` by concatenating clean claim texts in sequence.
   - Attaches `get_claim_final_status()` per claim for accurate UI status badge rendering.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.graph.state import RAGState
from app.verification.claim_verifier import get_claim_final_status

logger = get_logger(__name__)

__all__ = [
    "finalize_node",
]


def finalize_node(state: RAGState) -> dict[str, Any]:
    """LangGraph Node: Synthesize final user-facing answer text and calculate aggregate system status.

    Args:
        state: Terminal LangGraph state dictionary (RAGState).

    Returns:
        Dictionary containing `final_status` and `final_answer_text`.
    """
    claims = state.get("claims", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    logger.info("--- LANGGRAPH NODE: FINALIZE (%d claims, retry %d/%d) ---", len(claims), retry_count, max_retries)

    if not claims:
        logger.warning("Finalize node called with 0 claims.")
        return {
            "final_status": "unverifiable",
            "final_answer_text": "No valid factual statements could be verified from the provided context.",
        }

    statuses = [get_claim_final_status(c) for c in claims]
    verified_count = sum(1 for s in statuses if s == "verified")
    total_count = len(claims)

    if verified_count == total_count:
        final_status = "fully_verified"
    elif verified_count > 0:
        final_status = "partially_verified"
    else:
        final_status = "unverifiable"

    final_answer_text = " ".join(c.claim_text.strip() for c in claims if c.claim_text).strip()

    logger.info(
        "Finalize node complete: System Final Status = '%s' (%d/%d claims verified).",
        final_status,
        verified_count,
        total_count,
    )

    return {
        "final_status": final_status,
        "final_answer_text": final_answer_text,
    }
