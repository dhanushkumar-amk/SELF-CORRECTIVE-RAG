"""
Targeted Re-Retrieval Module for Phase 39: Self-Correction Context Enrichment.

Architecture & Design Decisions:
1. Targeted Claim-As-Query Search (`targeted_retrieve_node`):
   - Rather than re-running the broad user query, targeted re-retrieval uses the exact
     failed `claim.claim_text` as a focused search query.
   - Executes hybrid dense + sparse retrieval + CrossEncoder reranking for each failed claim independently.

2. Context Merging & Deduplication:
   - Newly retrieved chunks are merged into `state.retrieved_chunks` while preserving original order
     and deduplicating by `chunk_id`.

3. "Nothing New Found" Logging:
   - If targeted retrieval returns context chunks that are already in `retrieved_chunks` (or 0 new chunks),
     an explicit warning log is recorded. This signals that retrieval coverage is already saturated and
     the failure is likely a generation/interpretation issue.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.graph.routing import get_failed_claims
from app.graph.state import RAGState
from app.models.schemas import RetrievalResult
from app.reranking.reranker import select_relevant_chunks
from app.retrieval.hybrid_retriever import hybrid_search

logger = get_logger(__name__)

__all__ = [
    "targeted_retrieve_node",
]


def targeted_retrieve_node(state: RAGState) -> dict[str, Any]:
    """LangGraph Node: Execute targeted re-retrieval for failed claims and enrich state context.

    Args:
        state: Current LangGraph state dictionary (RAGState).

    Returns:
        Dictionary containing updated `retrieved_chunks` and incremented `retry_count`.
    """
    claims = state.get("claims", [])
    existing_chunks = state.get("retrieved_chunks", [])
    retry_count = state.get("retry_count", 0)

    failed_claims = get_failed_claims(claims)
    logger.info(
        "--- LANGGRAPH NODE: TARGETED RETRIEVE (Retry #%d for %d failed claims) ---",
        retry_count + 1,
        len(failed_claims),
    )

    if not failed_claims:
        logger.info("Targeted retrieve called with 0 failed claims. Skipping re-retrieval.")
        return {
            "retry_count": retry_count + 1,
            "retrieved_chunks": existing_chunks,
        }

    existing_ids = {c.chunk_id for c in existing_chunks if hasattr(c, "chunk_id")}
    newly_added_chunks: list[RetrievalResult] = []

    for failed_claim in failed_claims:
        claim_text = failed_claim.claim_text
        claim_id = failed_claim.claim_id
        logger.info(
            "Executing targeted search for failed claim [ID: %s]: '%s'...",
            claim_id,
            claim_text[:50],
        )

        # 1. Hybrid search using claim text as query
        raw_candidates = hybrid_search(query=claim_text)

        # 2. Threshold-based CrossEncoder reranking using claim text as query
        rerank_res = select_relevant_chunks(query=claim_text, candidates=raw_candidates)
        retrieved_for_claim = rerank_res.chunks

        # 3. Deduplicate against existing context chunks
        new_for_claim = 0
        for chunk in retrieved_for_claim:
            if chunk.chunk_id not in existing_ids:
                existing_ids.add(chunk.chunk_id)
                newly_added_chunks.append(chunk)
                new_for_claim += 1
            else:
                logger.info(
                    "Targeted re-retrieval for claim '%s' [ID: %s] returned chunk '%s' already in context.",
                    claim_text[:40],
                    claim_id,
                    chunk.chunk_id,
                )

        if new_for_claim == 0:
            logger.warning(
                "Targeted re-retrieval for failed claim '%s' [ID: %s] yielded NO new context chunks. "
                "(Original context already contains best available matches).",
                claim_text[:50],
                claim_id,
            )
        else:
            logger.info(
                "Targeted re-retrieval for failed claim '%s' [ID: %s] added %d new context chunks.",
                claim_text[:40],
                claim_id,
                new_for_claim,
            )

    merged_chunks = list(existing_chunks) + newly_added_chunks
    new_retry_count = retry_count + 1

    logger.info(
        "Targeted re-retrieval node complete: %d new chunks added (%d total context chunks). Retry count incremented to %d.",
        len(newly_added_chunks),
        len(merged_chunks),
        new_retry_count,
    )

    return {
        "retry_count": new_retry_count,
        "retrieved_chunks": merged_chunks,
    }
