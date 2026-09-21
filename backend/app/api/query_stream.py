"""
SSE Streaming Execution Module for Phase 43: Real-Time Self-Correction Progress Events.

Architecture & Design Decisions:
1. Event Schema:
   Emits structured SSE event blocks formatted as `event: <event_type>\ndata: <json_payload>\n\n`.
   Event Types:
   - `retrieval_started`: Query execution initialized.
   - `retrieval_complete`: Context chunks retrieved & reranked.
   - `generation_started`: Answer generation initiated.
   - `generation_complete`: Citation-forced claims produced.
   - `verification_started`: NLI claim verification initiated.
   - `verification_complete`: NLI claim statuses & confidence scores resolved.
   - `correction_triggered`: Unverified/contradicted claims detected; retry loop triggered.
   - `final_answer`: Execution complete; final synthesized text and aggregate status delivered.
   - `error`: Pipeline exception occurred; error payload delivered before stream closes.

2. Resilient Stream Exception Safeguard:
   Catches any mid-stream exception and yields a formatted `error` event before closing.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.generation.generator import generate_answer_with_citations
from app.generation.partial_regenerate import (
    merge_corrected_claims,
    regenerate_failed_claims,
)
from app.graph.finalize import finalize_node
from app.graph.regenerate import regenerate_node
from app.graph.routing import correction_router, get_failed_claims
from app.graph.targeted_retrieve import targeted_retrieve_node
from app.models.schemas import ClaimWithSource, GeneratedAnswer, RetrievalResult
from app.reranking.reranker import select_relevant_chunks
from app.retrieval.hybrid_retriever import hybrid_search
from app.verification.claim_mapper import map_claims_to_chunks
from app.verification.claim_verifier import verify_claims

logger = get_logger(__name__)

__all__ = [
    "stream_query_execution",
]


def _format_sse(event_type: str, payload: dict[str, Any]) -> str:
    """Format an event type and payload into a standard text/event-stream message string."""
    json_str = json.dumps(payload, default=str)
    return f"event: {event_type}\ndata: {json_str}\n\n"


async def stream_query_execution(
    query: str,
    document_id: str | None = None,
) -> AsyncIterator[str]:
    """Execute RAG query through self-correction graph and stream progress as SSE events.

    Args:
        query: Natural language query string.
        document_id: Optional UUID for single-document scoping.

    Yields:
        Formatted SSE event strings (`event: <type>\ndata: <json>\n\n`).
    """
    start_time = time.perf_counter()

    try:
        # ── 1. RETRIEVAL STAGE ─────────────────────────────────────────────
        yield _format_sse(
            "retrieval_started",
            {
                "stage": "retrieval",
                "query": query,
                "document_id": document_id,
            },
        )

        filter_dict = {"document_id": document_id} if document_id else None
        raw_candidates = hybrid_search(query=query, filter=filter_dict)
        rerank_res = select_relevant_chunks(query=query, candidates=raw_candidates)
        chunks: list[RetrievalResult] = rerank_res.chunks

        yield _format_sse(
            "retrieval_complete",
            {
                "stage": "retrieval",
                "chunk_count": len(chunks),
                "chunks": [c.model_dump() for c in chunks],
            },
        )

        # ── 2. GENERATION STAGE ────────────────────────────────────────────
        yield _format_sse(
            "generation_started",
            {
                "stage": "generation",
                "chunk_count": len(chunks),
            },
        )

        if not chunks:
            generated_answer = GeneratedAnswer(
                claims=[],
                insufficient_information=True,
                raw_response="No relevant context chunks found.",
            )
        else:
            generated_answer = generate_answer_with_citations(query=query, chunks=chunks)

        yield _format_sse(
            "generation_complete",
            {
                "stage": "generation",
                "claim_count": len(generated_answer.claims) + len(generated_answer.unverified_claims),
            },
        )

        # ── 3. VERIFICATION & CORRECTION LOOP ─────────────────────────────
        retry_count = 0
        max_retries = settings.CORRECTION_MAX_RETRIES
        current_answer: GeneratedAnswer | None = generated_answer
        current_chunks = chunks
        current_claims: list[ClaimWithSource] = []

        while True:
            yield _format_sse(
                "verification_started",
                {
                    "stage": "verification",
                    "retry_count": retry_count,
                },
            )

            # Map claims if initial pass, or use existing updated claims
            if current_claims:
                claims_to_verify = current_claims
            elif current_answer:
                claims_to_verify = map_claims_to_chunks(current_answer, current_chunks)
            else:
                claims_to_verify = []

            verified_claims = verify_claims(claims_to_verify)
            current_claims = verified_claims

            yield _format_sse(
                "verification_complete",
                {
                    "stage": "verification",
                    "claims": [c.model_dump() for c in verified_claims],
                    "retry_count": retry_count,
                },
            )

            # Evaluate routing decision
            temp_state = {
                "claims": current_claims,
                "retry_count": retry_count,
                "max_retries": max_retries,
            }
            route = correction_router(temp_state)

            if route == "finalize":
                fin = finalize_node(temp_state)
                duration_ms = (time.perf_counter() - start_time) * 1000.0

                yield _format_sse(
                    "final_answer",
                    {
                        "stage": "finalize",
                        "final_status": fin["final_status"],
                        "final_answer_text": fin["final_answer_text"],
                        "claims": [c.model_dump() for c in current_claims],
                        "retry_count": retry_count,
                        "retrieved_chunks": [c.model_dump() for c in current_chunks],
                        "latency_ms": round(duration_ms, 2),
                    },
                )
                break

            # Correction route triggered!
            failed = get_failed_claims(current_claims)
            yield _format_sse(
                "correction_triggered",
                {
                    "stage": "correction",
                    "retry_count": retry_count + 1,
                    "max_retries": max_retries,
                    "failed_claims": [c.model_dump() for c in failed],
                },
            )

            # Targeted retrieve
            t_res = targeted_retrieve_node(
                {
                    "claims": current_claims,
                    "retrieved_chunks": current_chunks,
                    "retry_count": retry_count,
                }
            )
            current_chunks = t_res["retrieved_chunks"]
            retry_count = t_res["retry_count"]

            # Partial regenerate
            r_res = regenerate_node(
                {
                    "query": query,
                    "claims": current_claims,
                    "retrieved_chunks": current_chunks,
                }
            )
            current_claims = r_res.get("claims", current_claims)

    except Exception as exc:
        logger.error("Unhandled exception during SSE stream execution: %s", exc, exc_info=True)
        yield _format_sse(
            "error",
            {
                "stage": "error",
                "error": type(exc).__name__,
                "detail": str(exc),
            },
        )
