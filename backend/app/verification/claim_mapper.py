"""
Claim-to-Chunk Resolution & Context Mapping Layer for Phase 31.

Architecture & Design Decisions:
1. Integrated Splitting + Resolution Pipeline:
   Combines `ensure_atomic_claims()` sentence decomposition with in-memory chunk metadata
   resolution into a single entry-point function `map_claims_to_chunks()`.

2. Pre-fetched In-Memory Chunk Lookup:
   Resolves claims directly against `retrieved_chunks` already in memory from the retrieval/reranking phase.
   Avoids redundant Pinecone queries or DB lookups.

3. Invalid / Hallucinated Source Citation Sentinel:
   If a claim's `source_chunk_id` does not exist in `retrieved_chunks` (e.g. LLM invented a fake ID),
   maps the claim to a `ClaimWithSource` instance with `is_valid_source=False` and `source_text=None`.
   This allows Phase 33's NLI verification engine to immediately flag the claim as unverifiable
   without attempting NLI inference against a non-existent premise.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.models.schemas import Claim, ClaimWithSource, GeneratedAnswer, RetrievalResult
from app.verification.claim_splitter import ensure_atomic_claims

logger = get_logger(__name__)

__all__ = [
    "map_claims_to_chunks",
]


def map_claims_to_chunks(
    answer: GeneratedAnswer,
    retrieved_chunks: list[RetrievalResult],
    include_unverified: bool = True,
) -> list[ClaimWithSource]:
    """Map generated answer claims to their resolved source chunk texts and metadata.

    First decomposes compound claims into atomic single-sentence claims, then resolves
    each claim's `source_chunk_id` against `retrieved_chunks`.

    Args:
        answer: GeneratedAnswer output from generator layer.
        retrieved_chunks: List of RetrievalResult context chunks fetched during retrieval/reranking.
        include_unverified: If True, also includes answer.unverified_claims in mapping.

    Returns:
        List of atomic ClaimWithSource objects ready for NLI premise-hypothesis verification.
    """
    if not answer or (not answer.claims and not answer.unverified_claims):
        logger.debug("map_claims_to_chunks called with empty or claimless GeneratedAnswer.")
        return []

    # 1. Combine valid and unverified claims if requested
    raw_claims: list[Claim] = list(answer.claims)
    if include_unverified and answer.unverified_claims:
        raw_claims.extend(answer.unverified_claims)

    # 2. Decompose any compound multi-sentence claims into atomic claims
    atomic_claims = ensure_atomic_claims(raw_claims)

    # 3. Build fast lookup map for retrieved chunks
    chunk_map: dict[str, RetrievalResult] = {c.chunk_id: c for c in retrieved_chunks}

    resolved_claims: list[ClaimWithSource] = []

    # 4. Resolve each atomic claim against in-memory chunk map
    for claim in atomic_claims:
        chunk = chunk_map.get(claim.source_chunk_id)

        if chunk:
            meta = chunk.metadata or {}
            source_text = str(meta.get("source_text") or meta.get("text") or "")
            page_num = meta.get("page_number")
            page_end = meta.get("page_number_end", page_num)
            doc_id = meta.get("document_id")
            filename = meta.get("filename")

            resolved = ClaimWithSource(
                claim_text=claim.claim_text,
                source_chunk_id=claim.source_chunk_id,
                source_text=source_text,
                page_number=int(page_num) if page_num is not None else None,
                page_number_end=int(page_end) if page_end is not None else None,
                is_valid_source=True,
                document_id=str(doc_id) if doc_id is not None else None,
                filename=str(filename) if filename is not None else None,
            )
        else:
            logger.warning(
                "Claim cited non-existent chunk_id '%s'. Marking is_valid_source=False.",
                claim.source_chunk_id,
            )
            resolved = ClaimWithSource(
                claim_text=claim.claim_text,
                source_chunk_id=claim.source_chunk_id,
                source_text=None,
                page_number=None,
                page_number_end=None,
                is_valid_source=False,
                document_id=None,
                filename=None,
            )

        resolved_claims.append(resolved)

    logger.info(
        "Mapped %d atomic claims (%d valid source, %d invalid/hallucinated source).",
        len(resolved_claims),
        sum(1 for c in resolved_claims if c.is_valid_source),
        sum(1 for c in resolved_claims if not c.is_valid_source),
    )

    return resolved_claims
