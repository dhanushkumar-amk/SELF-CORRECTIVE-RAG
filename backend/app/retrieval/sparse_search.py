"""
Sparse keyword retrieval using BM25 in-memory index.

Executes keyword relevance search using rank_bm25 against indexed text chunks.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.models.schemas import RetrievalResult
from app.retrieval.bm25_index import BM25Index, get_bm25_index

logger = get_logger(__name__)


def sparse_search(
    query: str,
    top_k: int = 5,
    filter: dict[str, Any] | None = None,
    bm25_index: BM25Index | None = None,
) -> list[RetrievalResult]:
    """Execute sparse BM25 keyword search against in-memory BM25 index.

    Args:
        query: Natural language query string.
        top_k: Maximum number of top matches to retrieve.
        filter: Optional metadata filtering dictionary (e.g. {"document_id": "doc1"}).
        bm25_index: Optional explicit BM25Index instance; defaults to global active index.

    Returns:
        List of RetrievalResult objects sorted by BM25 relevance score descending.
    """
    if not query or not query.strip():
        return []

    idx = bm25_index if bm25_index is not None else get_bm25_index()
    if idx is None:
        logger.warning("BM25 index is not initialized. sparse_search returning empty results.")
        return []

    matches = idx.search(query=query, top_k=top_k)

    results: list[RetrievalResult] = []
    for match in matches:
        chunk_id = str(match.get("id", ""))
        score = float(match.get("score", 0.0))
        metadata = dict(match.get("metadata") or {})

        # Apply metadata filtering if specified
        if filter:
            match_filter = True
            for k, v in filter.items():
                if metadata.get(k) != v:
                    match_filter = False
                    break
            if not match_filter:
                continue

        results.append(
            RetrievalResult(
                chunk_id=chunk_id,
                score=score,
                metadata=metadata,
            )
        )

    logger.debug("sparse_search returned %d result(s) for query '%s'.", len(results), query)
    return results
