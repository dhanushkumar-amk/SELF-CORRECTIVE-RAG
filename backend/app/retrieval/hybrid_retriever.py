"""
Hybrid Retriever Node for single, resilient hybrid search entrypoint.

Combines dense vector search (Pinecone) and sparse keyword search (BM25)
using Reciprocal Rank Fusion (RRF). Provides a single resilient entry point with:
- Input validation (empty / whitespace queries handled gracefully)
- Error handling & graceful degradation (if dense search fails, fallback to sparse; if sparse fails, fallback to dense)
- Observability & performance logging (tracks retrieval duration and result counts per modality)
"""

from __future__ import annotations

import time
from typing import Any

from app.core.logging import get_logger
from app.models.schemas import RetrievalResult
from app.retrieval.bm25_index import BM25Index
from app.retrieval.dense_search import dense_search
from app.retrieval.pinecone_client import PineconeClient
from app.retrieval.rrf_fusion import reciprocal_rank_fusion
from app.retrieval.sparse_search import sparse_search

logger = get_logger(__name__)


def hybrid_search(
    query: str,
    top_k: int = 5,
    filter: dict[str, Any] | None = None,
    k_rrf: int = 60,
    pinecone_client: PineconeClient | None = None,
    bm25_index: BM25Index | None = None,
) -> list[RetrievalResult]:
    """Execute hybrid retrieval combining dense vector search and sparse BM25 search via RRF.

    Args:
        query: Natural language query string.
        top_k: Maximum number of top fused chunks to return.
        filter: Optional metadata filtering dictionary (e.g. {"document_id": "doc1"}).
        k_rrf: Smoothing constant for Reciprocal Rank Fusion (default: 60).
        pinecone_client: Optional explicit PineconeClient instance.
        bm25_index: Optional explicit BM25Index instance.

    Returns:
        List of RetrievalResult objects sorted by fused RRF score in descending order.
    """
    if not query or not query.strip():
        logger.warning("Empty or whitespace query passed to hybrid_search; returning empty results.")
        return []

    if top_k <= 0:
        return []

    start_time = time.perf_counter()
    dense_results: list[RetrievalResult] = []
    sparse_results: list[RetrievalResult] = []

    # 1. Execute Dense Search with Graceful Fallback
    try:
        dense_results = dense_search(
            query=query,
            top_k=top_k * 2,  # retrieve extra candidates for fusion
            filter=filter,
            pinecone_client=pinecone_client,
        )
    except Exception as exc:
        logger.error("Dense vector search failed: %s. Degrading gracefully to sparse BM25 search.", exc)

    # 2. Execute Sparse BM25 Search with Graceful Fallback
    try:
        sparse_results = sparse_search(
            query=query,
            top_k=top_k * 2,  # retrieve extra candidates for fusion
            filter=filter,
            bm25_index=bm25_index,
        )
    except Exception as exc:
        logger.error("Sparse BM25 search failed: %s. Degrading gracefully to dense vector search.", exc)

    # 3. Handle total retrieval failure (both dense and sparse returned 0 results or failed)
    if not dense_results and not sparse_results:
        logger.warning("Both dense and sparse retrieval returned zero results for query: '%s'.", query)
        return []

    # 4. Perform Reciprocal Rank Fusion (RRF)
    result_lists: list[list[RetrievalResult]] = []
    if dense_results:
        result_lists.append(dense_results)
    if sparse_results:
        result_lists.append(sparse_results)

    fused_results = reciprocal_rank_fusion(result_lists, k=k_rrf)

    # Truncate to top_k requested results
    final_results = fused_results[:top_k]

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info(
        "hybrid_search completed in %.2f ms (dense=%d, sparse=%d, fused=%d, returned=%d).",
        elapsed_ms,
        len(dense_results),
        len(sparse_results),
        len(fused_results),
        len(final_results),
    )

    return final_results
