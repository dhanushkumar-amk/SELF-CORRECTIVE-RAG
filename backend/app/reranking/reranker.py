"""
Local Cross-Encoder Reranker Engine using sentence-transformers.

Architecture & Design Rationale:
1. Bi-Encoder vs. Cross-Encoder Tradeoffs (Standard Production Retrieve-Then-Rerank Pattern):
   - Bi-Encoder (Retrieval Stage - Phase 11 & Phase 16):
     Encodes query and document chunks SEPARATELY into dense vector embeddings. Document
     vectors are indexed offline. At query time, inner-product/cosine comparison runs in
     O(1) ANN search time over millions of chunks.
     Trade-off: Extremely fast and scalable across huge corpora, but loses token-to-token
     cross-attention between query and document.
   - Cross-Encoder (Reranking Stage - Phase 22 & Phase 23):
     Concatenates query and document chunk into a single input sequence ([CLS] query [SEP] document [SEP])
     and passes it through all Transformer encoder layers. Self-attention heads perform full N^2
     cross-token attention between every query token and every document token in a single forward pass.
     Trade-off: Significantly higher relevance scoring accuracy, but computationally expensive O(K)
     forward passes. It cannot scale to searching an entire multi-million document corpus, which is why
     it is applied strictly as a reranker over hybrid_search()'s narrowed candidate set (K ~ 10-50).

2. Model Choice: cross-encoder/ms-marco-MiniLM-L-6-v2
   - Pre-trained on MS MARCO passage ranking task.
   - Outputs raw, unbounded logit scores reflecting sentence-level relevance.
   - Max Sequence Length: 512 tokens (matching BERT max position embeddings).

3. Single-Pass Batched Prediction:
   - Scores all (query, document) pairs in ONE batched model.predict() call rather than a sequential loop.

4. Relevance Threshold Filtering:
   - select_relevant_chunks() filters candidates against RERANK_MIN_SCORE BEFORE applying top_n truncation.
   - If no candidates exceed the threshold, returns an explicit reason flag: 'no_relevant_chunks_found'.
"""

from __future__ import annotations

import threading
from typing import Any

from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import RerankResult, RetrievalResult

logger = get_logger(__name__)

_reranker_lock = threading.Lock()
_reranker_instance: CrossEncoder | None = None

__all__ = [
    "get_cross_encoder_model",
    "get_reranker_info",
    "rerank",
    "select_relevant_chunks",
]


def get_cross_encoder_model(
    model_name: str = settings.RERANKER_MODEL_NAME,
) -> CrossEncoder:
    """Retrieve cached CrossEncoder model singleton, loading it on first call.

    Thread-safe initialization ensures only one model instance lives in memory.
    """
    global _reranker_instance
    if _reranker_instance is None:
        with _reranker_lock:
            if _reranker_instance is None:
                logger.info("Loading local cross-encoder reranker model '%s'...", model_name)
                model = CrossEncoder(model_name)
                logger.info(
                    "Cross-encoder model '%s' loaded successfully (max_length: %d).",
                    model_name,
                    getattr(model, "max_length", 512),
                )
                _reranker_instance = model
    return _reranker_instance


def get_reranker_info() -> dict[str, Any]:
    """Return runtime metadata and configuration of active cross-encoder reranker."""
    model = get_cross_encoder_model()
    return {
        "model_name": settings.RERANKER_MODEL_NAME,
        "max_length": getattr(model, "max_length", 512),
        "tokenizer_type": type(model.tokenizer).__name__ if hasattr(model, "tokenizer") else "Unknown",
    }


def rerank(
    query: str,
    candidates: list[RetrievalResult],
    top_n: int = settings.RERANK_TOP_N,
) -> list[RetrievalResult]:
    """Score candidate chunks using joint cross-encoder self-attention and re-sort by score.

    Args:
        query: Natural language query string.
        candidates: Candidate RetrievalResult objects from hybrid_search().
        top_n: Maximum number of top-scoring chunks to return.

    Returns:
        List of RetrievalResult objects with scores overwritten by cross-encoder logits,
        sorted descending by score.
    """
    if not candidates or not query or not query.strip():
        return []

    model = get_cross_encoder_model()

    # Extract source text from metadata for each candidate
    pairs: list[tuple[str, str]] = []
    for cand in candidates:
        text = str(
            cand.metadata.get("source_text")
            or cand.metadata.get("text")
            or ""
        )
        pairs.append((query, text))

    # Single batched CrossEncoder.predict call over all (query, text) pairs
    raw_scores = model.predict(pairs, batch_size=32, show_progress_bar=False)

    # Overwrite RetrievalResult.score with cross-encoder logit score
    reranked: list[RetrievalResult] = []
    for idx, cand in enumerate(candidates):
        logit_score = float(raw_scores[idx])
        # Re-create RetrievalResult with updated score
        reranked.append(
            RetrievalResult(
                chunk_id=cand.chunk_id,
                score=logit_score,
                metadata=cand.metadata,
            )
        )

    # Sort descending by cross-encoder logit score
    reranked.sort(key=lambda r: r.score, reverse=True)

    # Truncate to top_n (if candidates count < top_n, returns all candidates)
    result = reranked[:top_n]
    logger.debug(
        "Reranked %d candidates for query '%s' -> top %d returned (top score: %.4f).",
        len(candidates),
        query,
        len(result),
        result[0].score if result else 0.0,
    )
    return result


def select_relevant_chunks(
    query: str,
    candidates: list[RetrievalResult],
    min_score: float | None = None,
    top_n: int = settings.RERANK_TOP_N,
) -> RerankResult:
    """Filter candidates against min_score threshold BEFORE applying top_n truncation.

    Args:
        query: Natural language query string.
        candidates: Candidate RetrievalResult objects from hybrid_search().
        min_score: Minimum cross-encoder logit threshold (defaults to settings.RERANK_MIN_SCORE).
        top_n: Maximum number of relevant chunks to return.

    Returns:
        RerankResult schema containing filtered chunks and an optional reason flag.
    """
    total_cand = len(candidates)
    if not candidates or not query or not query.strip():
        return RerankResult(
            chunks=[],
            reason="no_relevant_chunks_found",
            total_candidates=total_cand,
            relevant_count=0,
        )

    threshold = min_score if min_score is not None else settings.RERANK_MIN_SCORE

    # 1. Rerank all candidates (pass top_n=len(candidates) to score all without early cutoff)
    all_scored = rerank(query=query, candidates=candidates, top_n=len(candidates))

    # 2. Filter out below-threshold chunks BEFORE top_n truncation
    relevant_chunks = [c for c in all_scored if c.score >= threshold]
    relevant_count = len(relevant_chunks)

    # 3. Handle "nothing is relevant" case explicitly
    if not relevant_chunks:
        logger.warning(
            "All %d candidate chunks for query '%s' fell below relevance threshold %.2f.",
            total_cand,
            query,
            threshold,
        )
        return RerankResult(
            chunks=[],
            reason="no_relevant_chunks_found",
            total_candidates=total_cand,
            relevant_count=0,
        )

    # 4. Truncate to top_n AFTER threshold filtering
    final_chunks = relevant_chunks[:top_n]
    logger.info(
        "select_relevant_chunks for query '%s': %d/%d passed threshold %.2f (top %d returned).",
        query,
        relevant_count,
        total_cand,
        threshold,
        len(final_chunks),
    )
    return RerankResult(
        chunks=final_chunks,
        reason=None,
        total_candidates=total_cand,
        relevant_count=relevant_count,
    )
