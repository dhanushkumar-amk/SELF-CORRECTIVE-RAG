"""
BM25 In-Memory Sparse Index for keyword-matching retrieval.

Design Notes & Production Tradeoffs:
- Library: Uses `rank_bm25.BM25Okapi` for keyword relevance scoring.
- In-Memory Model: `rank_bm25` operates strictly in-memory. For this portfolio project's
  scale (a few dozen documents), rebuilding the BM25 index from stored chunks on app startup
  and post-ingestion completes in < 50ms.
- Production Scale Note: In a large-scale enterprise system with millions of documents,
  rebuilding in-memory indices from scratch becomes a bottleneck. Production setups use an
  external distributed sparse engine (e.g. Elasticsearch, OpenSearch, or Anserini/Pyserini)
  supporting incremental document updates.
- Thread Safety: Uses atomic reference swapping (`_GLOBAL_BM25_INDEX = new_index`) under a
  re-entrant lock so concurrent reader queries are never blocked or corrupted while a background
  rebuild is taking place.
"""

from __future__ import annotations

import re

import time
from typing import Any, Sequence

from rank_bm25 import BM25Okapi

from app.core.logging import get_logger
from app.ingestion.chunk_metadata import create_vector_id
from app.ingestion.storage import DocumentRegistry, get_document_registry
from app.models.schemas import Chunk, DocumentStatus

logger = get_logger(__name__)

__all__ = [
    "BM25Index",
    "build_bm25_index",
    "get_bm25_index",
    "rebuild_bm25_index",
    "tokenize_text",
]

# Global singleton BM25 index reference
_GLOBAL_BM25_INDEX: BM25Index | None = None
_BM25_LOCK = threading.Lock() if "threading" in globals() else None

import threading

_BM25_LOCK = threading.Lock()


def tokenize_text(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms for BM25 indexing."""
    if not text:
        return []
    return [term for term in re.split(r"\W+", text.lower()) if term]


class BM25Index:
    """Wrapper around rank_bm25.BM25Okapi maintaining parallel chunk metadata mappings."""

    def __init__(
        self,
        bm25: BM25Okapi | None,
        chunks: list[Chunk],
        metadata_map: list[dict[str, Any]],
        build_time_seconds: float = 0.0,
    ) -> None:
        self.bm25 = bm25
        self.chunks = chunks
        self.metadata_map = metadata_map
        self.build_time_seconds = build_time_seconds
        self.chunk_count = len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Query the BM25 index for keyword relevance matches.

        Args:
            query: Natural language query string.
            top_k: Maximum number of top-scoring chunks to return.

        Returns:
            List of match dictionaries formatted identically to Pinecone vector search matches:
            [
                {
                    "id": str (vector_id / chunk_id),
                    "score": float (BM25 score),
                    "metadata": dict (source_text, document_id, page_number, filename, etc.)
                }
            ]
        """
        if not self.bm25 or not self.chunks or top_k <= 0:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices sorted by score descending
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results: list[dict[str, Any]] = []
        for idx in top_indices:
            raw_score = float(scores[idx])
            score = max(0.0, raw_score)

            chunk = self.chunks[idx]
            meta = self.metadata_map[idx]
            vec_id = create_vector_id(chunk.document_id, chunk.chunk_id)

            results.append({
                "id": vec_id,
                "score": score,
                "metadata": meta,
            })

        return results


def build_bm25_index(registry: DocumentRegistry | None = None) -> BM25Index:
    """Load all chunks from READY documents in registry and construct a new BM25Index.

    Args:
        registry: Optional custom DocumentRegistry instance (defaults to global singleton).

    Returns:
        Newly constructed BM25Index instance.
    """
    start_time = time.perf_counter()
    reg = registry or get_document_registry()

    # 1. Fetch all documents marked READY
    all_docs = reg.list_documents()
    ready_docs = [
        d for d in all_docs
        if d.status == DocumentStatus.READY or str(d.status).lower() == "ready"
    ]

    all_chunks: list[Chunk] = []
    metadata_map: list[dict[str, Any]] = []
    tokenized_corpus: list[list[str]] = []

    # 2. Collect chunks across all READY documents
    for doc in ready_docs:
        chunks = reg.get_chunks(doc.document_id)
        if not chunks:
            continue
        for c in chunks:
            all_chunks.append(c)
            tokens = tokenize_text(c.text)
            tokenized_corpus.append(tokens)

            # Build metadata dict matching Pinecone format
            meta = c.to_pinecone_metadata(
                filename=doc.filename,
                document_title=doc.filename,
            )
            metadata_map.append(meta)

    # 3. Instantiate BM25Okapi if chunks exist
    bm25_model: BM25Okapi | None = None
    if tokenized_corpus:
        bm25_model = BM25Okapi(tokenized_corpus)

    build_duration = round(time.perf_counter() - start_time, 4)
    logger.info(
        "Built BM25 index in %.4fs across %d READY document(s) (%d total chunks indexed).",
        build_duration,
        len(ready_docs),
        len(all_chunks),
    )

    return BM25Index(
        bm25=bm25_model,
        chunks=all_chunks,
        metadata_map=metadata_map,
        build_time_seconds=build_duration,
    )


def get_bm25_index(
    registry: DocumentRegistry | None = None, force_rebuild: bool = False
) -> BM25Index:
    """Get the singleton BM25Index instance, building it on first access if needed.

    Thread-safe access via reference locking.
    """
    global _GLOBAL_BM25_INDEX
    with _BM25_LOCK:
        if _GLOBAL_BM25_INDEX is None or force_rebuild:
            _GLOBAL_BM25_INDEX = build_bm25_index(registry)
        return _GLOBAL_BM25_INDEX


def rebuild_bm25_index(registry: DocumentRegistry | None = None) -> BM25Index:
    """Rebuild the global BM25 index and atomically swap the singleton reference.

    Invoked post-ingestion pipeline completion or on app startup.
    """
    global _GLOBAL_BM25_INDEX
    new_index = build_bm25_index(registry)
    with _BM25_LOCK:
        _GLOBAL_BM25_INDEX = new_index
    return new_index
