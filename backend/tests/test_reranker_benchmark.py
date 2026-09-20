"""
Regression guard test for Phase 24: Reranker Performance Benchmarks.
"""

import time
from app.models.schemas import RetrievalResult
from app.reranking import rerank


def test_reranker_latency_ceiling_guard():
    """Verify reranker latency stays comfortably below the 2,000 ms performance ceiling guard for K=20 candidates."""
    query = "Sentence-transformers MiniLM dense embedding dimensions and BM25 hybrid search performance"
    candidates = [
        RetrievalResult(
            chunk_id=f"c_{i}",
            score=0.1,
            metadata={
                "source_text": f"Chunk {i}: Sentence transformers all-MiniLM-L6-v2 produces 384 dimensional vector embeddings for semantic document retrieval."
            }
        )
        for i in range(20)
    ]

    # Warmup
    rerank(query, candidates, top_n=20)

    # Benchmark 3 runs
    start_t = time.perf_counter()
    for _ in range(3):
        res = rerank(query, candidates, top_n=20)
        assert len(res) == 20

    duration_ms = ((time.perf_counter() - start_t) / 3.0) * 1000.0

    # Regression guard: 20 candidates must be reranked in under 2000 ms (typically ~80-150ms on CPU)
    assert duration_ms < 2000.0, f"Reranker latency regression! Took {duration_ms:.2f} ms for 20 candidates."
