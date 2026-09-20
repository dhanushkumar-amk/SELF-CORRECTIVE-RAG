"""
Reciprocal Rank Fusion (RRF) for hybrid dense-sparse search results.

RRF is a model-free rank aggregation method that combines multiple ranked result
lists (e.g. dense vector search and sparse BM25 keyword search) into a single unified
ranked list.

Formula:
    RRF_score(chunk) = sum_{L in result_lists} (1 / (k + rank(chunk, L)))

Key Characteristics & Design Rationale:
1. Score Scale Independence: Dense search returns cosine similarity scores (typically 0.0 - 1.0),
   while BM25 returns unbounded keyword frequency scores (e.g. 0.0 - 50.0+). Comparing raw scores
   directly is mathematically invalid. RRF operates solely on rank positions (1-indexed), eliminating
   the need for score normalization or calibration.
2. Agreement Bonus: A chunk appearing in multiple retrieval lists receives score contributions from
   each list. This strongly rewards agreement between independent retrieval modalities.
3. Parameter k: Constant (standard default k = 60) that controls the decay rate of rank position weight.
   Higher k values smooth the impact of rank differences, while smaller k values give heavily front-loaded
   advantage to top-1/top-2 ranks.
4. Weighted RRF (Note): Standard RRF treats all input result lists with equal weight (1.0).
   Weighted RRF (giving dense or sparse more influence, e.g. w_dense * RRF_dense + w_sparse * RRF_sparse)
   is possible if empirical evaluation indicates one retrieval mode is significantly more reliable for
   specific domain workloads. We use standard unweighted RRF as the well-established baseline default.
"""

from __future__ import annotations

from typing import Sequence

from app.core.logging import get_logger
from app.models.schemas import RetrievalResult

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[RetrievalResult]],
    k: int = 60,
) -> list[RetrievalResult]:
    """Combine multiple ranked retrieval result lists using Reciprocal Rank Fusion (RRF).

    Args:
        result_lists: Sequence of ranked lists of RetrievalResult objects (e.g. [dense_results, sparse_results]).
                      Each list must be ordered from highest rank (rank 1 at index 0) to lowest rank.
        k: Smoothing constant parameter controlling rank decay (default: 60).

    Returns:
        A deduplicated list of RetrievalResult objects sorted by RRF score in descending order.
        The score field of each output RetrievalResult contains its fused RRF score.

    Raises:
        ValueError: If k <= 0.
    """
    if k <= 0:
        raise ValueError(f"RRF parameter k must be positive, got k={k}")

    if not result_lists:
        return []

    fused_scores: dict[str, float] = {}
    fused_metadata: dict[str, dict] = {}

    for list_idx, rank_list in enumerate(result_lists):
        for rank_zero_idx, item in enumerate(rank_list):
            rank = rank_zero_idx + 1  # 1-indexed position
            chunk_id = item.chunk_id

            # Calculate RRF score contribution for this rank position
            contribution = 1.0 / (k + rank)

            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = contribution
                fused_metadata[chunk_id] = item.metadata or {}
            else:
                fused_scores[chunk_id] += contribution
                # Verify metadata consistency across result lists (Task 3)
                existing_meta = fused_metadata[chunk_id]
                new_meta = item.metadata or {}
                if existing_meta and new_meta and existing_meta != new_meta:
                    logger.warning(
                        "Metadata mismatch for chunk_id '%s' between retrieval lists. "
                        "List %d metadata: %s, Previous metadata: %s",
                        chunk_id,
                        list_idx,
                        new_meta,
                        existing_meta,
                    )

    # Note: The score field of RetrievalResult is set to the final aggregated RRF score.
    # Original dense cosine similarity and sparse BM25 scores are replaced because raw scores
    # are no longer comparable or meaningful once fused.
    fused_results: list[RetrievalResult] = []
    for chunk_id, rrf_score in fused_scores.items():
        fused_results.append(
            RetrievalResult(
                chunk_id=chunk_id,
                score=rrf_score,
                metadata=fused_metadata[chunk_id],
            )
        )

    # Sort descending by RRF score; break ties deterministically by chunk_id
    fused_results.sort(key=lambda r: (-r.score, r.chunk_id))

    logger.debug(
        "RRF fusion combined %d list(s) into %d unique fused result(s).",
        len(result_lists),
        len(fused_results),
    )

    return fused_results
