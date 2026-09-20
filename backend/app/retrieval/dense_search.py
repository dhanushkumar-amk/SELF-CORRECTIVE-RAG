"""
Dense vector retrieval using Pinecone vector database.

Executes dense similarity search by generating a vector embedding for the input query
and querying the Pinecone index for nearest neighbors.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.ingestion.embedder import embed_query
from app.models.schemas import RetrievalResult
from app.retrieval.pinecone_client import PineconeClient, get_pinecone_client

logger = get_logger(__name__)


def dense_search(
    query: str,
    top_k: int = 5,
    filter: dict[str, Any] | None = None,
    pinecone_client: PineconeClient | None = None,
) -> list[RetrievalResult]:
    """Execute dense vector similarity search against Pinecone index.

    Args:
        query: Natural language query string.
        top_k: Maximum number of top matches to retrieve.
        filter: Optional metadata filtering dictionary (e.g. {"document_id": "doc1"}).
        pinecone_client: Optional explicit PineconeClient instance.

    Returns:
        List of RetrievalResult objects sorted by cosine similarity score descending.
    """
    if not query or not query.strip():
        return []

    client = pinecone_client or get_pinecone_client()
    query_vector = embed_query(query)
    matches = client.query_vectors(
        vector=query_vector,
        top_k=top_k,
        filter=filter,
        include_metadata=True,
    )

    results: list[RetrievalResult] = []
    for match in matches:
        chunk_id = str(match.get("id", ""))
        score = float(match.get("score", 0.0))
        metadata = dict(match.get("metadata") or {})
        results.append(
            RetrievalResult(
                chunk_id=chunk_id,
                score=score,
                metadata=metadata,
            )
        )

    logger.debug("dense_search returned %d result(s) for query '%s'.", len(results), query)
    return results
