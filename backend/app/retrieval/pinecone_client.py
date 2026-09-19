"""
Reusable Pinecone client wrapper for vector operations.

Provides a singleton/cached client for interacting with the Pinecone vector index.
Used by ingestion, retrieval, and verification pipelines.
"""

from functools import lru_cache
from typing import Any, Mapping, Sequence

from pinecone import Index, Pinecone

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class PineconeClient:
    """Wrapper around the official Pinecone SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
    ) -> None:
        """Initialize the Pinecone wrapper.

        Args:
            api_key: Optional explicit API key; defaults to settings.PINECONE_API_KEY.
            index_name: Optional explicit index name; defaults to settings.PINECONE_INDEX_NAME.
        """
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.PINECONE_API_KEY
        self.index_name = (
            index_name if index_name is not None else settings.PINECONE_INDEX_NAME
        )
        self._pc: Pinecone | None = None
        self._index: Index | None = None

    @property
    def client(self) -> Pinecone:
        """Lazily initialize and return the Pinecone client instance."""
        if not self.api_key:
            raise ValueError(
                "PINECONE_API_KEY is not configured. Set it in backend/.env"
            )
        if self._pc is None:
            logger.info("Initializing Pinecone client...")
            self._pc = Pinecone(api_key=self.api_key)
        return self._pc

    @property
    def index(self) -> Index:
        """Lazily initialize and return the Pinecone Index instance."""
        if not self.index_name:
            raise ValueError(
                "PINECONE_INDEX_NAME is not configured. Set it in backend/.env"
            )
        if self._index is None:
            logger.info("Connecting to Pinecone index: '%s'...", self.index_name)
            self._index = self.client.Index(self.index_name)
        return self._index

    def upsert_vectors(
        self,
        vectors: Sequence[dict[str, Any] | tuple],
        namespace: str = "",
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Upsert vectors with metadata into the Pinecone index.

        Args:
            vectors: Sequence of vector items. Each item can be:
                - dict: {"id": str, "values": list[float], "metadata": dict}
                - tuple: (id, values, metadata) or (id, values)
            namespace: The namespace to write to (default: root namespace "").
            batch_size: Optional batch size for chunked upserts.

        Returns:
            Dictionary containing upsert result summary (e.g. {'upserted_count': N}).
        """
        logger.info(
            "Upserting %d vectors into index '%s' (namespace='%s')...",
            len(vectors),
            self.index_name,
            namespace,
        )
        response = self.index.upsert(
            vectors=vectors,
            namespace=namespace,
            batch_size=batch_size,
        )
        count = getattr(response, "upserted_count", len(vectors))
        logger.info("Successfully upserted %s vectors.", count)
        return {"upserted_count": count}

    def query_vectors(
        self,
        vector: Sequence[float],
        top_k: int = 5,
        filter: Mapping[str, Any] | None = None,
        namespace: str = "",
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        """Query the vector index for similar vectors.

        Args:
            vector: Query dense vector (e.g. 384-dimensional list of floats).
            top_k: Maximum number of closest vectors to retrieve.
            filter: Optional metadata filtering dictionary (e.g. {"document_id": "doc1"}).
            namespace: Namespace to query within.
            include_metadata: Whether to return chunk metadata along with IDs and scores.

        Returns:
            List of match dictionaries, each containing:
                - id: str
                - score: float (cosine similarity)
                - metadata: dict | None
        """
        logger.debug(
            "Querying index '%s' with top_k=%d (namespace='%s', filter=%s)...",
            self.index_name,
            top_k,
            namespace,
            filter,
        )
        response = self.index.query(
            vector=vector,
            top_k=top_k,
            filter=filter,
            namespace=namespace,
            include_metadata=include_metadata,
        )
        raw_matches = getattr(response, "matches", []) or []
        matches: list[dict[str, Any]] = []
        for m in raw_matches:
            matches.append(
                {
                    "id": getattr(m, "id", None),
                    "score": getattr(m, "score", None),
                    "metadata": getattr(m, "metadata", None),
                }
            )
        logger.debug("Found %d matches for query vector.", len(matches))
        return matches

    def delete_vectors(
        self,
        ids: Sequence[str] | None = None,
        delete_all: bool = False,
        namespace: str = "",
        filter: Mapping[str, Any] | None = None,
    ) -> None:
        """Delete vectors by IDs, filter, or delete all in a namespace.

        Args:
            ids: Optional list of vector IDs to remove.
            delete_all: If True, deletes all vectors within the target namespace.
            namespace: Target namespace (default: root namespace "").
            filter: Optional metadata filter identifying records to delete.
        """
        if delete_all:
            logger.warning(
                "Deleting ALL vectors from index '%s' (namespace='%s').",
                self.index_name,
                namespace,
            )
            self.index.delete(delete_all=True, namespace=namespace)
        elif ids:
            logger.info(
                "Deleting %d vectors by ID from index '%s'.",
                len(ids),
                self.index_name,
            )
            self.index.delete(ids=list(ids), namespace=namespace)
        elif filter:
            logger.info(
                "Deleting vectors matching filter from index '%s'.",
                self.index_name,
            )
            self.index.delete(filter=filter, namespace=namespace)

    def get_index_stats(
        self, filter: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Fetch index statistics including total vector count and dimensions.

        Args:
            filter: Optional filter to restrict statistics.

        Returns:
            Dictionary containing:
                - dimension: int | None
                - total_vector_count: int
                - namespaces: dict[str, dict]
                - index_fullness: float
        """
        logger.debug("Fetching index stats for index '%s'...", self.index_name)
        stats = self.index.describe_index_stats(filter=filter)
        namespaces = {}
        raw_ns = getattr(stats, "namespaces", None)
        if raw_ns:
            for k, v in raw_ns.items():
                namespaces[k] = {"vector_count": getattr(v, "vector_count", 0)}

        return {
            "dimension": getattr(stats, "dimension", None),
            "total_vector_count": getattr(stats, "total_vector_count", 0),
            "namespaces": namespaces,
            "index_fullness": getattr(stats, "index_fullness", 0.0),
        }


@lru_cache(maxsize=1)
def get_pinecone_client() -> PineconeClient:
    """Get or create the singleton PineconeClient instance.

    Uses lru_cache so client connection is established once and reused
    across requests rather than reconnecting on every call.
    """
    return PineconeClient()
