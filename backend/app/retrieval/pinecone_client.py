"""
Reusable Pinecone client wrapper for vector operations.

Provides a singleton/cached client for interacting with the Pinecone vector index.
Used by ingestion, retrieval, and verification pipelines.
"""

from functools import lru_cache
import time
from typing import Any, Mapping, Sequence

from pinecone import Index, Pinecone

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.exceptions import TransientIngestionError

logger = get_logger(__name__)

# Pinecone official recommendation: 100 vectors per batch maintains request payload < 2MB
DEFAULT_UPSERT_BATCH_SIZE: int = 100
MAX_UPSERT_RETRIES: int = 3
INITIAL_RETRY_DELAY: float = 0.5


class PineconeBatchUpsertError(TransientIngestionError):
    """Raised when one or more batches permanently fail during Pinecone upsert."""

    def __init__(
        self,
        message: str,
        successful_ids: list[str] | None = None,
        failed_ids: list[str] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message, stage="upserting")
        self.message = message
        self.successful_ids = successful_ids or []
        self.failed_ids = failed_ids or []
        self.original_exception = original_exception


def _get_vector_id(item: Any) -> str:
    """Extract string vector ID from dictionary or tuple vector payload."""
    if isinstance(item, dict):
        return str(item.get("id", ""))
    elif isinstance(item, (tuple, list)) and len(item) > 0:
        return str(item[0])
    return getattr(item, "id", "")


class PineconeClient:
    """Wrapper around the official Pinecone SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize the Pinecone wrapper.

        Args:
            api_key: Optional explicit API key; defaults to settings.PINECONE_API_KEY.
            index_name: Optional explicit index name; defaults to settings.PINECONE_INDEX_NAME.
            timeout: Optional explicit network timeout in seconds; defaults to settings.PINECONE_TIMEOUT_SECONDS.
        """
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.PINECONE_API_KEY
        self.index_name = (
            index_name if index_name is not None else settings.PINECONE_INDEX_NAME
        )
        self.timeout = (
            timeout if timeout is not None else getattr(settings, "PINECONE_TIMEOUT_SECONDS", 30.0)
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
            logger.info("Initializing Pinecone client (timeout=%.1fs)...", self.timeout)
            self._pc = Pinecone(api_key=self.api_key, timeout=self.timeout)
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
        batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
        max_retries: int = MAX_UPSERT_RETRIES,
        initial_backoff: float = INITIAL_RETRY_DELAY,
    ) -> dict[str, Any]:
        """Upsert vectors in batches with exponential backoff and partial failure tracking.

        Batching Strategy:
        - Defaults to 100 vectors per batch per official Pinecone SDK documentation recommendations
          to ensure encoded metadata payloads (e.g. source_text) remain safely under the 2MB request limit.
        - Retries failed batches up to max_retries attempts with exponential backoff.
        - On permanent failure, leaves successfully upserted batches in Pinecone, records exact
          succeeded vs failed IDs, and raises PineconeBatchUpsertError.

        Args:
            vectors: Sequence of vector items (dicts or tuples).
            namespace: The namespace to write to (default: root namespace "").
            batch_size: Number of vectors per batch request (default: 100).
            max_retries: Number of retry attempts per batch before failing.
            initial_backoff: Initial retry backoff delay in seconds.

        Returns:
            Dictionary containing:
                - upserted_count (int): Total successfully upserted vectors.
                - successful_ids (list[str]): List of all successfully persisted vector IDs.
                - failed_ids (list[str]): Empty on success.
        """
        if not vectors:
            return {"upserted_count": 0, "successful_ids": [], "failed_ids": []}

        effective_batch_size = max(1, batch_size)
        total_batches = (len(vectors) + effective_batch_size - 1) // effective_batch_size
        logger.info(
            "Upserting %d vectors into index '%s' in %d batch(es) (batch_size=%d, namespace='%s')...",
            len(vectors),
            self.index_name,
            total_batches,
            effective_batch_size,
            namespace,
        )

        successful_ids: list[str] = []
        failed_ids: list[str] = []
        total_upserted = 0

        for batch_num, start_idx in enumerate(range(0, len(vectors), effective_batch_size), start=1):
            batch = vectors[start_idx : start_idx + effective_batch_size]
            batch_ids = [_get_vector_id(v) for v in batch]

            batch_success = False
            last_error: Exception | None = None

            for attempt in range(1, max_retries + 1):
                try:
                    response = self.index.upsert(vectors=batch, namespace=namespace)
                    count = getattr(response, "upserted_count", len(batch))
                    total_upserted += count
                    successful_ids.extend(batch_ids)
                    batch_success = True
                    logger.debug(
                        "Batch %d/%d (%d vectors) upserted successfully on attempt %d.",
                        batch_num,
                        total_batches,
                        len(batch),
                        attempt,
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < max_retries:
                        backoff = initial_backoff * (2 ** (attempt - 1))
                        logger.warning(
                            "Batch %d/%d upsert attempt %d failed: %s. Retrying in %.2fs...",
                            batch_num,
                            total_batches,
                            attempt,
                            exc,
                            backoff,
                        )
                        time.sleep(backoff)
                    else:
                        logger.error(
                            "Batch %d/%d permanently failed after %d attempts: %s",
                            batch_num,
                            total_batches,
                            max_retries,
                            exc,
                        )

            if not batch_success:
                failed_ids.extend(batch_ids)
                # Any subsequent batches are also marked failed
                remaining_items = vectors[start_idx + effective_batch_size :]
                for rem in remaining_items:
                    failed_ids.append(_get_vector_id(rem))

                err_msg = (
                    f"Batch {batch_num}/{total_batches} permanently failed after {max_retries} attempts: {last_error}. "
                    f"Succeeded vectors: {len(successful_ids)}, Failed vectors: {len(failed_ids)}"
                )
                logger.error(err_msg)
                raise PineconeBatchUpsertError(
                    err_msg,
                    successful_ids=successful_ids,
                    failed_ids=failed_ids,
                    original_exception=last_error,
                )

        logger.info(
            "Successfully upserted all %d vectors across %d batch(es) into index '%s'.",
            total_upserted,
            total_batches,
            self.index_name,
        )
        return {
            "upserted_count": total_upserted,
            "successful_ids": successful_ids,
            "failed_ids": [],
        }

    def fetch_vectors(
        self,
        ids: Sequence[str],
        namespace: str = "",
    ) -> dict[str, dict[str, Any]]:
        """Fetch vectors and their metadata by ID from the Pinecone index.

        Args:
            ids: Sequence of vector IDs to retrieve.
            namespace: Namespace to fetch from (default: root namespace "").

        Returns:
            Dictionary mapping vector_id -> {"id": str, "values": list[float], "metadata": dict}.
        """
        if not ids:
            return {}

        logger.debug("Fetching %d vectors by ID from index '%s'...", len(ids), self.index_name)
        response = self.index.fetch(ids=list(ids), namespace=namespace)
        raw_vectors = getattr(response, "vectors", {}) or {}
        result: dict[str, dict[str, Any]] = {}
        for vid, vdata in raw_vectors.items():
            result[vid] = {
                "id": getattr(vdata, "id", vid),
                "values": list(getattr(vdata, "values", []) or []),
                "metadata": dict(getattr(vdata, "metadata", {}) or {}),
            }
        return result

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
