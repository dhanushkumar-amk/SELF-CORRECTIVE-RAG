"""Tests for Pinecone client, metadata serialization, and live connection smoke testing."""

import random
import time
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import get_settings
from app.models.schemas import ChunkMetadata
from app.retrieval.pinecone_client import PineconeClient, get_pinecone_client


# ── Metadata Schema Unit Tests ────────────────────────────────────────────────


class TestChunkMetadata:
    """Validate ChunkMetadata schema and Pinecone compatibility."""

    def test_chunk_metadata_serialization(self) -> None:
        """ChunkMetadata should serialize to Pinecone-compatible primitives."""
        meta = ChunkMetadata(
            document_id="doc-123",
            chunk_id="chunk-456",
            page_number=1,
            source_text="This is a test chunk.",
            char_start=0,
            char_end=21,
        )
        d = meta.to_pinecone_metadata()
        assert d["document_id"] == "doc-123"
        assert d["chunk_id"] == "chunk-456"
        assert d["page_number"] == 1
        assert d["page_number_end"] == 1
        assert d["source_text"] == "This is a test chunk."
        # char_start and char_end are explicitly excluded from Pinecone metadata
        assert "char_start" not in d
        assert "char_end" not in d
        # Confirm all types strictly conform to Pinecone metadata constraints:
        # string, number (int/float), boolean, or list of strings
        for key, val in d.items():
            assert isinstance(
                val, (str, int, float, bool)
            ), f"Field '{key}' has invalid Pinecone metadata type: {type(val)}"

    def test_chunk_metadata_deserialization(self) -> None:
        """ChunkMetadata should reconstruct correctly from raw dict."""
        raw = {
            "document_id": "doc-abc",
            "chunk_id": "chunk-xyz",
            "page_number": 3,
            "source_text": "Sample context text.",
            "char_start": 100,
            "char_end": 120,
        }
        meta = ChunkMetadata.from_pinecone_metadata(raw)
        assert meta.document_id == "doc-abc"
        assert meta.chunk_id == "chunk-xyz"
        assert meta.page_number == 3
        assert meta.source_text == "Sample context text."


# ── Pinecone Client Mocked Unit Tests ─────────────────────────────────────────


class TestPineconeClientUnit:
    """Verify PineconeClient wrapper methods with unit test isolation."""

    @patch("app.retrieval.pinecone_client.Pinecone")
    def test_client_methods_with_mock(self, mock_pinecone_cls: MagicMock) -> None:
        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pinecone_cls.return_value = mock_pc
        mock_pc.Index.return_value = mock_index

        client = PineconeClient(api_key="mock-key", index_name="mock-index")

        # Test upsert
        mock_index.upsert.return_value = MagicMock(upserted_count=5)
        res = client.upsert_vectors([{"id": "1", "values": [0.1] * 384}])
        assert res["upserted_count"] == 5
        mock_index.upsert.assert_called_once()

        # Test query
        mock_match = MagicMock()
        mock_match.id = "vec-1"
        mock_match.score = 0.95
        mock_match.metadata = {"document_id": "doc-1"}
        mock_index.query.return_value = MagicMock(matches=[mock_match])

        matches = client.query_vectors(vector=[0.1] * 384, top_k=1)
        assert len(matches) == 1
        assert matches[0]["id"] == "vec-1"
        assert matches[0]["score"] == 0.95
        assert matches[0]["metadata"]["document_id"] == "doc-1"

        # Test delete
        client.delete_vectors(ids=["vec-1"])
        mock_index.delete.assert_called_once_with(ids=["vec-1"], namespace="")

        # Test stats
        mock_index.describe_index_stats.return_value = MagicMock(
            dimension=384,
            total_vector_count=0,
            namespaces={},
            index_fullness=0.0,
        )
        stats = client.get_index_stats()
        assert stats["dimension"] == 384
        assert stats["total_vector_count"] == 0

    def test_missing_api_key_raises(self) -> None:
        """Client should raise ValueError if api_key is empty when connecting."""
        client = PineconeClient(api_key="", index_name="test-index")
        with pytest.raises(ValueError, match="PINECONE_API_KEY is not configured"):
            _ = client.client

    def test_missing_index_name_raises(self) -> None:
        """Client should raise ValueError if index_name is empty when connecting."""
        client = PineconeClient(api_key="valid-key", index_name="")
        with pytest.raises(ValueError, match="PINECONE_INDEX_NAME is not configured"):
            _ = client.index


# ── Live Integration Tests ───────────────────────────────────────────────────

# Determine if a live Pinecone API key is configured in the environment
_settings = get_settings(REQUIRE_API_KEYS=False)
_api_key = _settings.PINECONE_API_KEY
_has_live_key = bool(
    _api_key
    and not _api_key.startswith("your")
    and "placeholder" not in _api_key.lower()
)


@pytest.mark.skipif(
    not _has_live_key,
    reason="PINECONE_API_KEY not set or is a placeholder. Provide a valid key in backend/.env to run live tests.",
)
class TestPineconeLiveConnection:
    """Smoke test running against real Pinecone index."""

    @pytest.fixture(autouse=True)
    def setup_client(self) -> None:
        self.client = get_pinecone_client()
        self.test_prefix = f"test_phase3_{int(time.time())}"
        self.test_ids = [f"{self.test_prefix}_{i}" for i in range(5)]

    def test_live_index_connection_and_stats(self) -> None:
        """Verify connecting to the live index succeeds and returns index statistics."""
        stats = self.client.get_index_stats()
        assert "total_vector_count" in stats

    def test_live_upsert_query_delete_lifecycle(self) -> None:
        """Test full lifecycle: upsert 5 dummy vectors, query them back, and clean up."""
        random.seed(42)
        dummy_vectors = []
        for i, vec_id in enumerate(self.test_ids):
            # Generate random 384-dimensional unit vector
            raw_vec = [random.uniform(-1.0, 1.0) for _ in range(384)]
            norm = sum(x**2 for x in raw_vec) ** 0.5
            norm_vec = [x / norm for x in raw_vec]

            meta = ChunkMetadata(
                document_id=f"doc_{self.test_prefix}",
                chunk_id=f"chunk_{i}",
                page_number=i + 1,
                source_text=f"Sample text for test vector {i}",
                char_start=i * 100,
                char_end=(i + 1) * 100,
            )
            dummy_vectors.append(
                {
                    "id": vec_id,
                    "values": norm_vec,
                    "metadata": meta.to_pinecone_metadata(),
                }
            )

        # 1. Upsert 5 dummy vectors
        upsert_res = self.client.upsert_vectors(dummy_vectors)
        assert upsert_res["upserted_count"] == 5

        # Allow consistency propagation in Pinecone
        time.sleep(2)

        try:
            # 2. Query vectors
            query_res = self.client.query_vectors(
                vector=dummy_vectors[0]["values"],
                top_k=5,
                filter={"document_id": f"doc_{self.test_prefix}"},
                include_metadata=True,
            )
            assert len(query_res) > 0
            matched_ids = [m["id"] for m in query_res]
            assert dummy_vectors[0]["id"] in matched_ids
            top_match = query_res[0]
            assert "source_text" in top_match["metadata"]
        finally:
            # 3. Clean up — delete dummy vectors so no test vectors remain
            self.client.delete_vectors(ids=self.test_ids)
            time.sleep(1)

            # Confirm cleanup
            cleanup_query = self.client.query_vectors(
                vector=dummy_vectors[0]["values"],
                top_k=5,
                filter={"document_id": f"doc_{self.test_prefix}"},
            )
            assert len(cleanup_query) == 0
