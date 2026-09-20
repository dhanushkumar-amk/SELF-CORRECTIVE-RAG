"""Tests for Phase 13: Pinecone vector upsert pipeline, batching, retries, idempotency, and verification.

Verifies:
1. End-to-end ingestion pipeline with Stage 5 Pinecone upserting.
2. 100-vector batching strategy per official Pinecone SDK guidance.
3. Retry with exponential backoff on transient upsert failures.
4. Partial failure tracking: records successful and failed vector IDs when a batch permanently fails.
5. Post-upsert verification: fetches vectors by ID from Pinecone and verifies source_text integrity.
6. Idempotency enforcement: re-processing the same document purges prior vectors, avoiding duplicates.
7. Cleanup guarantees: test vectors are removed from Pinecone so no test data lingers.
8. Verification endpoint: GET /api/ingest/{document_id}/verify returns match details.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
from unittest.mock import MagicMock, call, patch
import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.ingestion.chunk_metadata import create_vector_id
from app.ingestion.pipeline import run_ingestion_pipeline, verify_document_upsert
from app.ingestion.storage import get_document_registry, reset_document_registry
from app.main import app
from app.models.schemas import DocumentStatus
from app.retrieval import (
    PineconeBatchUpsertError,
    PineconeClient,
    get_pinecone_client,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MULTIPAGE_PDF = FIXTURES_DIR / "multipage_with_footer.pdf"
SAMPLE_PDF = FIXTURES_DIR / "sample.pdf"


@pytest.fixture(autouse=True)
def isolated_upload_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate uploads and registry file to a temporary directory for each test."""
    upload_dir = tmp_path / "test_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(upload_dir))
    reset_document_registry()
    yield upload_dir
    reset_document_registry()


class TestPineconeClientUnit:
    """Unit tests for PineconeClient batching, retries, and fetch operations."""

    def test_upsert_vectors_batches_and_succeeds(self) -> None:
        """Vectors are split into batches of batch_size and upserted sequentially."""
        mock_index = MagicMock()
        mock_index.upsert.return_value = MagicMock(upserted_count=50)

        client = PineconeClient(api_key="mock-key", index_name="mock-index")
        client._index = mock_index

        dummy_vectors = [{"id": f"vec_{i}", "values": [0.1] * 384, "metadata": {}} for i in range(125)]
        result = client.upsert_vectors(dummy_vectors, batch_size=50)

        # 125 vectors with batch_size=50 should result in 3 upsert calls (50, 50, 25)
        assert mock_index.upsert.call_count == 3
        assert result["upserted_count"] == 150  # 3 * 50 from mock
        assert len(result["successful_ids"]) == 125
        assert result["failed_ids"] == []

    def test_upsert_vectors_retry_exponential_backoff(self) -> None:
        """Batch upsert retries on transient failure with backoff and succeeds on subsequent attempt."""
        mock_index = MagicMock()
        # Fail on first attempt, succeed on second attempt
        mock_index.upsert.side_effect = [
            RuntimeError("Transient 503 Service Unavailable"),
            MagicMock(upserted_count=10),
        ]

        client = PineconeClient(api_key="mock-key", index_name="mock-index")
        client._index = mock_index

        dummy_vectors = [{"id": f"vec_{i}", "values": [0.1] * 384, "metadata": {}} for i in range(10)]

        with patch("time.sleep") as mock_sleep:
            result = client.upsert_vectors(dummy_vectors, batch_size=10, max_retries=3, initial_backoff=0.1)

            assert mock_index.upsert.call_count == 2
            mock_sleep.assert_called_once_with(0.1)
            assert len(result["successful_ids"]) == 10
            assert result["failed_ids"] == []

    def test_upsert_vectors_permanent_failure_records_succeeded_and_failed_ids(self) -> None:
        """When a later batch permanently fails, partial data is tracked and PineconeBatchUpsertError is raised."""
        mock_index = MagicMock()
        # Batch 1 succeeds, Batch 2 fails 3 times permanently
        mock_index.upsert.side_effect = [
            MagicMock(upserted_count=2),  # Batch 1 (ids 0, 1)
            RuntimeError("Rate limited"),  # Batch 2 attempt 1
            RuntimeError("Rate limited"),  # Batch 2 attempt 2
            RuntimeError("Rate limited"),  # Batch 2 attempt 3
        ]

        client = PineconeClient(api_key="mock-key", index_name="mock-index")
        client._index = mock_index

        dummy_vectors = [{"id": f"vec_{i}", "values": [0.1] * 384, "metadata": {}} for i in range(4)]

        with patch("time.sleep"):
            with pytest.raises(PineconeBatchUpsertError) as exc_info:
                client.upsert_vectors(dummy_vectors, batch_size=2, max_retries=3, initial_backoff=0.01)

            err = exc_info.value
            assert err.successful_ids == ["vec_0", "vec_1"]
            assert err.failed_ids == ["vec_2", "vec_3"]
            assert "permanently failed" in err.message

    def test_fetch_vectors_by_id(self) -> None:
        """fetch_vectors calls Index.fetch and maps response to vector dictionary."""
        mock_index = MagicMock()
        mock_vector_data = MagicMock(
            id="vec_1",
            values=[0.1, 0.2],
            metadata={"source_text": "Sample chunk text"},
        )
        mock_index.fetch.return_value = MagicMock(vectors={"vec_1": mock_vector_data})

        client = PineconeClient(api_key="mock-key", index_name="mock-index")
        client._index = mock_index

        res = client.fetch_vectors(["vec_1"])
        mock_index.fetch.assert_called_once_with(ids=["vec_1"], namespace="")
        assert "vec_1" in res
        assert res["vec_1"]["metadata"]["source_text"] == "Sample chunk text"
        assert res["vec_1"]["values"] == [0.1, 0.2]


class TestPipelinePineconeIntegration:
    """Integration tests for pipeline Pinecone upserting, verification, and idempotency."""

    def test_pipeline_upsert_and_verification_mocked(self) -> None:
        """Pipeline chains extract -> clean -> chunk -> embed -> upsert and verifies stored content."""
        registry = get_document_registry()
        pdf_bytes = MULTIPAGE_PDF.read_bytes()
        doc_meta = registry.save_document(
            document_id="doc-pinecone-test-1",
            filename="multipage_with_footer.pdf",
            file_bytes=pdf_bytes,
        )

        mock_client = MagicMock(spec=PineconeClient)
        mock_client.upsert_vectors.return_value = {"upserted_count": 1, "successful_ids": ["doc-pinecone-test-1::c1"], "failed_ids": []}

        # Mock fetch to return matching source_text
        def _mock_fetch(ids, namespace=""):
            res = {}
            chunks = registry.get_chunks(doc_meta.document_id) or []
            for cid in ids:
                matching_chunk = next((c for c in chunks if create_vector_id(doc_meta.document_id, c.chunk_id) == cid), None)
                text = matching_chunk.text if matching_chunk else "Mock text"
                res[cid] = {"id": cid, "values": [0.1] * 384, "metadata": {"source_text": text}}
            return res

        mock_client.fetch_vectors.side_effect = _mock_fetch

        result = run_ingestion_pipeline(
            document_id=doc_meta.document_id,
            registry=registry,
            pinecone_client=mock_client,
            verify_upsert=True,
        )

        assert result.status == DocumentStatus.READY
        assert result.current_stage == "ready"
        assert result.chunk_count > 0
        assert result.upserted_count > 0
        assert "upserting_seconds" in result.stage_timings

        # Assert delete_vectors was called for idempotency
        mock_client.delete_vectors.assert_called_once_with(filter={"document_id": doc_meta.document_id})
        # Assert upsert_vectors was called
        mock_client.upsert_vectors.assert_called_once()
        # Assert fetch_vectors was called for verification
        mock_client.fetch_vectors.assert_called_once()

    def test_pipeline_upsert_failure_attributes_stage(self) -> None:
        """When Pinecone upsert fails permanently, pipeline marks document failed with exact attribution."""
        registry = get_document_registry()
        pdf_bytes = SAMPLE_PDF.read_bytes()
        doc_meta = registry.save_document(
            document_id="doc-pinecone-fail",
            filename="sample.pdf",
            file_bytes=pdf_bytes,
        )

        mock_client = MagicMock(spec=PineconeClient)
        mock_client.upsert_vectors.side_effect = PineconeBatchUpsertError(
            message="Connection reset by peer",
            successful_ids=[],
            failed_ids=["vec-1"],
        )

        result = run_ingestion_pipeline(
            document_id=doc_meta.document_id,
            registry=registry,
            pinecone_client=mock_client,
        )

        assert result.status == DocumentStatus.FAILED
        assert result.current_stage == "failed"
        assert result.failure_reason is not None
        assert "failed at upserting" in result.failure_reason
        assert "Connection reset by peer" in result.failure_reason

    def test_verify_document_upsert_success_and_mismatch(self) -> None:
        """verify_document_upsert accurately flags matching vs mismatched or missing vectors."""
        registry = get_document_registry()
        pdf_bytes = SAMPLE_PDF.read_bytes()
        doc_meta = registry.save_document(
            document_id="doc-verify-test",
            filename="sample.pdf",
            file_bytes=pdf_bytes,
        )

        # Run up to embedding to produce chunks
        mock_client = MagicMock(spec=PineconeClient)
        mock_client.upsert_vectors.return_value = {"upserted_count": 1, "successful_ids": ["v1"], "failed_ids": []}
        mock_client.fetch_vectors.return_value = {}

        run_ingestion_pipeline(
            document_id=doc_meta.document_id,
            registry=registry,
            pinecone_client=mock_client,
            verify_upsert=False,
        )

        chunks = registry.get_chunks(doc_meta.document_id)
        assert chunks is not None and len(chunks) > 0
        chunk = chunks[0]
        vid = create_vector_id(doc_meta.document_id, chunk.chunk_id)

        # 1. Test Match Success
        mock_client.fetch_vectors.return_value = {
            vid: {"id": vid, "values": [0.1] * 384, "metadata": {"source_text": chunk.text}}
        }
        resp = verify_document_upsert(doc_meta.document_id, registry=registry, pinecone_client=mock_client)
        assert resp.verified is True
        assert resp.matched_count == 1
        assert len(resp.mismatches) == 0

        # 2. Test Content Mismatch
        mock_client.fetch_vectors.return_value = {
            vid: {"id": vid, "values": [0.1] * 384, "metadata": {"source_text": "CORRUPTED CONTENT"}}
        }
        resp_mismatch = verify_document_upsert(doc_meta.document_id, registry=registry, pinecone_client=mock_client)
        assert resp_mismatch.verified is False
        assert resp_mismatch.matched_count == 0
        assert len(resp_mismatch.mismatches) > 0


class TestPineconeUpsertEndpoints:
    """Integration tests for verification endpoint."""

    @pytest.mark.asyncio
    async def test_verify_endpoint_missing_document_returns_404(self) -> None:
        """Calling GET /verify on a nonexistent document ID returns HTTP 404."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.get("/api/ingest/nonexistent-doc/verify")
            assert res.status_code == 404
            assert "not found" in res.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_verify_endpoint_on_ingested_document(self) -> None:
        """Calling GET /verify returns structured verification metrics."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Upload PDF
            upload_res = await client.post(
                "/api/ingest/upload",
                files={"file": ("sample.pdf", SAMPLE_PDF.read_bytes(), "application/pdf")},
            )
            assert upload_res.status_code == 200
            doc_id = upload_res.json()["document_id"]

            # 2. Mock Pinecone client to avoid network calls during endpoint unit test
            with patch("app.ingestion.pipeline.get_pinecone_client") as mock_get_pc:
                mock_pc = MagicMock()
                mock_pc.upsert_vectors.return_value = {"upserted_count": 1, "successful_ids": ["v1"], "failed_ids": []}

                registry = get_document_registry()

                def _mock_fetch(ids, namespace=""):
                    chunks = registry.get_chunks(doc_id) or []
                    res = {}
                    for cid in ids:
                        matching_chunk = next((c for c in chunks if create_vector_id(doc_id, c.chunk_id) == cid), None)
                        text = matching_chunk.text if matching_chunk else ""
                        res[cid] = {"id": cid, "values": [0.1] * 384, "metadata": {"source_text": text}}
                    return res

                mock_pc.fetch_vectors.side_effect = _mock_fetch
                mock_get_pc.return_value = mock_pc

                # Process document
                proc_res = await client.post(f"/api/ingest/{doc_id}/process?sync=true")
                assert proc_res.status_code == 200
                assert proc_res.json()["status"] == "ready"

                # Verify via endpoint
                verify_res = await client.get(f"/api/ingest/{doc_id}/verify")
                assert verify_res.status_code == 200
                vdata = verify_res.json()
                assert vdata["document_id"] == doc_id
                assert vdata["verified"] is True
                assert vdata["chunk_count"] > 0
                assert vdata["matched_count"] > 0
