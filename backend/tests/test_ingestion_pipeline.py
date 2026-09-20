"""Tests for Phase 12: Embedding generation pipeline orchestrator and endpoints.

Verifies:
1. Full end-to-end pipeline execution (extract -> clean -> chunk -> embed -> ready).
2. Valid 384-dimensional dense embeddings attached to every Chunk and persisted to disk.
3. Stage progress tracking (uploaded -> extracting -> cleaning -> chunking -> embedding -> ready).
4. Stage failure attribution (corrupted PDF, password-protected PDF, empty/scanned PDF, chunking errors).
5. Background execution via FastAPI BackgroundTasks and live status polling.
6. Per-stage performance logging and latency timing breakdown.
7. Synchronous mode override and 404 handling.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch
import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.ingestion.pipeline import run_ingestion_pipeline
from app.ingestion.storage import get_document_registry, reset_document_registry
from app.main import app
from app.models.schemas import DocumentStatus

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample.pdf"
PROTECTED_PDF = FIXTURES_DIR / "protected.pdf"
MULTIPAGE_PDF = FIXTURES_DIR / "multipage_with_footer.pdf"
SCANNED_BLANK_PDF = FIXTURES_DIR / "scanned_blank.pdf"


@pytest.fixture(autouse=True)
def isolated_upload_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate uploads and registry file to a temporary directory for each test."""
    upload_dir = tmp_path / "test_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(upload_dir))
    reset_document_registry()
    yield upload_dir
    reset_document_registry()


class TestIngestionPipelineUnit:
    """Unit tests for the run_ingestion_pipeline function."""

    def test_full_pipeline_success_multipage_pdf(self) -> None:
        """Run pipeline on real multi-page PDF: verify ready status, chunk counts, and embeddings."""
        registry = get_document_registry()
        pdf_bytes = MULTIPAGE_PDF.read_bytes()
        doc_meta = registry.save_document(
            document_id="doc-multipage-123",
            filename="multipage_with_footer.pdf",
            file_bytes=pdf_bytes,
        )

        result = run_ingestion_pipeline(doc_meta.document_id)

        # 1. Verification of IngestionResult
        assert result.status == DocumentStatus.READY
        assert result.current_stage == "ready"
        assert result.failure_reason is None
        assert result.chunk_count > 0
        assert result.total_tokens > 0
        assert result.processing_time_seconds > 0

        # 2. Timing breakdown verification
        timings = result.stage_timings
        assert "extraction_seconds" in timings
        assert "cleaning_seconds" in timings
        assert "chunking_seconds" in timings
        assert "embedding_seconds" in timings
        assert "total_seconds" in timings
        assert timings["extraction_seconds"] >= 0
        assert timings["cleaning_seconds"] >= 0
        assert timings["chunking_seconds"] >= 0
        assert timings["embedding_seconds"] >= 0

        # 3. Document registry verification
        stored_doc = registry.get_document(doc_meta.document_id)
        assert stored_doc is not None
        assert stored_doc.status == DocumentStatus.READY
        assert stored_doc.current_stage == "ready"
        assert stored_doc.chunk_count == result.chunk_count
        assert stored_doc.total_tokens == result.total_tokens

        # 4. Chunk embedding verification on disk
        stored_chunks = registry.get_chunks(doc_meta.document_id)
        assert stored_chunks is not None
        assert len(stored_chunks) == result.chunk_count
        for chunk in stored_chunks:
            assert chunk.embedding is not None, f"Chunk {chunk.chunk_id} missing embedding"
            assert len(chunk.embedding) == 384, f"Chunk embedding dimension was {len(chunk.embedding)}, expected 384"
            assert all(isinstance(v, float) for v in chunk.embedding)

    def test_pipeline_failure_corrupted_pdf(self, isolated_upload_dir: Path) -> None:
        """Corrupted PDF must fail at extraction stage without continuing to chunk/embed."""
        registry = get_document_registry()
        # Create a corrupted PDF that has valid magic bytes but truncated garbage body
        corrupt_bytes = b"%PDF-1.4\ncorrupted garbage stream without xref or trailer"
        doc_meta = registry.save_document(
            document_id="doc-corrupt-123",
            filename="corrupt.pdf",
            file_bytes=corrupt_bytes,
        )

        result = run_ingestion_pipeline(doc_meta.document_id)

        assert result.status == DocumentStatus.FAILED
        assert result.current_stage == "failed"
        assert result.failure_reason is not None
        assert "failed at extraction" in result.failure_reason

        # Ensure no chunks were produced or saved
        stored_chunks = registry.get_chunks(doc_meta.document_id)
        assert stored_chunks is None

    def test_pipeline_failure_password_protected_pdf(self) -> None:
        """Password-protected PDF must fail cleanly at extraction with specific stage attribution."""
        registry = get_document_registry()
        pdf_bytes = PROTECTED_PDF.read_bytes()
        doc_meta = registry.save_document(
            document_id="doc-protected-123",
            filename="protected.pdf",
            file_bytes=pdf_bytes,
        )

        result = run_ingestion_pipeline(doc_meta.document_id)

        assert result.status == DocumentStatus.FAILED
        assert result.current_stage == "failed"
        assert result.failure_reason is not None
        assert "failed at extraction" in result.failure_reason
        assert "password-protected" in result.failure_reason

        stored_doc = registry.get_document(doc_meta.document_id)
        assert stored_doc is not None
        assert stored_doc.status == DocumentStatus.FAILED
        assert stored_doc.failure_reason == result.failure_reason

    def test_pipeline_failure_scanned_pdf(self) -> None:
        """Image-only/scanned PDF with no text must fail cleanly at extraction."""
        registry = get_document_registry()
        pdf_bytes = SCANNED_BLANK_PDF.read_bytes()
        doc_meta = registry.save_document(
            document_id="doc-scanned-123",
            filename="scanned_blank.pdf",
            file_bytes=pdf_bytes,
        )

        result = run_ingestion_pipeline(doc_meta.document_id)

        assert result.status == DocumentStatus.FAILED
        assert result.current_stage == "failed"
        assert result.failure_reason is not None
        assert "failed at extraction" in result.failure_reason
        assert "scanned" in result.failure_reason.lower() or "no extractable text" in result.failure_reason.lower()

    def test_pipeline_failure_at_chunking_stops_before_embedding(self) -> None:
        """If chunking fails, pipeline must attribute to chunking and never call embed_texts."""
        registry = get_document_registry()
        pdf_bytes = SAMPLE_PDF.read_bytes()
        doc_meta = registry.save_document(
            document_id="doc-chunk-fail-123",
            filename="sample.pdf",
            file_bytes=pdf_bytes,
        )

        with patch("app.ingestion.pipeline.chunk_document", side_effect=RuntimeError("Token limit explosion")):
            with patch("app.ingestion.pipeline.embed_texts") as mock_embed:
                result = run_ingestion_pipeline(doc_meta.document_id)

                # Verify failure attribution
                assert result.status == DocumentStatus.FAILED
                assert result.current_stage == "failed"
                assert result.failure_reason is not None
                assert "failed at chunking" in result.failure_reason
                assert "Token limit explosion" in result.failure_reason

                # Embedder must NOT be called
                mock_embed.assert_not_called()

    def test_pipeline_stage_timing_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Pipeline must log per-stage latency timings for performance profiling."""
        import logging
        caplog.set_level(logging.INFO)

        registry = get_document_registry()
        pdf_bytes = SAMPLE_PDF.read_bytes()
        doc_meta = registry.save_document(
            document_id="doc-logging-123",
            filename="sample.pdf",
            file_bytes=pdf_bytes,
        )

        result = run_ingestion_pipeline(doc_meta.document_id)
        assert result.status == DocumentStatus.READY

        # Check logs for stage announcements
        log_text = caplog.text
        assert "Stage 1/4 [Extracting]" in log_text
        assert "Stage 2/4 [Cleaning]" in log_text
        assert "Stage 3/4 [Chunking]" in log_text
        assert "Stage 4/4 [Embedding]" in log_text
        assert "Stage timing breakdown" in log_text


class TestIngestionEndpoints:
    """Integration tests for the /process and /status endpoints."""

    @pytest.mark.asyncio
    async def test_process_missing_document_returns_404(self) -> None:
        """Calling /process on a nonexistent document ID returns HTTP 404."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/ingest/nonexistent-id/process")
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_status_missing_document_returns_404(self) -> None:
        """Calling /status on a nonexistent document ID returns HTTP 404."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/ingest/nonexistent-id/status")
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_sync_process_endpoint(self) -> None:
        """Calling POST /process?sync=true returns completed IngestionResult synchronously."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Upload valid PDF
            upload_res = await client.post(
                "/api/ingest/upload",
                files={"file": ("sample.pdf", SAMPLE_PDF.read_bytes(), "application/pdf")},
            )
            assert upload_res.status_code == 200
            doc_id = upload_res.json()["document_id"]

            # 2. Trigger synchronous process
            proc_res = await client.post(f"/api/ingest/{doc_id}/process?sync=true")
            assert proc_res.status_code == 200
            data = proc_res.json()
            assert data["status"] == "ready"
            assert data["current_stage"] == "ready"
            assert data["chunk_count"] > 0
            assert data["total_tokens"] > 0
            assert "stage_timings" in data

    @pytest.mark.asyncio
    async def test_background_process_and_status_polling(self) -> None:
        """Calling POST /process returns immediately with 'processing' while worker finishes in background."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Upload document
            upload_res = await client.post(
                "/api/ingest/upload",
                files={"file": ("multipage.pdf", MULTIPAGE_PDF.read_bytes(), "application/pdf")},
            )
            assert upload_res.status_code == 200
            doc_id = upload_res.json()["document_id"]

            # 2. Trigger background process
            proc_res = await client.post(f"/api/ingest/{doc_id}/process")
            assert proc_res.status_code == 200
            proc_data = proc_res.json()
            assert proc_data["status"] == "processing"

            # 3. Immediately inspect /status - must show processing
            status_res = await client.get(f"/api/ingest/{doc_id}/status")
            assert status_res.status_code == 200
            status_data = status_res.json()
            # Status should be processing or already progressing
            assert status_data["status"] in ("processing", "ready")

            # 4. Poll until pipeline reaches "ready"
            max_retries = 50
            final_status_data = None
            for _ in range(max_retries):
                await asyncio.sleep(0.15)
                poll_res = await client.get(f"/api/ingest/{doc_id}/status")
                assert poll_res.status_code == 200
                poll_data = poll_res.json()
                if poll_data["status"] == "ready":
                    final_status_data = poll_data
                    break
                elif poll_data["status"] == "failed":
                    pytest.fail(f"Pipeline failed unexpectedly: {poll_data['failure_reason']}")

            assert final_status_data is not None, "Pipeline did not reach 'ready' within polling timeout"
            assert final_status_data["status"] == "ready"
            assert final_status_data["current_stage"] == "ready"
            assert final_status_data["chunk_count"] > 0
            assert final_status_data["stage_timings"] is not None
