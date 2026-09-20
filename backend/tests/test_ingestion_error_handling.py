"""
Unit and integration tests for Phase 14: Ingestion error handling & retries.

Tests:
1. PermanentIngestionError (corrupted PDF) fails immediately with exactly 1 attempt and retryable=False.
2. TransientIngestionError (flaky Pinecone client) retries automatically and succeeds on 2nd attempt.
3. TransientIngestionError exhausting 3 attempts sets status=failed with retryable=True.
4. Manual POST /api/ingest/{document_id}/retry endpoint re-runs and succeeds for retryable=True document.
5. Manual POST /api/ingest/{document_id}/retry endpoint rejects retryable=False document with HTTP 400.
"""

from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from fastapi import status
from fastapi.testclient import TestClient

from app.ingestion.exceptions import PermanentIngestionError
from app.ingestion.pipeline import run_ingestion_pipeline
from app.ingestion.storage import DocumentRegistry
from app.main import app
from app.models.schemas import DocumentStatus, PageText
from app.retrieval import PineconeBatchUpsertError, PineconeClient


@pytest.fixture
def mock_registry(tmp_path: Path) -> DocumentRegistry:
    """Provide a fresh DocumentRegistry isolated in temporary directory."""
    return DocumentRegistry(upload_dir=tmp_path)


@pytest.fixture
def test_client() -> TestClient:
    """Provide FastAPI test client."""
    return TestClient(app)


def test_permanent_error_no_retry(mock_registry: DocumentRegistry):
    """Verify that a PermanentIngestionError (corrupted PDF) fails immediately with 1 attempt and retryable=False."""
    pdf_path = mock_registry.upload_dir / "corrupt_doc" / "bad.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"NOT_A_REAL_PDF_HEADER_12345")

    mock_registry.save_document(
        document_id="corrupt_doc",
        filename="bad.pdf",
        file_bytes=b"NOT_A_REAL_PDF_HEADER_12345",
    )

    extract_call_count = 0

    def mock_extract(*args, **kwargs):
        nonlocal extract_call_count
        extract_call_count += 1
        raise PermanentIngestionError("corrupted or unreadable PDF file", stage="extracting")

    with patch("app.ingestion.pipeline.extract_raw_pages", side_effect=mock_extract):
        result = run_ingestion_pipeline("corrupt_doc", registry=mock_registry)

    # Must fail immediately with retryable=False and exactly 1 attempt
    assert extract_call_count == 1
    assert result.status == DocumentStatus.FAILED
    assert result.retryable is False
    assert "corrupted or unreadable" in (result.failure_reason or "")

    stored = mock_registry.get_document("corrupt_doc")
    assert stored is not None
    assert stored.status == DocumentStatus.FAILED
    assert stored.retryable is False


def test_transient_error_retries_and_succeeds(mock_registry: DocumentRegistry):
    """Verify that a TransientIngestionError retries via tenacity and succeeds on the 2nd attempt."""
    doc_id = "transient_success_doc"
    pdf_path = mock_registry.upload_dir / doc_id / "sample.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-dummy")

    mock_registry.save_document(doc_id, "sample.pdf", b"%PDF-dummy")

    mock_pages = [PageText(page_number=1, text="Sample text content for transient error test", char_count=43)]

    mock_pc = MagicMock(spec=PineconeClient)
    mock_pc.delete_vectors.return_value = None

    attempt_counter = 0

    def mock_upsert(*args, **kwargs):
        nonlocal attempt_counter
        attempt_counter += 1
        if attempt_counter == 1:
            raise PineconeBatchUpsertError("Pinecone connection reset", failed_ids=["vec-1"])
        return {"upserted_count": 1, "successful_ids": ["vec-1"], "failed_ids": []}

    mock_pc.upsert_vectors.side_effect = mock_upsert

    with patch("app.ingestion.pipeline.extract_raw_pages", return_value=mock_pages), \
         patch("tenacity.nap.sleep", return_value=None):
        result = run_ingestion_pipeline(
            doc_id,
            registry=mock_registry,
            pinecone_client=mock_pc,
            verify_upsert=False,
        )

    assert attempt_counter == 2
    assert result.status == DocumentStatus.READY
    assert result.failure_reason is None

    stored = mock_registry.get_document(doc_id)
    assert stored is not None
    assert stored.status == DocumentStatus.READY


def test_transient_error_exhausts_retries_marks_retryable(mock_registry: DocumentRegistry):
    """Verify that a TransientIngestionError that exhausts all 3 attempts sets retryable=True."""
    doc_id = "transient_exhaust_doc"
    pdf_path = mock_registry.upload_dir / doc_id / "sample.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-dummy")

    mock_registry.save_document(doc_id, "sample.pdf", b"%PDF-dummy")

    mock_pages = [PageText(page_number=1, text="Sample text for exhausted retries test", char_count=39)]

    mock_pc = MagicMock(spec=PineconeClient)
    mock_pc.delete_vectors.return_value = None

    attempt_count = 0

    def mock_upsert_count(*args, **kwargs):
        nonlocal attempt_count
        attempt_count += 1
        raise PineconeBatchUpsertError("Pinecone API 503 unavailable")

    mock_pc.upsert_vectors.side_effect = mock_upsert_count

    with patch("app.ingestion.pipeline.extract_raw_pages", return_value=mock_pages), \
         patch("tenacity.nap.sleep", return_value=None):
        result = run_ingestion_pipeline(
            doc_id,
            registry=mock_registry,
            pinecone_client=mock_pc,
            verify_upsert=False,
        )

    assert attempt_count == 3
    assert result.status == DocumentStatus.FAILED
    assert result.retryable is True
    assert "Pinecone API 503" in (result.failure_reason or "")

    stored = mock_registry.get_document(doc_id)
    assert stored is not None
    assert stored.status == DocumentStatus.FAILED
    assert stored.retryable is True


def test_retry_endpoint_rejects_permanent_failure(test_client: TestClient, tmp_path: Path):
    """Verify that POST /api/ingest/{document_id}/retry returns HTTP 400 for permanently failed document."""
    with patch("app.api.routes.ingest.get_document_registry") as mock_get_reg:
        reg = DocumentRegistry(upload_dir=tmp_path)
        mock_get_reg.return_value = reg

        reg.save_document("perm_doc", "scanned.pdf", b"dummy")
        reg.update_document_metadata(
            "perm_doc",
            status=DocumentStatus.FAILED,
            failure_reason="scanned PDF with no text",
            retryable=False,
        )

        response = test_client.post("/api/ingest/perm_doc/retry?sync=true")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "permanently failed" in data["detail"]


def test_retry_endpoint_succeeds_for_retryable_failure(test_client: TestClient, tmp_path: Path):
    """Verify that POST /api/ingest/{document_id}/retry purges vectors and re-runs pipeline for retryable document."""
    with patch("app.api.routes.ingest.get_document_registry") as mock_get_reg:
        reg = DocumentRegistry(upload_dir=tmp_path)
        mock_get_reg.return_value = reg

        pdf_path = tmp_path / "retry_doc" / "retry.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-dummy")

        reg.save_document("retry_doc", "retry.pdf", b"%PDF-dummy")
        reg.update_document_metadata(
            "retry_doc",
            status=DocumentStatus.FAILED,
            failure_reason="Pinecone timeout",
            retryable=True,
        )

        mock_pages = [PageText(page_number=1, text="Sample text for retry endpoint test", char_count=36)]

        mock_pc = MagicMock(spec=PineconeClient)
        mock_pc.delete_vectors.return_value = None
        mock_pc.upsert_vectors.return_value = {"upserted_count": 1, "successful_ids": ["vec-1"], "failed_ids": []}

        with patch("app.ingestion.pipeline.extract_raw_pages", return_value=mock_pages), \
             patch("app.ingestion.pipeline.get_pinecone_client", return_value=mock_pc), \
             patch("app.retrieval.get_pinecone_client", return_value=mock_pc), \
             patch("app.ingestion.pipeline.get_document_registry", return_value=reg):
            response = test_client.post("/api/ingest/retry_doc/retry?sync=true")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == DocumentStatus.READY
        assert data["document_id"] == "retry_doc"

        mock_pc.delete_vectors.assert_called_with(filter={"document_id": "retry_doc"})
