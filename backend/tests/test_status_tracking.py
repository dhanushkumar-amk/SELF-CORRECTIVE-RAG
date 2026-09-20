"""
Unit and integration tests for Phase 15: Ingestion status tracking.

Tests:
1. Full state machine progression: process document, poll status, verify stage sequence & progress_percent ending at READY (100%).
2. State machine transition validation: validate_transition() rejects invalid state changes (e.g. UPLOADED -> READY).
3. Registry consistency auditor: check_registry_consistency() catches corrupted states (READY with failure_reason, FAILED missing retryable).
4. Thread safety & concurrency: 20 simultaneous threads performing registry updates without JSON corruption.
5. GET /api/ingest/documents listing endpoint returns summary info for documents in various states.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from fastapi import status
from fastapi.testclient import TestClient

from app.ingestion.pipeline import run_ingestion_pipeline
from app.ingestion.storage import DocumentRegistry, check_registry_consistency
from app.main import app
from app.models.schemas import (
    DocumentMetadata,
    DocumentStatus,
    PageText,
    calculate_progress_percent,
    validate_transition,
)
from app.retrieval import PineconeClient


@pytest.fixture
def mock_registry(tmp_path: Path) -> DocumentRegistry:
    """Provide a fresh DocumentRegistry isolated in temporary directory."""
    return DocumentRegistry(upload_dir=tmp_path)


@pytest.fixture
def test_client() -> TestClient:
    """Provide FastAPI test client."""
    return TestClient(app)


def test_full_state_machine_and_progress_percent(mock_registry: DocumentRegistry):
    """Verify document status progression through valid states ending at READY (100%)."""
    doc_id = "doc_status_progression"
    pdf_path = mock_registry.upload_dir / doc_id / "sample.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-dummy")

    doc = mock_registry.save_document(doc_id, "sample.pdf", b"%PDF-dummy")
    assert doc.status == DocumentStatus.UPLOADED
    assert calculate_progress_percent(doc.status, doc.current_stage) == 0

    mock_pages = [PageText(page_number=1, text="Sample text for state machine test", char_count=35)]
    mock_pc = MagicMock(spec=PineconeClient)
    mock_pc.delete_vectors.return_value = None
    mock_pc.upsert_vectors.return_value = {"upserted_count": 1, "successful_ids": ["v1"], "failed_ids": []}
    mock_pc.fetch_vectors.return_value = {
        f"vec_{doc_id}_chunk0": {
            "id": f"vec_{doc_id}_chunk0",
            "metadata": {"source_text": "Sample text for state machine test"},
        }
    }

    observed_stages: list[tuple[str, int]] = []

    original_update = mock_registry.update_document_metadata

    def tracking_update(document_id: str, **kwargs):
        res = original_update(document_id, **kwargs)
        if res:
            pct = calculate_progress_percent(res.status, res.current_stage)
            stage_name = res.current_stage or res.status.value
            observed_stages.append((stage_name, pct))
        return res

    mock_registry.update_document_metadata = tracking_update

    with patch("app.ingestion.pipeline.extract_raw_pages", return_value=mock_pages), \
         patch("app.ingestion.pipeline.get_pinecone_client", return_value=mock_pc):
        result = run_ingestion_pipeline(doc_id, registry=mock_registry, verify_upsert=False)

    assert result.status == DocumentStatus.READY
    final_doc = mock_registry.get_document(doc_id)
    assert final_doc is not None
    assert final_doc.status == DocumentStatus.READY

    stage_names = [s[0] for s in observed_stages]
    assert "extracting" in stage_names
    assert "cleaning" in stage_names
    assert "chunking" in stage_names
    assert "embedding" in stage_names
    assert "upserting" in stage_names
    assert "ready" in stage_names
    assert calculate_progress_percent(DocumentStatus.READY) == 100


def test_validate_transition_rejection(mock_registry: DocumentRegistry):
    """Verify validate_transition() permits valid transitions and rejects invalid state changes."""
    # Valid transitions
    assert validate_transition(DocumentStatus.UPLOADED, DocumentStatus.EXTRACTING) is True
    assert validate_transition(DocumentStatus.EXTRACTING, DocumentStatus.CLEANING) is True
    assert validate_transition(DocumentStatus.CLEANING, DocumentStatus.CHUNKING) is True
    assert validate_transition(DocumentStatus.CHUNKING, DocumentStatus.EMBEDDING) is True
    assert validate_transition(DocumentStatus.EMBEDDING, DocumentStatus.UPSERTING) is True
    assert validate_transition(DocumentStatus.UPSERTING, DocumentStatus.READY) is True
    assert validate_transition(DocumentStatus.READY, DocumentStatus.UPLOADED) is True

    # Invalid transitions
    assert validate_transition(DocumentStatus.READY, DocumentStatus.CHUNKING) is False
    assert validate_transition(DocumentStatus.UPLOADED, DocumentStatus.READY) is False
    assert validate_transition(DocumentStatus.EXTRACTING, DocumentStatus.READY) is False

    # Enforce rejection in registry updates
    mock_registry.save_document("test_trans_doc", "test.pdf", b"%PDF-dummy")

    with pytest.raises(ValueError) as exc_info:
        mock_registry.update_document_metadata("test_trans_doc", status=DocumentStatus.READY)

    assert "Invalid status transition" in str(exc_info.value)


def test_registry_consistency_checker():
    """Verify check_registry_consistency() detects corrupted document records."""
    # Valid READY document
    valid_ready = DocumentMetadata(
        document_id="doc1",
        filename="valid.pdf",
        upload_timestamp="2026-09-20T12:00:00Z",
        size_bytes=100,
        status=DocumentStatus.READY,
        chunk_count=5,
        upserted_count=5,
    )
    is_valid, violations = check_registry_consistency(valid_ready)
    assert is_valid is True
    assert len(violations) == 0

    # Corrupted READY document with failure_reason set
    corrupt_ready = DocumentMetadata(
        document_id="doc2",
        filename="corrupt.pdf",
        upload_timestamp="2026-09-20T12:00:00Z",
        size_bytes=100,
        status=DocumentStatus.READY,
        failure_reason="Some old error",
    )
    is_valid_2, violations_2 = check_registry_consistency(corrupt_ready)
    assert is_valid_2 is False
    assert any("failure_reason" in v for v in violations_2)

    # Corrupted FAILED document missing retryable flag
    corrupt_failed = DocumentMetadata(
        document_id="doc3",
        filename="corrupt_failed.pdf",
        upload_timestamp="2026-09-20T12:00:00Z",
        size_bytes=100,
        status=DocumentStatus.FAILED,
        failure_reason="Corrupted PDF stream",
        retryable=None,
    )
    is_valid_3, violations_3 = check_registry_consistency(corrupt_failed)
    assert is_valid_3 is False
    assert any("retryable" in v for v in violations_3)


def test_concurrent_registry_writes_thread_safety(tmp_path: Path):
    """Verify 20 concurrent threads performing registry updates leave registry.json strictly valid JSON."""
    reg = DocumentRegistry(upload_dir=tmp_path)

    # Initial document
    reg.save_document("concurrent_doc", "test.pdf", b"content")

    def worker_update(worker_id: int):
        for i in range(10):
            reg.update_document_metadata(
                "concurrent_doc",
                total_char_count=worker_id * 1000 + i,
            )

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(worker_update, w) for w in range(20)]
        for f in as_completed(futures):
            f.result()

    registry_file = tmp_path / "registry.json"
    assert registry_file.exists()

    content = registry_file.read_text(encoding="utf-8")
    parsed = json.loads(content)
    assert "concurrent_doc" in parsed
    assert parsed["concurrent_doc"]["filename"] == "test.pdf"


def test_list_documents_and_single_status_endpoint(test_client: TestClient, tmp_path: Path):
    """Verify GET /api/ingest/documents and GET /api/ingest/{document_id}/status return complete metadata."""
    with patch("app.api.routes.ingest.get_document_registry") as mock_get_reg:
        reg = DocumentRegistry(upload_dir=tmp_path)
        mock_get_reg.return_value = reg

        # Save documents with initial target status
        reg.save_document("doc_ready", "ready.pdf", b"%PDF-1", status=DocumentStatus.READY)
        reg.update_document_metadata(
            "doc_ready",
            current_stage="ready",
            chunk_count=3,
            upserted_count=3,
            processing_time_seconds=1.25,
        )

        reg.save_document("doc_failed", "bad.pdf", b"%PDF-2", status=DocumentStatus.FAILED)
        reg.update_document_metadata(
            "doc_failed",
            current_stage="failed",
            failure_reason="scanned PDF with no text",
            retryable=False,
        )

        # 1. Test GET /api/ingest/documents
        resp_list = test_client.get("/api/ingest/documents")
        assert resp_list.status_code == status.HTTP_200_OK
        list_data = resp_list.json()
        assert list_data["total"] == 2
        doc_ids = [d["document_id"] for d in list_data["documents"]]
        assert "doc_ready" in doc_ids
        assert "doc_failed" in doc_ids

        # 2. Test GET /api/ingest/{document_id}/status for READY document
        resp_status_1 = test_client.get("/api/ingest/doc_ready/status")
        assert resp_status_1.status_code == status.HTTP_200_OK
        s1 = resp_status_1.json()
        assert s1["document_id"] == "doc_ready"
        assert s1["filename"] == "ready.pdf"
        assert s1["status"] == "ready"
        assert s1["progress_percent"] == 100
        assert s1["chunk_count"] == 3
        assert s1["processing_time_seconds"] == 1.25

        # 3. Test GET /api/ingest/{document_id}/status for FAILED document
        resp_status_2 = test_client.get("/api/ingest/doc_failed/status")
        assert resp_status_2.status_code == status.HTTP_200_OK
        s2 = resp_status_2.json()
        assert s2["document_id"] == "doc_failed"
        assert s2["status"] == "failed"
        assert s2["failure_reason"] == "scanned PDF with no text"
        assert s2["retryable"] is False
