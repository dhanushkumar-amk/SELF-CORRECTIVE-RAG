"""
Verification script for Phase 14: Ingestion error handling & retries.

Runs 4 demonstration scenarios:
1. Transient failure recovery via Tenacity retries (mocked flaky Pinecone client).
2. Permanent failure immediate halt with 0 retries (corrupted PDF upload).
3. POST /api/ingest/{document_id}/retry rejecting permanently failed document (retryable=False -> HTTP 400).
4. POST /api/ingest/{document_id}/retry succeeding for transiently failed document (retryable=True -> HTTP 200).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure backend directory is on sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from fastapi import status
from fastapi.testclient import TestClient

from app.ingestion.exceptions import PermanentIngestionError, TransientIngestionError
from app.ingestion.pipeline import run_ingestion_pipeline
from app.ingestion.storage import DocumentRegistry
from app.main import app
from app.models.schemas import DocumentStatus, PageText
from app.retrieval import PineconeBatchUpsertError, PineconeClient


def run_demonstration():
    print("\n" + "=" * 80)
    print("PHASE 14 VERIFICATION: INGESTION ERROR HANDLING & RETRIES")
    print("=" * 80 + "\n")

    scratch_upload_dir = backend_dir / "scratch" / "test_uploads"
    scratch_upload_dir.mkdir(parents=True, exist_ok=True)
    registry = DocumentRegistry(upload_dir=scratch_upload_dir)
    client = TestClient(app)

    # -------------------------------------------------------------------------
    # Scenario 1: Transient Failure Recovery & Structured Logging
    # -------------------------------------------------------------------------
    print("--- SCENARIO 1: Transient Failure Recovery (Flaky Pinecone Connection) ---")
    doc_id_1 = "demo_transient_recovery"
    pdf_file = scratch_upload_dir / doc_id_1 / "demo.pdf"
    pdf_file.parent.mkdir(parents=True, exist_ok=True)
    pdf_file.write_bytes(b"%PDF-1.4 sample content")

    registry.save_document(doc_id_1, "demo.pdf", b"%PDF-1.4 sample content")

    mock_pages = [PageText(page_number=1, text="Self-Correcting RAG verification test document", char_count=46)]
    mock_pc = MagicMock(spec=PineconeClient)
    mock_pc.delete_vectors.return_value = None

    attempts = 0
    def flaky_upsert(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            print(f"  [Attempt {attempts}] Simulating Pinecone connection timeout (raising TransientIngestionError)...")
            raise PineconeBatchUpsertError("Pinecone connection timeout", failed_ids=["v1"])
        print(f"  [Attempt {attempts}] Pinecone connection restored! Upserting vectors successfully...")
        return {"upserted_count": 1, "successful_ids": ["v1"], "failed_ids": []}

    mock_pc.upsert_vectors.side_effect = flaky_upsert

    with patch("app.ingestion.pipeline.extract_raw_pages", return_value=mock_pages), \
         patch("tenacity.nap.sleep", return_value=None):
        res1 = run_ingestion_pipeline(doc_id_1, registry=registry, pinecone_client=mock_pc, verify_upsert=False)

    print(f"  --> Final Status: {res1.status.value}")
    print(f"  --> Total Attempts: {attempts}")
    print(f"  --> Retryable Flag: {res1.retryable}")
    assert res1.status == DocumentStatus.READY
    assert attempts == 2
    print("  [SUCCESS] Transient error recovered on 2nd attempt!\n")

    # -------------------------------------------------------------------------
    # Scenario 2: Permanent Failure (Corrupted PDF Upload)
    # -------------------------------------------------------------------------
    print("--- SCENARIO 2: Permanent Failure Immediate Halt (Corrupted PDF) ---")
    doc_id_2 = "demo_permanent_corrupt"
    pdf_file_2 = scratch_upload_dir / doc_id_2 / "corrupt.pdf"
    pdf_file_2.parent.mkdir(parents=True, exist_ok=True)
    pdf_file_2.write_bytes(b"INVALID_CORRUPTED_BYTES")

    registry.save_document(doc_id_2, "corrupt.pdf", b"INVALID_CORRUPTED_BYTES")

    extract_calls = 0
    def corrupt_extract(*args, **kwargs):
        nonlocal extract_calls
        extract_calls += 1
        print(f"  [Attempt {extract_calls}] Corrupted PDF header detected! Raising PermanentIngestionError...")
        raise PermanentIngestionError("corrupted or unreadable PDF file", stage="extracting")

    with patch("app.ingestion.pipeline.extract_raw_pages", side_effect=corrupt_extract):
        res2 = run_ingestion_pipeline(doc_id_2, registry=registry)

    print(f"  --> Final Status: {res2.status.value}")
    print(f"  --> Total Attempts: {extract_calls}")
    print(f"  --> Retryable Flag: {res2.retryable}")
    print(f"  --> Failure Reason: {res2.failure_reason}")
    assert res2.status == DocumentStatus.FAILED
    assert extract_calls == 1
    assert res2.retryable is False
    print("  [SUCCESS] Permanent error failed immediately with 0 retries & retryable=False!\n")

    # -------------------------------------------------------------------------
    # Scenario 3: POST /api/ingest/{document_id}/retry Rejection
    # -------------------------------------------------------------------------
    print("--- SCENARIO 3: Manual Retry Endpoint Rejects Permanent Failure ---")
    with patch("app.api.routes.ingest.get_document_registry", return_value=registry):
        resp3 = client.post(f"/api/ingest/{doc_id_2}/retry?sync=true")

    print(f"  --> Endpoint HTTP Status Code: {resp3.status_code}")
    print(f"  --> Response Detail: {resp3.json().get('detail')}")
    assert resp3.status_code == status.HTTP_400_BAD_REQUEST
    print("  [SUCCESS] Endpoint correctly rejected retry for retryable=False document!\n")

    # -------------------------------------------------------------------------
    # Scenario 4: POST /api/ingest/{document_id}/retry Acceptance
    # -------------------------------------------------------------------------
    print("--- SCENARIO 4: Manual Retry Endpoint Accepts & Processes Retryable Failure ---")
    doc_id_4 = "demo_manual_retry"
    pdf_file_4 = scratch_upload_dir / doc_id_4 / "retryable.pdf"
    pdf_file_4.parent.mkdir(parents=True, exist_ok=True)
    pdf_file_4.write_bytes(b"%PDF-1.4 sample content")

    registry.save_document(doc_id_4, "retryable.pdf", b"%PDF-1.4 sample content")
    registry.update_document_metadata(
        doc_id_4,
        status=DocumentStatus.FAILED,
        failure_reason="Pinecone connection timeout",
        retryable=True,
    )

    mock_pc_4 = MagicMock(spec=PineconeClient)
    mock_pc_4.delete_vectors.return_value = None
    mock_pc_4.upsert_vectors.return_value = {"upserted_count": 1, "successful_ids": ["v4"], "failed_ids": []}

    with patch("app.api.routes.ingest.get_document_registry", return_value=registry), \
         patch("app.ingestion.pipeline.get_document_registry", return_value=registry), \
         patch("app.ingestion.pipeline.extract_raw_pages", return_value=mock_pages), \
         patch("app.ingestion.pipeline.get_pinecone_client", return_value=mock_pc_4), \
         patch("app.retrieval.get_pinecone_client", return_value=mock_pc_4):
        resp4 = client.post(f"/api/ingest/{doc_id_4}/retry?sync=true")

    print(f"  --> Endpoint HTTP Status Code: {resp4.status_code}")
    print(f"  --> Final Document Status: {resp4.json().get('status')}")
    assert resp4.status_code == status.HTTP_200_OK
    assert resp4.json().get("status") == "ready"
    mock_pc_4.delete_vectors.assert_called_with(filter={"document_id": doc_id_4})
    print("  [SUCCESS] Vectors purged and document reprocessed successfully!\n")

    print("=" * 80)
    print("ALL PHASE 14 VERIFICATION SCENARIOS PASSED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_demonstration()
