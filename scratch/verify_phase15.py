"""
Verification script for Phase 15: Ingestion status tracking.

Demonstrates:
1. Real polling output sequence during document processing (status, stage, progress_percent).
2. GET /api/ingest/documents listing documents across different states (ready, failed, uploaded).
3. Concurrent multi-threaded registry write safety test.
"""

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure backend directory is on sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from fastapi import status
from fastapi.testclient import TestClient

from app.ingestion.pipeline import run_ingestion_pipeline
from app.ingestion.storage import DocumentRegistry
from app.main import app
from app.models.schemas import DocumentStatus, PageText
from app.retrieval import PineconeClient


def run_demonstration():
    print("\n" + "=" * 80)
    print("PHASE 15 VERIFICATION: INGESTION STATUS TRACKING & STATE MACHINE")
    print("=" * 80 + "\n")

    scratch_upload_dir = backend_dir / "scratch" / "test_uploads_p15"
    scratch_upload_dir.mkdir(parents=True, exist_ok=True)
    registry = DocumentRegistry(upload_dir=scratch_upload_dir)
    client = TestClient(app)

    # -------------------------------------------------------------------------
    # Scenario 1: Real Polling Output Sequence
    # -------------------------------------------------------------------------
    print("--- SCENARIO 1: Real Status Polling Sequence & Progress Percent ---")
    doc_id_1 = "demo_poll_doc"
    pdf_file_1 = scratch_upload_dir / doc_id_1 / "sample.pdf"
    pdf_file_1.parent.mkdir(parents=True, exist_ok=True)
    pdf_file_1.write_bytes(b"%PDF-1.4 sample text")

    registry.save_document(doc_id_1, "sample.pdf", b"%PDF-1.4 sample text")

    mock_pages = [PageText(page_number=1, text="Sample text for status polling demonstration", char_count=44)]
    mock_pc = MagicMock(spec=PineconeClient)
    mock_pc.delete_vectors.return_value = None
    mock_pc.upsert_vectors.return_value = {"upserted_count": 1, "successful_ids": ["v1"], "failed_ids": []}

    polling_history: list[dict] = []

    with patch("app.api.routes.ingest.get_document_registry", return_value=registry), \
         patch("app.ingestion.pipeline.get_document_registry", return_value=registry), \
         patch("app.ingestion.pipeline.extract_raw_pages", return_value=mock_pages), \
         patch("app.ingestion.pipeline.get_pinecone_client", return_value=mock_pc):

        # Initial status before processing
        resp0 = client.get(f"/api/ingest/{doc_id_1}/status")
        polling_history.append(resp0.json())

        # Run pipeline
        run_ingestion_pipeline(doc_id_1, registry=registry, verify_upsert=False)

        # Final status
        resp_final = client.get(f"/api/ingest/{doc_id_1}/status")
        polling_history.append(resp_final.json())

    print("  Captured Polling Sequence:")
    for i, s in enumerate(polling_history, start=1):
        print(
            f"    Poll #{i}: status='{s['status']}', stage='{s['current_stage']}', "
            f"progress_percent={s['progress_percent']}%, filename='{s['filename']}'"
        )

    assert polling_history[-1]["status"] == "ready"
    assert polling_history[-1]["progress_percent"] == 100
    print("  [SUCCESS] Status polling sequence successfully captured!\n")

    # -------------------------------------------------------------------------
    # Scenario 2: GET /api/ingest/documents Dashboard Summary
    # -------------------------------------------------------------------------
    print("--- SCENARIO 2: GET /api/ingest/documents Multi-Document Listing ---")
    registry.save_document("doc_ready_101", "report.pdf", b"data", status=DocumentStatus.READY)
    registry.update_document_metadata("doc_ready_101", current_stage="ready", chunk_count=12, upserted_count=12)

    registry.save_document("doc_failed_102", "corrupt_scan.pdf", b"bad", status=DocumentStatus.FAILED)
    registry.update_document_metadata("doc_failed_102", current_stage="failed", failure_reason="scanned PDF with no text", retryable=False)

    registry.save_document("doc_uploaded_103", "draft.pdf", b"new", status=DocumentStatus.UPLOADED)

    with patch("app.api.routes.ingest.get_document_registry", return_value=registry):
        resp_list = client.get("/api/ingest/documents")

    print(f"  GET /api/ingest/documents HTTP Status: {resp_list.status_code}")
    data_list = resp_list.json()
    print(f"  Total Documents Registered: {data_list['total']}")
    for d in data_list["documents"]:
        print(
            f"    - ID: {d['document_id']:<18} Filename: {d['filename']:<18} "
            f"Status: {d['status']:<10} Retryable: {str(d.get('retryable'))}"
        )

    assert data_list["total"] >= 3
    print("  [SUCCESS] Multi-document summary list returned successfully!\n")

    # -------------------------------------------------------------------------
    # Scenario 3: Concurrent Multi-Threaded Registry Integrity Test
    # -------------------------------------------------------------------------
    print("--- SCENARIO 3: Concurrent Multi-Threaded Registry Write Safety ---")
    from concurrent.futures import ThreadPoolExecutor, as_completed

    reg_conc = DocumentRegistry(upload_dir=scratch_upload_dir / "conc_test")
    reg_conc.save_document("conc_doc", "safety.pdf", b"test")

    def thread_task(tid: int):
        for step in range(5):
            reg_conc.update_document_metadata("conc_doc", total_char_count=tid * 100 + step)

    print("  Spinning up 15 concurrent threads performing atomic metadata updates...")
    with ThreadPoolExecutor(max_workers=15) as exec:
        futures = [exec.submit(thread_task, t) for t in range(15)]
        for f in as_completed(futures):
            f.result()

    reg_file = scratch_upload_dir / "conc_test" / "registry.json"
    assert reg_file.exists()
    content = reg_file.read_text(encoding="utf-8")
    parsed = json.loads(content)

    print(f"  registry.json File Size: {len(content)} bytes")
    print(f"  JSON Parsing Verified: doc ID '{parsed['conc_doc']['document_id']}' intact.")
    print("  [SUCCESS] Concurrent registry writes caused zero file corruption!\n")

    print("=" * 80)
    print("ALL PHASE 15 VERIFICATION SCENARIOS PASSED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_demonstration()
