"""Real End-to-End Verification Script for Phase 12.

Simulates real HTTP client calls (like curl):
1. Upload real multi-page PDF (multipage_with_footer.pdf)
2. Trigger POST /api/ingest/{document_id}/process
3. Poll GET /api/ingest/{document_id}/status showing stage progression
4. Display final "ready" status, chunk_count, total_tokens, and per-stage timing breakdown
5. Verify on-disk chunks.json has 384-dimensional embeddings attached to all chunks
6. Deliberately trigger failure with corrupted PDF and demonstrate stage attribution
"""

import asyncio
import json
from pathlib import Path
import sys
import time

from httpx import ASGITransport, AsyncClient

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.ingestion.storage import get_document_registry
from app.main import app

FIXTURES_DIR = backend_dir / "tests" / "fixtures"
MULTIPAGE_PDF = FIXTURES_DIR / "multipage_with_footer.pdf"


async def main() -> None:
    print("=" * 80)
    print("PHASE 12 END-TO-END VERIFICATION: EMBEDDING GENERATION PIPELINE")
    print("=" * 80)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://localhost:8000") as client:
        # ---------------------------------------------------------------------
        # 1. Upload real multi-page PDF
        # ---------------------------------------------------------------------
        print("\n[Step 1] Uploading real multi-page PDF fixture (multipage_with_footer.pdf)...")
        pdf_bytes = MULTIPAGE_PDF.read_bytes()
        upload_resp = await client.post(
            "/api/ingest/upload",
            files={"file": ("multipage_with_footer.pdf", pdf_bytes, "application/pdf")},
        )
        assert upload_resp.status_code == 200, f"Upload failed: {upload_resp.text}"
        doc_data = upload_resp.json()
        doc_id = doc_data["document_id"]
        print(f"  [OK] Upload Successful! Document ID: {doc_id} ({doc_data['size_bytes']} bytes)")

        # Initial Status
        status_resp = await client.get(f"/api/ingest/{doc_id}/status")
        print(f"  Initial Status: {status_resp.json()}")

        # ---------------------------------------------------------------------
        # 2. Trigger Pipeline Processing (BackgroundTasks)
        # ---------------------------------------------------------------------
        print(f"\n[Step 2] Triggering POST /api/ingest/{doc_id}/process...")
        proc_resp = await client.post(f"/api/ingest/{doc_id}/process")
        assert proc_resp.status_code == 200, f"Process trigger failed: {proc_resp.text}"
        proc_data = proc_resp.json()
        print(f"  [OK] Pipeline Triggered! Immediate Response:")
        print(f"    - Status: {proc_data['status']}")
        print(f"    - Current Stage: {proc_data['current_stage']}")

        # Preload embedding model singleton so first cold-start download does not delay polling
        from app.ingestion.embedder import get_embedding_model
        get_embedding_model()

        # ---------------------------------------------------------------------
        # 3. Poll GET /api/ingest/{document_id}/status showing progression
        # ---------------------------------------------------------------------
        print("\n[Step 3] Polling GET /api/ingest/{doc_id}/status for stage progression...")
        seen_stages = set()
        final_status = None
        max_polls = 100

        for poll_num in range(1, max_polls + 1):
            poll_resp = await client.get(f"/api/ingest/{doc_id}/status")
            status_data = poll_resp.json()
            curr_stage = status_data.get("current_stage")
            curr_status = status_data.get("status")

            if curr_stage not in seen_stages:
                seen_stages.add(curr_stage)
                print(f"  Poll #{poll_num}: status='{curr_status}', stage='{curr_stage}'")

            if curr_status == "ready":
                final_status = status_data
                break
            elif curr_status == "failed":
                print(f"  Poll #{poll_num}: FAILED with reason: {status_data.get('failure_reason')}")
                sys.exit(1)

            await asyncio.sleep(0.2)

        assert final_status is not None, "Pipeline did not reach 'ready' status within polling timeout."
        print(f"\n[Step 4] Final 'ready' status reached successfully!")
        print(f"  - Document ID: {final_status['document_id']}")
        print(f"  - Status: {final_status['status']}")
        print(f"  - Current Stage: {final_status['current_stage']}")
        print(f"  - Page Count: {final_status['page_count']}")
        print(f"  - Chunk Count: {final_status['chunk_count']}")
        print(f"  - Total Tokens: {final_status['total_tokens']}")
        print(f"  - Total Processing Time: {final_status['processing_time_seconds']}s")

        print("\n[Step 5] Per-Stage Timing Breakdown:")
        timings = final_status.get("stage_timings", {})
        for stage_name, duration in timings.items():
            print(f"    - {stage_name:25s}: {duration:.4f}s")

        # ---------------------------------------------------------------------
        # 4. Verify Chunks on Disk
        # ---------------------------------------------------------------------
        print("\n[Step 6] Inspecting chunks on disk via storage registry...")
        registry = get_document_registry()
        chunks = registry.get_chunks(doc_id)
        assert chunks is not None and len(chunks) == final_status['chunk_count']
        print(f"  [OK] Verified {len(chunks)} chunks loaded from disk.")
        for idx, chunk in enumerate(chunks):
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 384
            emb_preview = [round(x, 4) for x in chunk.embedding[:4]]
            print(f"    Chunk #{idx} ({chunk.chunk_id[:8]}...): tokens={chunk.token_count}, "
                  f"pages={chunk.page_number}->{chunk.page_number_end}, "
                  f"emb_dim={len(chunk.embedding)}, emb_prefix={emb_preview}...")

        # ---------------------------------------------------------------------
        # 5. Deliberate Failure Verification
        # ---------------------------------------------------------------------
        print("\n[Step 7] Deliberately triggering failure with a corrupted PDF...")
        corrupt_bytes = b"%PDF-1.4\nTRUNCATED_AND_MALFORMED_HEADER_WITHOUT_XREF"
        corrupt_upload = await client.post(
            "/api/ingest/upload",
            files={"file": ("corrupt.pdf", corrupt_bytes, "application/pdf")},
        )
        corrupt_id = corrupt_upload.json()["document_id"]
        print(f"  Uploaded corrupted PDF fixture as Document ID: {corrupt_id}")

        print(f"  Executing synchronous process with sync=true to inspect error attribution...")
        corrupt_proc = await client.post(f"/api/ingest/{corrupt_id}/process?sync=true")
        corrupt_data = corrupt_proc.json()
        print(f"  [OK] Received Failure Response:")
        print(f"    - Status: {corrupt_data['status']}")
        print(f"    - Current Stage: {corrupt_data['current_stage']}")
        print(f"    - Failure Reason: {corrupt_data['failure_reason']}")
        assert corrupt_data["status"] == "failed"
        assert corrupt_data["current_stage"] == "failed"
        assert "failed at extraction" in corrupt_data["failure_reason"]
        print("  [OK] Correct stage attribution verified ('failed at extraction')!")

    print("\n" + "=" * 80)
    print("ALL PHASE 12 VERIFICATION CHECKS COMPLETED AND VALIDATED!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
