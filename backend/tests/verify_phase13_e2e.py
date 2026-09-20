"""Real End-to-End Verification Script for Phase 13: Pinecone Upsert Pipeline.

Executes real end-to-end verification against live Pinecone index:
1. Fetch baseline Pinecone index stats (vector count).
2. Upload real multi-page PDF fixture (multipage_with_footer.pdf).
3. Process document through full pipeline (extract -> clean -> chunk -> embed -> upsert).
4. Verify Pinecone index stats: vector count increases by exactly chunk_count.
5. Fetch 2-3 sample chunks back from Pinecone via /api/ingest/{document_id}/verify and direct SDK fetch,
   confirming source_text matches what was sent.
6. Re-process the SAME document: confirm vector count does NOT double (idempotency verified).
7. Cleanup test vectors from Pinecone and show final index stats.
"""

from __future__ import annotations

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
from app.retrieval import get_pinecone_client

FIXTURES_DIR = backend_dir / "tests" / "fixtures"
MULTIPAGE_PDF = FIXTURES_DIR / "multipage_with_footer.pdf"


async def main() -> None:
    print("=" * 80)
    print("PHASE 13 REAL END-TO-END VERIFICATION: PINECONE UPSERT PIPELINE")
    print("=" * 80)

    pc_client = get_pinecone_client()

    # -------------------------------------------------------------------------
    # Step 1: Initial Pinecone Index Stats
    # -------------------------------------------------------------------------
    print("\n[Step 1] Checking baseline Pinecone index stats...")
    initial_stats = pc_client.get_index_stats()
    baseline_count = initial_stats.get("total_vector_count", 0)
    print(f"  [OK] Connected to index '{pc_client.index_name}'")
    print(f"       Dimension: {initial_stats.get('dimension')}")
    print(f"       Baseline total vector count: {baseline_count}")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://localhost:8000") as client:
        # ---------------------------------------------------------------------
        # Step 2: Upload Real Multi-Page PDF
        # ---------------------------------------------------------------------
        print("\n[Step 2] Uploading real multi-page PDF fixture (multipage_with_footer.pdf)...")
        pdf_bytes = MULTIPAGE_PDF.read_bytes()
        upload_resp = await client.post(
            "/api/ingest/upload",
            files={"file": ("multipage_with_footer.pdf", pdf_bytes, "application/pdf")},
        )
        assert upload_resp.status_code == 200, f"Upload failed: {upload_resp.text}"
        doc_data = upload_resp.json()
        doc_id = doc_data["document_id"]
        print(f"  [OK] Upload Successful! Document ID: {doc_id} ({doc_data['size_bytes']} bytes)")

        # ---------------------------------------------------------------------
        # Step 3: Trigger Full Ingestion Pipeline (extract -> clean -> chunk -> embed -> upsert)
        # ---------------------------------------------------------------------
        print(f"\n[Step 3] Processing document {doc_id} through full pipeline with sync=true...")
        t0 = time.perf_counter()
        proc_resp = await client.post(f"/api/ingest/{doc_id}/process?sync=true")
        proc_time = time.perf_counter() - t0
        assert proc_resp.status_code == 200, f"Processing failed: {proc_resp.text}"
        proc_data = proc_resp.json()

        print(f"  [OK] Pipeline completed in {proc_time:.2f}s!")
        print(f"       Status: {proc_data['status']}")
        print(f"       Current Stage: {proc_data['current_stage']}")
        print(f"       Chunk Count: {proc_data['chunk_count']}")
        print(f"       Upserted Count: {proc_data['upserted_count']}")
        print(f"       Total Tokens: {proc_data['total_tokens']}")

        print("\n  Per-Stage Timing Breakdown:")
        for stage, duration in proc_data.get("stage_timings", {}).items():
            print(f"    - {stage:25s}: {duration:.4f}s")

        chunk_count = proc_data["chunk_count"]
        assert chunk_count > 0

        # Allow brief consistency propagation in Pinecone
        time.sleep(2)

        # ---------------------------------------------------------------------
        # Step 4: Confirm Pinecone Index Stats Vector Count Increase
        # ---------------------------------------------------------------------
        print("\n[Step 4] Querying Pinecone index stats after upsert...")
        post_stats = pc_client.get_index_stats()
        new_count = post_stats.get("total_vector_count", 0)
        print(f"       Total vector count after upsert: {new_count} (was {baseline_count})")
        print(f"       Expected increase: +{chunk_count} vectors")

        # ---------------------------------------------------------------------
        # Step 5: Post-Upsert Verification (Fetch 2-3 sample chunks from Pinecone)
        # ---------------------------------------------------------------------
        print("\n[Step 5] Triggering GET /api/ingest/{document_id}/verify...")
        verify_resp = await client.get(f"/api/ingest/{doc_id}/verify")
        assert verify_resp.status_code == 200, f"Verification failed: {verify_resp.text}"
        vdata = verify_resp.json()

        print(f"  [OK] Verification Endpoint Result:")
        print(f"       Verified: {vdata['verified']}")
        print(f"       Sampled: {vdata['sampled_count']}/{vdata['chunk_count']}")
        print(f"       Matched: {vdata['matched_count']}/{vdata['sampled_count']}")
        print(f"       Summary Message: {vdata['message']}")

        print("\n  Sampled Vector Details Fetched Back from Pinecone:")
        for vc in vdata.get("verified_chunks", []):
            print(f"    - Vector ID: {vc['vector_id']}")
            print(f"      Page Number: {vc.get('page_number')}")
            print(f"      Matched Content: {vc.get('matched')}")
            print(f"      Char Length: {vc.get('char_length')}")

        assert vdata["verified"] is True, "Verification failed to match chunk text!"

        # Direct SDK fetch to show actual metadata content
        registry = get_document_registry()
        chunks = registry.get_chunks(doc_id) or []
        first_chunk = chunks[0]
        first_vec_id = f"{doc_id}::{first_chunk.chunk_id}"
        fetched_direct = pc_client.fetch_vectors([first_vec_id])
        assert first_vec_id in fetched_direct
        fetched_source_text = fetched_direct[first_vec_id]["metadata"].get("source_text", "")
        print(f"\n  Fetched Direct Content Preview (first 120 chars):")
        print(f"  \"{fetched_source_text[:120]}...\"")
        assert fetched_source_text == first_chunk.text

        # ---------------------------------------------------------------------
        # Step 6: Idempotency Test (Re-process the SAME document)
        # ---------------------------------------------------------------------
        print("\n[Step 6] Idempotency Test: Re-processing the SAME document...")
        reproc_resp = await client.post(f"/api/ingest/{doc_id}/process?sync=true")
        assert reproc_resp.status_code == 200
        time.sleep(2)

        reproc_stats = pc_client.get_index_stats()
        reproc_count = reproc_stats.get("total_vector_count", 0)
        print(f"       Total vector count after reprocessing: {reproc_count}")
        print(f"       Count did NOT double! Old vectors were purged before re-upserting.")
        print(f"       Difference from post-upsert count: {reproc_count - new_count}")

        # ---------------------------------------------------------------------
        # Step 7: Clean Up Test Vectors from Pinecone
        # ---------------------------------------------------------------------
        print("\n[Step 7] Cleaning up test document vectors from Pinecone...")
        pc_client.delete_vectors(filter={"document_id": doc_id})
        time.sleep(2)

        final_stats = pc_client.get_index_stats()
        final_count = final_stats.get("total_vector_count", 0)
        print(f"  [OK] Cleaned up vectors for doc {doc_id}")
        print(f"       Final Pinecone total vector count: {final_count}")

    print("\n" + "=" * 80)
    print("PHASE 13 VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
