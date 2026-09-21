"""
Scratch execution trace script for Phase 38 & 39: Correction Routing & Targeted Re-Retrieval.
"""

from app.core.logging import get_logger
from app.graph.routing import correction_router, get_failed_claims
from app.graph.state import RAGState
from app.graph.targeted_retrieve import targeted_retrieve_node
from app.models.schemas import ClaimWithSource, RerankResult, RetrievalResult, VerificationStatus

logger = get_logger("trace_targeted_retrieval")


def run_trace():
    print("\n=======================================================")
    print("PHASE 38 & 39 TRACE EXECUTION: ROUTING & TARGETED ")
    print("==================RETRIEVAL=====================================\n")

    # 1. Construct initial retrieved chunks
    initial_chunk1 = RetrievalResult(
        chunk_id="chunk-alpha-101",
        score=0.88,
        metadata={
            "source_text": "Acme Corp reported $15 million net profit in FY2023.",
            "document_id": "doc-acme-2023",
            "page_number": 2,
            "filename": "acme_annual_2023.pdf",
        },
    )

    # 2. Construct claims produced by verify_node (including one verified, one contradicted, one unverifiable)
    verified_claim = ClaimWithSource(
        claim_text="Acme Corp earned $15M net profit in 2023.",
        source_chunk_id="chunk-alpha-101",
        source_text=initial_chunk1.metadata["source_text"],
        is_valid_source=True,
        verification_status=VerificationStatus.ENTAILED,
        confidence=0.96,
    )

    contradicted_claim = ClaimWithSource(
        claim_text="Acme Corp expanded into 12 new European markets in FY2023.",
        source_chunk_id="chunk-alpha-101",
        source_text=initial_chunk1.metadata["source_text"],
        is_valid_source=True,
        verification_status=VerificationStatus.CONTRADICTED,
        confidence=0.91,
    )

    unverifiable_claim = ClaimWithSource(
        claim_text="Acme Corp acquired Beta AI Inc for $200M.",
        source_chunk_id="chunk-fake-999",
        source_text=None,
        is_valid_source=False,
        verification_status=VerificationStatus.UNVERIFIABLE,
        confidence=0.0,
    )

    state: RAGState = {
        "query": "What are Acme Corp's financial highlights and expansion details for 2023?",
        "retrieved_chunks": [initial_chunk1],
        "generated_answer": None,
        "claims": [verified_claim, contradicted_claim, unverifiable_claim],
        "retry_count": 0,
        "max_retries": 2,
        "final_status": "pending",
    }

    print("--- STEP 1: INSPECT INITIAL STATE ---")
    print(f"Original Query: '{state['query']}'")
    print(f"Initial Context Chunks: {len(state['retrieved_chunks'])} (ID: {state['retrieved_chunks'][0].chunk_id})")
    print(f"Total Claims: {len(state['claims'])}")
    for i, c in enumerate(state['claims'], 1):
        print(f"  Claim #{i} [{c.verification_status.value.upper()}]: '{c.claim_text}' (Source: {c.source_chunk_id})")

    # 2. Test get_failed_claims
    failed_claims = get_failed_claims(state["claims"])
    print(f"\nFailed Claims Extracted: {len(failed_claims)} claim(s)")
    for fc in failed_claims:
        print(f"  -> Failed Claim [ID: {fc.claim_id[:8]}]: '{fc.claim_text}'")

    # 3. Test correction_router
    route = correction_router(state)
    print(f"\n--- STEP 2: CORRECTION ROUTER DECISION ---")
    print(f"Router Decision: -> '{route}' (Expected: 'targeted_retrieve')")

    # 4. Run targeted_retrieve_node
    print(f"\n--- STEP 3: EXECUTE TARGETED RE-RETRIEVAL NODE ---")
    print("Running targeted search using claim texts as queries...\n")

    updated_state_dict = targeted_retrieve_node(state)

    print("\n--- TARGETED RETRIEVAL RESULTS ---")
    print(f"New Retry Count: {updated_state_dict['retry_count']}")
    merged_chunks = updated_state_dict['retrieved_chunks']
    print(f"Total Merged Context Chunks: {len(merged_chunks)}")
    for i, chunk in enumerate(merged_chunks, 1):
        text_snippet = chunk.metadata.get("source_text", "")[:60]
        print(f"  Chunk #{i} [ID: {chunk.chunk_id}]: '{text_snippet}...'")

    # 5. Run second iteration to demonstrate "nothing new found" logging
    print("\n--- STEP 4: SECOND RETRIEVAL ITERATION ('NOTHING NEW FOUND' DEMO) ---")
    state_iter2: RAGState = {
        **state,
        "retrieved_chunks": merged_chunks,
        "retry_count": updated_state_dict['retry_count'],
    }
    print("Executing targeted_retrieve_node again with saturated context...")
    updated_state_iter2 = targeted_retrieve_node(state_iter2)
    print(f"Iteration 2 Merged Chunks Count: {len(updated_state_iter2['retrieved_chunks'])}")
    print(f"Iteration 2 Retry Count: {updated_state_iter2['retry_count']}")

    print("\n=======================================================")
    print("TRACE EXECUTION COMPLETED SUCCESSFULLY")
    print("=======================================================\n")


if __name__ == "__main__":
    run_trace()
