"""
Scratch execution trace script for Phase 40 & 41: End-to-End Correction Loop Execution.
"""

from unittest.mock import patch

from app.graph.graph import app_graph
from app.graph.state import RAGState
from app.models.schemas import (
    Claim,
    ClaimWithSource,
    GeneratedAnswer,
    LLMResponse,
    RerankResult,
    RetrievalResult,
    VerificationStatus,
)


def run_e2e_trace():
    print("\n=======================================================================")
    print("PHASES 40 & 41 END-TO-END CORRECTION LOOP EXECUTION TRACE")
    print("=======================================================================\n")

    # 1. Setup Initial Chunks & Answer
    chunk_1 = RetrievalResult(
        chunk_id="chunk-acme-101",
        score=0.92,
        metadata={
            "source_text": "Acme Corp reported $15 million in net profit for fiscal year 2023.",
            "document_id": "doc-acme-report",
            "page_number": 2,
            "filename": "acme_annual_2023.pdf",
        },
    )
    chunk_2 = RetrievalResult(
        chunk_id="chunk-acme-102",
        score=0.88,
        metadata={
            "source_text": "Acme Corp currently operates 5 regional research centers across North America.",
            "document_id": "doc-acme-report",
            "page_number": 4,
            "filename": "acme_annual_2023.pdf",
        },
    )

    initial_answer = GeneratedAnswer(
        claims=[
            Claim(
                claim_text="Acme Corp earned $15 million in net profit during fiscal year 2023.",
                source_chunk_id="chunk-acme-101",
            ),
            Claim(
                claim_text="Acme Corp operates 50 regional research centers worldwide.",  # Flagged hallucination!
                source_chunk_id="chunk-acme-102",
            ),
        ],
        raw_response="Raw initial generation",
    )

    claim1_verified = ClaimWithSource(
        claim_text="Acme Corp earned $15 million in net profit during fiscal year 2023.",
        source_chunk_id="chunk-acme-101",
        source_text=chunk_1.metadata["source_text"],
        is_valid_source=True,
        verification_status=VerificationStatus.ENTAILED,
        confidence=0.97,
    )
    claim2_contradicted = ClaimWithSource(
        claim_text="Acme Corp operates 50 regional research centers worldwide.",
        source_chunk_id="chunk-acme-102",
        source_text=chunk_2.metadata["source_text"],
        is_valid_source=True,
        verification_status=VerificationStatus.CONTRADICTED,
        confidence=0.94,
    )

    claim2_corrected_mapped = ClaimWithSource(
        claim_text="Acme Corp operates 5 regional research centers across North America.",
        source_chunk_id="chunk-acme-102",
        source_text=chunk_2.metadata["source_text"],
        is_valid_source=True,
        verification_status=VerificationStatus.ENTAILED,
        confidence=0.95,
    )

    initial_state: RAGState = {
        "query": "What were Acme Corp's financial results and research facility footprint in 2023?",
        "retry_count": 0,
        "max_retries": 2,
    }

    print("--- STEP 1: INITIALIZE GRAPH EXECUTION ---")
    print(f"User Query: '{initial_state['query']}'")

    with patch("app.graph.graph.hybrid_search") as mock_hybrid, \
         patch("app.graph.graph.select_relevant_chunks") as mock_rerank, \
         patch("app.graph.graph.generate_answer_with_citations") as mock_gen, \
         patch("app.graph.targeted_retrieve.hybrid_search") as mock_targeted_hybrid, \
         patch("app.graph.targeted_retrieve.select_relevant_chunks") as mock_targeted_rerank, \
         patch("app.generation.partial_regenerate.generate_answer") as mock_partial_llm:

        mock_hybrid.return_value = [chunk_1, chunk_2]
        mock_rerank.return_value = RerankResult(chunks=[chunk_1, chunk_2], relevant_count=2)
        mock_gen.return_value = initial_answer

        mock_targeted_hybrid.return_value = [chunk_2]
        mock_targeted_rerank.return_value = RerankResult(chunks=[chunk_2], relevant_count=1)

        corrected_json = '{"claims": [{"claim_text": "Acme Corp operates 5 regional research centers across North America.", "source_chunk_id": "chunk-acme-102"}], "insufficient_information": false}'
        mock_partial_llm.return_value = LLMResponse(content=corrected_json, provider="groq", model_name="llama3")

        verify_count = 0

        def mock_verify_side_effect(claims_to_verify):
            nonlocal verify_count
            verify_count += 1
            print(f"\n--- VERIFY PASS #{verify_count} ---")
            if verify_count == 1:
                print("Evaluating initial claims:")
                print(f"  Claim 1: '{claim1_verified.claim_text}' -> Status: {claim1_verified.verification_status.value.upper()}")
                print(f"  Claim 2: '{claim2_contradicted.claim_text}' -> Status: {claim2_contradicted.verification_status.value.upper()} (FLAGGED!)")
                return [claim1_verified, claim2_contradicted]
            else:
                print("Re-evaluating updated claims post partial regeneration:")
                print(f"  Claim 1 (Preserved): '{claim1_verified.claim_text}' -> Status: {claim1_verified.verification_status.value.upper()}")
                print(f"  Claim 2 (Corrected): '{claim2_corrected_mapped.claim_text}' -> Status: {claim2_corrected_mapped.verification_status.value.upper()} (VERIFIED!)")
                return [claim1_verified, claim2_corrected_mapped]

        with patch("app.graph.graph.verify_claims", side_effect=mock_verify_side_effect):
            final_state = app_graph.invoke(initial_state)

    print("\n--- STEP 2: FINAL GRAPH EXECUTION RESULTS ---")
    print(f"Final Status: '{final_state['final_status']}'")
    print(f"Total Retries Executed: {final_state['retry_count']}")
    print(f"\nFinal Synthesized Answer Text:\n\"{final_state['final_answer_text']}\"")

    print("\nFinal Claim Breakdown:")
    for i, c in enumerate(final_state['claims'], 1):
        print(f"  Claim #{i} [{c.verification_status.value.upper()} | Conf: {c.confidence:.2f}]: '{c.claim_text}' (Source: {c.source_chunk_id})")

    print("\n=======================================================================")
    print("END-TO-END CORRECTION LOOP TRACE COMPLETED SUCCESSFULLY")
    print("=======================================================================\n")


if __name__ == "__main__":
    run_e2e_trace()
