"""
End-to-End Integration Test for Phase 41: Full Self-Correction Loop Execution.
"""

from unittest.mock import patch

from app.graph.graph import app_graph
from app.graph.state import RAGState
from app.models.schemas import Claim, ClaimWithSource, GeneratedAnswer, LLMResponse, RerankResult, RetrievalResult, VerificationStatus


def test_full_correction_loop_end_to_end_success():
    """Integration test proving the complete LangGraph correction loop end-to-end.

    Flow:
    1. retrieve_node: returns chunk-1 and chunk-2
    2. generate_node: generates Claim 1 (valid) and Claim 2 (contradicted)
    3. verify_node: verifies Claim 1 as ENTAILED, Claim 2 as CONTRADICTED
    4. correction_router: detects failed Claim 2 -> routes to targeted_retrieve
    5. targeted_retrieve_node: searches with Claim 2 text -> returns enriched chunk-3
    6. regenerate_node: partially regenerates Claim 2 into corrected Claim 2_new
    7. verify_node: re-verifies Claim 1 and Claim 2_new -> BOTH now ENTAILED
    8. correction_router: all claims verified -> routes to finalize
    9. finalize_node: outputs final_status = 'fully_verified' and synthesized text.
    """
    chunk1 = RetrievalResult(
        chunk_id="chunk-1",
        score=0.9,
        metadata={"source_text": "Acme Corp reported $15M revenue in 2023.", "document_id": "doc-1", "page_number": 1},
    )
    chunk2 = RetrievalResult(
        chunk_id="chunk-2",
        score=0.85,
        metadata={"source_text": "Acme Corp operates 5 regional offices.", "document_id": "doc-1", "page_number": 2},
    )

    initial_answer = GeneratedAnswer(
        claims=[
            Claim(claim_text="Acme Corp reported $15M revenue in 2023.", source_chunk_id="chunk-1"),
            Claim(claim_text="Acme Corp operates 50 regional offices worldwide.", source_chunk_id="chunk-2"),
        ],
        raw_response="Initial raw answer",
    )

    # First verify pass: claim 1 entailed, claim 2 contradicted
    claim1_verified = ClaimWithSource(
        claim_text="Acme Corp reported $15M revenue in 2023.",
        source_chunk_id="chunk-1",
        source_text=chunk1.metadata["source_text"],
        is_valid_source=True,
        verification_status=VerificationStatus.ENTAILED,
        confidence=0.96,
    )
    claim2_contradicted = ClaimWithSource(
        claim_text="Acme Corp operates 50 regional offices worldwide.",
        source_chunk_id="chunk-2",
        source_text=chunk2.metadata["source_text"],
        is_valid_source=True,
        verification_status=VerificationStatus.CONTRADICTED,
        confidence=0.92,
    )

    # Corrected claim produced by partial regeneration
    claim2_corrected_mapped = ClaimWithSource(
        claim_text="Acme Corp operates 5 regional offices.",
        source_chunk_id="chunk-2",
        source_text=chunk2.metadata["source_text"],
        is_valid_source=True,
        verification_status=VerificationStatus.ENTAILED,
        confidence=0.94,
    )

    initial_state: RAGState = {
        "query": "What is Acme Corp's revenue and office footprint?",
        "retry_count": 0,
        "max_retries": 2,
    }

    with patch("app.graph.graph.hybrid_search") as mock_hybrid, \
         patch("app.graph.graph.select_relevant_chunks") as mock_rerank, \
         patch("app.graph.graph.generate_answer_with_citations") as mock_gen, \
         patch("app.graph.targeted_retrieve.hybrid_search") as mock_targeted_hybrid, \
         patch("app.graph.targeted_retrieve.select_relevant_chunks") as mock_targeted_rerank, \
         patch("app.generation.partial_regenerate.generate_answer") as mock_partial_llm:

        # Step 1: Initial retrieval
        mock_hybrid.return_value = [chunk1, chunk2]
        mock_rerank.return_value = RerankResult(chunks=[chunk1, chunk2], relevant_count=2)
        mock_gen.return_value = initial_answer

        # Step 5: Targeted search
        mock_targeted_hybrid.return_value = [chunk2]
        mock_targeted_rerank.return_value = RerankResult(chunks=[chunk2], relevant_count=1)

        # Step 6: Partial regeneration LLM response
        corrected_json = '{"claims": [{"claim_text": "Acme Corp operates 5 regional offices.", "source_chunk_id": "chunk-2"}], "insufficient_information": false}'
        mock_partial_llm.return_value = LLMResponse(content=corrected_json, provider="groq", model_name="llama3")

        # Mock verify_claims behavior across iterations
        verify_call_count = 0

        def mock_verify_side_effect(claims_to_verify):
            nonlocal verify_call_count
            verify_call_count += 1
            if verify_call_count == 1:
                return [claim1_verified, claim2_contradicted]
            else:
                return [claim1_verified, claim2_corrected_mapped]

        with patch("app.graph.graph.verify_claims", side_effect=mock_verify_side_effect):
            final_state = app_graph.invoke(initial_state)

        # Assertions on final compiled graph execution
        assert final_state["final_status"] == "fully_verified"
        assert final_state["retry_count"] == 1
        assert len(final_state["claims"]) == 2
        assert final_state["final_answer_text"] == "Acme Corp reported $15M revenue in 2023. Acme Corp operates 5 regional offices."
        assert verify_call_count == 2
