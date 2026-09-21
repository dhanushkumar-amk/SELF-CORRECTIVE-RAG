"""
Scratch script to capture and display raw SSE event-stream output for a query triggering self-correction.
"""

import asyncio
from unittest.mock import patch

from app.api.query_stream import stream_query_execution
from app.models.schemas import (
    Claim,
    ClaimWithSource,
    GeneratedAnswer,
    LLMResponse,
    RerankResult,
    RetrievalResult,
    VerificationStatus,
)


async def capture_sse_stream():
    print("\n=======================================================================")
    print("PHASE 43 REAL SSE EVENT-STREAM TRACE CAPTURE (raw text/event-stream)")
    print("=======================================================================\n")

    chunk1 = RetrievalResult(
        chunk_id="chunk-acme-101",
        score=0.92,
        metadata={
            "source_text": "Acme Corp reported $15 million in net profit for fiscal year 2023.",
            "document_id": "doc-acme-2023",
            "page_number": 2,
        },
    )
    chunk2 = RetrievalResult(
        chunk_id="chunk-acme-102",
        score=0.88,
        metadata={
            "source_text": "Acme Corp currently operates 5 regional research centers across North America.",
            "document_id": "doc-acme-2023",
            "page_number": 4,
        },
    )

    initial_answer = GeneratedAnswer(
        claims=[
            Claim(
                claim_text="Acme Corp earned $15 million in net profit during fiscal year 2023.",
                source_chunk_id="chunk-acme-101",
            ),
            Claim(
                claim_text="Acme Corp operates 50 regional research centers worldwide.",  # Flagged claim
                source_chunk_id="chunk-acme-102",
            ),
        ],
        raw_response="Initial answer",
    )

    claim1_verified = ClaimWithSource(
        claim_text="Acme Corp earned $15 million in net profit during fiscal year 2023.",
        source_chunk_id="chunk-acme-101",
        source_text=chunk1.metadata["source_text"],
        is_valid_source=True,
        verification_status=VerificationStatus.ENTAILED,
        confidence=0.97,
    )
    claim2_contradicted = ClaimWithSource(
        claim_text="Acme Corp operates 50 regional research centers worldwide.",
        source_chunk_id="chunk-acme-102",
        source_text=chunk2.metadata["source_text"],
        is_valid_source=True,
        verification_status=VerificationStatus.CONTRADICTED,
        confidence=0.94,
    )
    claim2_corrected_mapped = ClaimWithSource(
        claim_text="Acme Corp operates 5 regional research centers across North America.",
        source_chunk_id="chunk-acme-102",
        source_text=chunk2.metadata["source_text"],
        is_valid_source=True,
        verification_status=VerificationStatus.ENTAILED,
        confidence=0.95,
    )

    verify_count = 0

    def mock_verify_side_effect(claims_to_verify):
        nonlocal verify_count
        verify_count += 1
        if verify_count == 1:
            return [claim1_verified, claim2_contradicted]
        else:
            return [claim1_verified, claim2_corrected_mapped]

    corrected_json = '{"claims": [{"claim_text": "Acme Corp operates 5 regional research centers across North America.", "source_chunk_id": "chunk-acme-102"}], "insufficient_information": false}'

    with patch("app.api.query_stream.hybrid_search", return_value=[chunk1, chunk2]), \
         patch("app.api.query_stream.select_relevant_chunks", return_value=RerankResult(chunks=[chunk1, chunk2], relevant_count=2)), \
         patch("app.api.query_stream.generate_answer_with_citations", return_value=initial_answer), \
         patch("app.api.query_stream.verify_claims", side_effect=mock_verify_side_effect), \
         patch("app.graph.targeted_retrieve.hybrid_search", return_value=[chunk2]), \
         patch("app.graph.targeted_retrieve.select_relevant_chunks", return_value=RerankResult(chunks=[chunk2], relevant_count=1)), \
         patch("app.generation.partial_regenerate.generate_answer", return_value=LLMResponse(content=corrected_json, provider="groq", model_name="llama3")):

        stream_gen = stream_query_execution(
            query="What were Acme Corp's financial results and research facility footprint in 2023?",
            document_id="doc-acme-2023",
        )

        async for sse_chunk in stream_gen:
            print(sse_chunk, end="")

    print("=======================================================================")
    print("SSE EVENT-STREAM TRACE CAPTURE COMPLETED")
    print("=======================================================================\n")


if __name__ == "__main__":
    asyncio.run(capture_sse_stream())
