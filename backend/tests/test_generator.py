"""
Unit & integration tests for Phase 27: Structured Output Schema & Generator.
"""

from unittest.mock import patch
import pytest

from app.core.config import _is_placeholder, settings
from app.generation import (
    Claim,
    GeneratedAnswer,
    LLMResponse,
    clean_json_output,
    generate_answer_with_citations,
)
from app.models.schemas import RetrievalResult


def test_clean_json_output():
    """Verify clean_json_output defensively strips markdown code fences, preambles, and whitespace."""
    # 1. Clean JSON string
    assert clean_json_output('{"claims": []}') == '{"claims": []}'

    # 2. Wrapped in ```json ... ``` code fences
    fenced = """```json
{
  "claims": [
    {"claim_text": "Sample claim", "source_chunk_id": "c1"}
  ],
  "insufficient_information": false
}
```"""
    cleaned = clean_json_output(fenced)
    assert cleaned.startswith("{") and cleaned.endswith("}")
    assert '"claim_text": "Sample claim"' in cleaned

    # 3. Preamble text before JSON
    preamble = """Here is the structured JSON output:
{
  "claims": [],
  "insufficient_information": true
}
Hope this helps!"""
    assert clean_json_output(preamble) == '{\n  "claims": [],\n  "insufficient_information": true\n}'


def test_generate_answer_hallucinated_chunk_id_quarantined():
    """Verify that when an LLM cites a non-existent chunk_id, it is quarantined into unverified_claims."""
    valid_chunks = [
        RetrievalResult(
            chunk_id="valid_chunk_101",
            score=0.9,
            metadata={"source_text": "Reciprocal Rank Fusion k=60 constant."},
        )
    ]

    llm_payload = """{
      "claims": [
        {"claim_text": "RRF uses a k=60 constant.", "source_chunk_id": "valid_chunk_101"},
        {"claim_text": "Pinecone vector DB uses cosine similarity.", "source_chunk_id": "hallucinated_fake_chunk_999"}
      ],
      "insufficient_information": false
    }"""

    mock_llm_res = LLMResponse(
        content=llm_payload,
        provider="groq",
        model_name="openai/gpt-oss-120b",
        latency_ms=250.0,
    )

    with patch("app.generation.generator.generate_answer", return_value=mock_llm_res):
        res = generate_answer_with_citations("Explain RRF and Pinecone", valid_chunks)

        # Assert valid claim is accepted
        assert len(res.claims) == 1
        assert res.claims[0].source_chunk_id == "valid_chunk_101"

        # Assert hallucinated claim is quarantined in unverified_claims
        assert len(res.unverified_claims) == 1
        assert res.unverified_claims[0].source_chunk_id == "hallucinated_fake_chunk_999"


def test_generate_answer_unanswerable_insufficient_information():
    """Verify unanswerable queries or empty chunk lists return insufficient_information=True."""
    # 1. Empty chunk list
    res1 = generate_answer_with_citations("What is the capital of France?", [])
    assert res1.insufficient_information is True
    assert res1.claims == []

    # 2. Out-of-scope query where LLM reports insufficient_information=true
    chunks = [
        RetrievalResult(
            chunk_id="chunk_bm25",
            score=0.9,
            metadata={"source_text": "BM25 keyword search tokenizes documents into lowercase terms."},
        )
    ]

    llm_payload = '{"claims": [], "insufficient_information": true}'
    mock_llm_res = LLMResponse(content=llm_payload, provider="groq", model_name="openai/gpt-oss-120b")

    with patch("app.generation.generator.generate_answer", return_value=mock_llm_res):
        res2 = generate_answer_with_citations("What is the recipe for baking sourdough bread?", chunks)
        assert res2.insufficient_information is True
        assert res2.claims == []


def test_generate_answer_live_e2e_integration():
    """Live integration test: Run an answerable query against real chunks via active LLM provider."""
    if (not settings.GROQ_API_KEY or _is_placeholder(settings.GROQ_API_KEY)) and (not settings.GEMINI_API_KEY or _is_placeholder(settings.GEMINI_API_KEY)):
        pytest.skip("Neither GROQ_API_KEY nor GEMINI_API_KEY configured; skipping live E2E generation test.")

    chunks = [
        RetrievalResult(
            chunk_id="doc_test::c1",
            score=0.95,
            metadata={"source_text": "Sentence-transformers all-MiniLM-L6-v2 model generates 384 dimensional dense vector embeddings."},
        )
    ]

    query = "What dimension vectors does the sentence-transformers model generate?"
    res = generate_answer_with_citations(query, chunks, temperature=0.0)

    assert isinstance(res, GeneratedAnswer)
    assert res.insufficient_information is False
    assert len(res.claims) >= 1
    # Check that at least one claim cites doc_test::c1
    cited_ids = {c.source_chunk_id for c in res.claims}
    assert "doc_test::c1" in cited_ids
