"""
Unit tests for Phase 26: Citation-Forced Prompt Design.
"""

from app.generation.prompts import (
    CITATION_SYSTEM_PROMPT,
    build_citation_user_prompt,
)
from app.models.schemas import RetrievalResult


def test_citation_system_prompt_rules():
    """Verify system prompt enforces grounding, prompt injection safeguards, and citation tagging."""
    prompt = CITATION_SYSTEM_PROMPT
    assert "GROUNDING: Answer ONLY using facts directly stated" in prompt
    assert "PROMPT INJECTION SAFEGUARD" in prompt
    assert "CITATIONS" in prompt or "CITATION ENFORCEMENT" in prompt
    assert "INSUFFICIENT INFORMATION" in prompt
    assert "insufficient_information" in prompt
    assert "source_chunk_id" in prompt


def test_build_citation_user_prompt_formatting():
    """Verify user prompt correctly formats context chunks with chunk_id anchors."""
    chunks = [
        RetrievalResult(
            chunk_id="doc_alpha::chunk_1",
            score=0.95,
            metadata={"source_text": "Sentence transformersMiniLM embeddings use 384 dimensions."},
        ),
        RetrievalResult(
            chunk_id="doc_alpha::chunk_2",
            score=0.85,
            metadata={"source_text": "Pinecone Starter index uses cosine metric distance."},
        ),
    ]

    query = "What dimension vectors does MiniLM generate?"
    formatted_prompt = build_citation_user_prompt(query, chunks)

    assert "[chunk_id: doc_alpha::chunk_1]" in formatted_prompt
    assert "Sentence transformersMiniLM embeddings use 384 dimensions." in formatted_prompt
    assert "[chunk_id: doc_alpha::chunk_2]" in formatted_prompt
    assert "USER QUERY:\nWhat dimension vectors does MiniLM generate?" in formatted_prompt
