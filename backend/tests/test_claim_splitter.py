"""
Unit tests for sentence splitting and claim atomicity verification (Phase 30).
"""

import pytest

from app.models.schemas import Claim
from app.verification.claim_splitter import (
    ensure_atomic_claims,
    is_atomic_claim,
    split_into_sentences,
)


def test_split_into_sentences_basic():
    text = (
        "Retrieval-Augmented Generation improves LLM accuracy. "
        "It retrieves relevant document chunks from a vector database. "
        "Then, the model generates an answer anchored on the retrieved context."
    )
    sentences = split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "Retrieval-Augmented Generation improves LLM accuracy."
    assert sentences[1] == "It retrieves relevant document chunks from a vector database."
    assert sentences[2] == "Then, the model generates an answer anchored on the retrieved context."


def test_is_atomic_claim_single_vs_multiple():
    single_claim = "Dense vector embeddings capture semantic meaning."
    assert is_atomic_claim(single_claim) is True

    compound_claim = "Dense vector embeddings capture semantic meaning. BM25 relies on exact keyword frequencies."
    assert is_atomic_claim(compound_claim) is False


def test_ensure_atomic_claims_decomposes_compound_claim():
    claims = [
        Claim(
            claim_text="RRF combines dense and sparse search scores.",
            source_chunk_id="chunk_1",
        ),
        Claim(
            claim_text="Pinecone stores dense vectors. SentenceTransformers generates 384d vectors.",
            source_chunk_id="chunk_2",
        ),
    ]

    atomic_claims = ensure_atomic_claims(claims)

    assert len(atomic_claims) == 3
    assert atomic_claims[0].claim_text == "RRF combines dense and sparse search scores."
    assert atomic_claims[0].source_chunk_id == "chunk_1"

    assert atomic_claims[1].claim_text == "Pinecone stores dense vectors."
    assert atomic_claims[1].source_chunk_id == "chunk_2"

    assert atomic_claims[2].claim_text == "SentenceTransformers generates 384d vectors."
    assert atomic_claims[2].source_chunk_id == "chunk_2"


def test_abbreviation_and_formatting():
    text = "Dr. Smith visited the U.S. in 2026. He reported significant efficiency gains."
    sentences = split_into_sentences(text)
    assert len(sentences) == 2
    assert "Dr. Smith" in sentences[0]
    assert "U.S." in sentences[0]
    assert sentences[1] == "He reported significant efficiency gains."
