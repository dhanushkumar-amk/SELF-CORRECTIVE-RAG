"""
Unit tests for claim-to-chunk mapping and context resolution (Phase 31).
"""

import pytest

from app.models.schemas import Claim, ClaimWithSource, GeneratedAnswer, RetrievalResult
from app.verification.claim_mapper import map_claims_to_chunks


@pytest.fixture
def sample_retrieved_chunks() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="chunk_101",
            score=0.92,
            metadata={
                "source_text": "SentenceTransformers models produce 384-dimensional dense vectors.",
                "page_number": 3,
                "page_number_end": 3,
                "document_id": "doc_abc123",
                "filename": "embedding_guide.pdf",
            },
        ),
        RetrievalResult(
            chunk_id="chunk_102",
            score=0.88,
            metadata={
                "source_text": "BM25 sparse search ranks documents using term frequency and inverse document frequency.",
                "page_number": 5,
                "page_number_end": 6,
                "document_id": "doc_abc123",
                "filename": "embedding_guide.pdf",
            },
        ),
    ]


def test_map_claims_to_chunks_success(sample_retrieved_chunks):
    answer = GeneratedAnswer(
        claims=[
            Claim(
                claim_text="SentenceTransformers models produce 384-dimensional dense vectors.",
                source_chunk_id="chunk_101",
            ),
            Claim(
                claim_text="BM25 search uses term frequency.",
                source_chunk_id="chunk_102",
            ),
        ],
        insufficient_information=False,
    )

    mapped_claims = map_claims_to_chunks(answer, sample_retrieved_chunks)

    assert len(mapped_claims) == 2
    assert isinstance(mapped_claims[0], ClaimWithSource)

    # Claim 1 assertion
    assert mapped_claims[0].claim_id is not None
    assert mapped_claims[0].claim_text == "SentenceTransformers models produce 384-dimensional dense vectors."
    assert mapped_claims[0].source_chunk_id == "chunk_101"
    assert mapped_claims[0].source_text == "SentenceTransformers models produce 384-dimensional dense vectors."
    assert mapped_claims[0].page_number == 3
    assert mapped_claims[0].is_valid_source is True
    assert mapped_claims[0].filename == "embedding_guide.pdf"

    # Claim 2 assertion
    assert mapped_claims[1].claim_text == "BM25 search uses term frequency."
    assert mapped_claims[1].source_chunk_id == "chunk_102"
    assert mapped_claims[1].page_number == 5
    assert mapped_claims[1].page_number_end == 6
    assert mapped_claims[1].is_valid_source is True


def test_map_claims_to_chunks_invalid_chunk_id(sample_retrieved_chunks):
    answer = GeneratedAnswer(
        claims=[
            Claim(
                claim_text="Valid claim from chunk 101.",
                source_chunk_id="chunk_101",
            ),
        ],
        unverified_claims=[
            Claim(
                claim_text="Hallucinated claim with fake chunk ID.",
                source_chunk_id="fake_chunk_999",
            )
        ],
        insufficient_information=False,
    )

    mapped_claims = map_claims_to_chunks(answer, sample_retrieved_chunks, include_unverified=True)

    assert len(mapped_claims) == 2

    # Valid claim
    assert mapped_claims[0].is_valid_source is True
    assert mapped_claims[0].source_text is not None

    # Invalid claim sentinel check
    assert mapped_claims[1].claim_text == "Hallucinated claim with fake chunk ID."
    assert mapped_claims[1].source_chunk_id == "fake_chunk_999"
    assert mapped_claims[1].is_valid_source is False
    assert mapped_claims[1].source_text is None
    assert mapped_claims[1].page_number is None


def test_map_claims_to_chunks_compound_claim_splitting(sample_retrieved_chunks):
    answer = GeneratedAnswer(
        claims=[
            Claim(
                claim_text="SentenceTransformers models produce 384-dimensional vectors. They are used for dense retrieval.",
                source_chunk_id="chunk_101",
            )
        ],
        insufficient_information=False,
    )

    mapped_claims = map_claims_to_chunks(answer, sample_retrieved_chunks)

    assert len(mapped_claims) == 2
    assert mapped_claims[0].claim_text == "SentenceTransformers models produce 384-dimensional vectors."
    assert mapped_claims[0].source_chunk_id == "chunk_101"
    assert mapped_claims[0].source_text == "SentenceTransformers models produce 384-dimensional dense vectors."

    assert mapped_claims[1].claim_text == "They are used for dense retrieval."
    assert mapped_claims[1].source_chunk_id == "chunk_101"
    assert mapped_claims[1].source_text == "SentenceTransformers models produce 384-dimensional dense vectors."
