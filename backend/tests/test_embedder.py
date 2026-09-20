"""Tests for local embedding model integration, determinism, batching, and sequence length behavior."""

import math
import pytest

from app.ingestion.embedder import (
    EMBEDDING_DIMENSION,
    DEFAULT_MAX_SEQ_LENGTH,
    embed_query,
    embed_texts,
    get_embedding_model,
    get_model_info,
)


class TestEmbedderUnit:
    """Unit tests verifying dimensions, normalization, determinism, and batch behavior."""

    def test_model_metadata_and_dimensions(self) -> None:
        """Model info must report 384 dimensions and 256 default max sequence length."""
        info = get_model_info()
        assert info["dimension"] == 384
        assert info["dimension"] == EMBEDDING_DIMENSION
        assert info["max_seq_length"] == DEFAULT_MAX_SEQ_LENGTH
        assert info["max_position_embeddings"] == 512

    def test_embed_query_dimension_and_normalization(self) -> None:
        """embed_query should produce a 384-dimensional unit-normalized vector."""
        query = "What is retrieval-augmented generation and hallucination detection?"
        vec = embed_query(query)

        assert isinstance(vec, list)
        assert len(vec) == 384
        assert all(isinstance(x, float) for x in vec)

        # Confirm unit vector normalization (L2 norm ≈ 1.0)
        l2_norm = math.sqrt(sum(x * x for x in vec))
        assert pytest.approx(l2_norm, abs=1e-4) == 1.0

    def test_embed_texts_batch_dimension_and_count(self) -> None:
        """embed_texts should produce exactly N vectors of dimension 384."""
        samples = [
            "Self-correcting RAG combines autonomous reflection with dense retrieval.",
            "Pinecone is a cloud-native vector database optimized for similarity search.",
            "Natural Language Inference verifies factual consistency between context and claims.",
            "Cross-encoders score pairs of query and document for reranking.",
        ]
        vectors = embed_texts(samples, batch_size=2)

        assert len(vectors) == len(samples)
        for vec in vectors:
            assert len(vec) == 384
            l2_norm = math.sqrt(sum(x * x for x in vec))
            assert pytest.approx(l2_norm, abs=1e-4) == 1.0

    def test_embedding_determinism(self) -> None:
        """The exact same input text must deterministically produce the exact same vector."""
        text = "Deterministic inference guarantees reproducible retrieval and evaluation benchmarks."

        vec1 = embed_query(text)
        vec2 = embed_query(text)

        assert len(vec1) == len(vec2) == 384
        # Assert each floating point value matches exactly
        for a, b in zip(vec1, vec2):
            assert a == b, f"Vector mismatch at component: {a} != {b}"

    def test_batch_matches_single_embedding(self) -> None:
        """Batch embedding must match sequential single embeddings within floating-point precision."""
        texts = [
            "Passage Alpha discusses dense retrieval models.",
            "Passage Beta discusses sparse BM25 token representations.",
            "Passage Gamma discusses reciprocal rank fusion algorithms.",
        ]

        batch_results = embed_texts(texts, batch_size=3)
        single_results = [embed_query(t) for t in texts]

        assert len(batch_results) == len(single_results) == len(texts)
        for b_vec, s_vec in zip(batch_results, single_results):
            assert len(b_vec) == len(s_vec) == 384
            # Floating point cosine similarity between batch and single should be 1.0
            dot_product = sum(a * b for a, b in zip(b_vec, s_vec))
            assert pytest.approx(dot_product, abs=1e-5) == 1.0

    def test_empty_and_whitespace_inputs(self) -> None:
        """Empty texts should return zero-vectors without throwing exceptions."""
        assert embed_texts([]) == []

        # Single empty query returns zero vector
        empty_vec = embed_query("")
        assert len(empty_vec) == 384
        assert all(x == 0.0 for x in empty_vec)

        # Batch containing empty string
        mixed = ["Valid text passage.", "", "   ", "Another valid passage."]
        mixed_vecs = embed_texts(mixed)
        assert len(mixed_vecs) == 4
        # Index 1 and 2 are empty/whitespace -> zero vector
        assert all(x == 0.0 for x in mixed_vecs[1])
        assert all(x == 0.0 for x in mixed_vecs[2])
        # Index 0 and 3 are valid unit vectors
        assert pytest.approx(math.sqrt(sum(x * x for x in mixed_vecs[0])), abs=1e-4) == 1.0
        assert pytest.approx(math.sqrt(sum(x * x for x in mixed_vecs[3])), abs=1e-4) == 1.0

    def test_max_sequence_length_truncation_behavior(self) -> None:
        """Document and test the truncation behavior when input exceeds model.max_seq_length."""
        model = get_embedding_model()
        tokenizer = model.tokenizer

        # Short text: well under 256 tokens
        short_text = "Dense vector embeddings capture semantic meaning in high-dimensional space."
        short_tokens = len(tokenizer.encode(short_text))
        assert short_tokens < 256

        # Very long text: ~450 tokens (typical of Phase 9's ~500 token chunks)
        repeated_sentence = "Self-correcting RAG improves generation quality and prevents hallucinations. "
        long_text = repeated_sentence * 35  # ~420 tokens
        long_tokens = len(tokenizer.encode(long_text))
        assert long_tokens > 256

        # When encoded with default max_seq_length = 256, sentence-transformers auto-truncates
        emb_long = embed_query(long_text)
        assert len(emb_long) == 384

        # Verify that text beyond token 256 is truncated under max_seq_length = 256:
        # Construct a text with the identical first 256 tokens, but completely different afterwards
        token_ids_prefix = tokenizer.encode(long_text, truncation=True, max_length=256)[1:-1]
        prefix_text = tokenizer.decode(token_ids_prefix)
        divergent_long_text = prefix_text + " COMPLETELY DIVERGENT TAIL " + ("Oranges bananas apples grapes. " * 30)

        emb_prefix = embed_query(prefix_text)
        emb_divergent = embed_query(divergent_long_text)

        # Cosine similarity between original long text and divergent text is 1.0 because tokenizer truncated before divergence!
        sim = sum(a * b for a, b in zip(emb_long, emb_divergent))
        assert pytest.approx(sim, abs=1e-4) == 1.0, (
            "Expected tokenizer truncation at max_seq_length=256 to produce identical embeddings for divergent tails"
        )
