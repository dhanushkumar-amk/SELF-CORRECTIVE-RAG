"""
Unit tests for Phase 23: Cross-Encoder Relevance Thresholding & Selection.
"""

from app.models.schemas import RetrievalResult
from app.reranking import select_relevant_chunks


def test_threshold_filtering_excludes_low_score_chunks():
    """Verify select_relevant_chunks filters out candidates below min_score threshold."""
    cands = [
        RetrievalResult(
            chunk_id="c_relevant",
            score=0.1,
            metadata={"source_text": "Sentence-transformers MiniLM produces 384 dimensional dense embeddings."},
        ),
        RetrievalResult(
            chunk_id="c_irrelevant",
            score=0.1,
            metadata={"source_text": "Chocolate chip cookie recipe requires baking soda and brown sugar."},
        ),
    ]

    # Query about sentence-transformers embeddings
    res = select_relevant_chunks(
        query="What dimension embeddings does MiniLM produce?",
        candidates=cands,
        min_score=-2.0,
        top_n=5,
    )

    assert res.reason is None
    assert len(res.chunks) == 1
    assert res.chunks[0].chunk_id == "c_relevant"
    assert res.total_candidates == 2
    assert res.relevant_count == 1


def test_no_relevant_chunks_found_trigger():
    """Verify that when all candidates fall below threshold, 'no_relevant_chunks_found' is returned."""
    cands = [
        RetrievalResult(
            chunk_id="c1",
            score=0.1,
            metadata={"source_text": "Reciprocal Rank Fusion combines sparse and dense retrieval rankings."},
        ),
        RetrievalResult(
            chunk_id="c2",
            score=0.1,
            metadata={"source_text": "Pinecone vector index manages cosine similarity search."},
        ),
    ]

    # Out-of-scope query
    irrelevant_query = "What is the capital city of Australia and weather forecast?"
    res = select_relevant_chunks(
        query=irrelevant_query,
        candidates=cands,
        min_score=-2.0,
        top_n=5,
    )

    assert res.chunks == []
    assert res.reason == "no_relevant_chunks_found"
    assert res.total_candidates == 2
    assert res.relevant_count == 0


def test_relevant_query_does_not_trigger_reason():
    """Verify that a genuinely relevant query returns valid chunks with reason=None."""
    cands = [
        RetrievalResult(
            chunk_id="c_bm25",
            score=0.1,
            metadata={"source_text": "BM25Okapi scores keyword frequency against inverse document frequency."},
        ),
    ]

    res = select_relevant_chunks(
        query="How does BM25 score keyword frequency?",
        candidates=cands,
        min_score=-2.0,
        top_n=5,
    )

    assert len(res.chunks) == 1
    assert res.reason is None
    assert res.relevant_count == 1
    assert res.chunks[0].chunk_id == "c_bm25"


def test_threshold_filtering_happens_before_top_n_truncation():
    """Verify that threshold filtering is applied BEFORE top_n truncation.

    If 3 out of 6 candidates pass the relevance threshold and top_n=5, the function
    must return exactly 3 chunks (not 5 padded candidates).
    """
    cands = [
        # 3 Relevant chunks
        RetrievalResult(
            chunk_id="rel_1",
            score=0.1,
            metadata={"source_text": "Sentence-transformers MiniLM model embeds text into 384 dim vectors."},
        ),
        RetrievalResult(
            chunk_id="rel_2",
            score=0.1,
            metadata={"source_text": "MiniLM dense vector embeddings are normalized for cosine similarity."},
        ),
        RetrievalResult(
            chunk_id="rel_3",
            score=0.1,
            metadata={"source_text": "The sentence-transformers library supports local CPU embedding inference."},
        ),
        # 3 Irrelevant chunks
        RetrievalResult(
            chunk_id="irrel_1",
            score=0.1,
            metadata={"source_text": "Baking sourdough bread requires flour, water, salt, and wild yeast culture."},
        ),
        RetrievalResult(
            chunk_id="irrel_2",
            score=0.1,
            metadata={"source_text": "Paris is the capital and largest city of France situated on the Seine river."},
        ),
        RetrievalResult(
            chunk_id="irrel_3",
            score=0.1,
            metadata={"source_text": "Toyota Camry features a 2.5 liter 4-cylinder engine with front-wheel drive."},
        ),
    ]

    query = "Sentence-transformers MiniLM dense embedding dimensions"

    # Request top_n=5 when 6 total candidates exist and only 3 are relevant
    res = select_relevant_chunks(
        query=query,
        candidates=cands,
        min_score=-2.0,
        top_n=5,
    )

    # Must return EXACTLY 3 relevant chunks, proving filtering happened BEFORE top_n truncation!
    assert res.total_candidates == 6
    assert res.relevant_count == 3
    assert len(res.chunks) == 3
    assert res.reason is None
    retrieved_ids = {c.chunk_id for c in res.chunks}
    assert retrieved_ids == {"rel_1", "rel_2", "rel_3"}
