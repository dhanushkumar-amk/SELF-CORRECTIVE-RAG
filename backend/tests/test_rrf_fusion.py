"""
Unit and integration tests for Phase 19: Reciprocal Rank Fusion (RRF) Logic.

Tests:
1. Hand-computed test with known mathematical RRF values (exact score assertions).
2. Agreement boosting: chunk present in both lists ranks higher than single-list chunks.
3. Deduplication: unique chunk_ids appear exactly once in fused output.
4. Single-list appearance handling: chunks in only one list get valid non-zero score.
5. Metadata preservation and consistency verification.
6. Edge cases: empty input lists, single list, invalid k values.
7. End-to-end integration test with dense_search() + sparse_search() + RRF.
"""

from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from app.ingestion.pipeline import run_ingestion_pipeline
from app.ingestion.storage import DocumentRegistry
from app.models.schemas import DocumentStatus, PageText, RetrievalResult
from app.retrieval import (
    build_bm25_index,
    dense_search,
    reciprocal_rank_fusion,
    sparse_search,
)
from app.retrieval.pinecone_client import PineconeClient


def test_rrf_hand_computed_known_values():
    """Verify RRF formula against hand-calculated expected values with exact precision.
    
    Hand-calculated setup:
    k = 60
    List 1 (Dense):  [chunk_A (rank 1), chunk_B (rank 2)]
    List 2 (Sparse): [chunk_B (rank 1), chunk_C (rank 2)]

    Expected RRF Scores:
    - chunk_B: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 123/3782 = 0.03252247488096245
    - chunk_A: 1/(60+1) = 1/61 = 0.01639344262295082
    - chunk_C: 1/(60+2) = 1/62 = 0.016129032258064516
    
    Expected Fused Ranking: chunk_B -> chunk_A -> chunk_C
    """
    dense_results = [
        RetrievalResult(chunk_id="chunk_A", score=0.95, metadata={"source_text": "Text A", "page_number": 1}),
        RetrievalResult(chunk_id="chunk_B", score=0.85, metadata={"source_text": "Text B", "page_number": 2}),
    ]
    sparse_results = [
        RetrievalResult(chunk_id="chunk_B", score=12.5, metadata={"source_text": "Text B", "page_number": 2}),
        RetrievalResult(chunk_id="chunk_C", score=9.1, metadata={"source_text": "Text C", "page_number": 3}),
    ]

    fused = reciprocal_rank_fusion([dense_results, sparse_results], k=60)

    # 1. Output length (deduplicated: 3 unique chunks)
    assert len(fused) == 3

    # 2. Ranking order
    assert [r.chunk_id for r in fused] == ["chunk_B", "chunk_A", "chunk_C"]

    # 3. Exact hand-computed score assertions
    expected_chunk_b_score = (1.0 / 62.0) + (1.0 / 61.0)  # ~0.03252247
    expected_chunk_a_score = 1.0 / 61.0                   # ~0.01639344
    expected_chunk_c_score = 1.0 / 62.0                   # ~0.01612903

    assert fused[0].score == pytest.approx(expected_chunk_b_score, rel=1e-6)
    assert fused[1].score == pytest.approx(expected_chunk_a_score, rel=1e-6)
    assert fused[2].score == pytest.approx(expected_chunk_c_score, rel=1e-6)


def test_rrf_agreement_boost():
    """Verify that agreement between retrieval methods boosts a chunk above single-list matches.
    
    Setup:
    List 1 (Dense):  [chunk_top_dense (rank 1), chunk_dual (rank 2)]
    List 2 (Sparse): [chunk_dual (rank 1), chunk_top_sparse (rank 2)]

    RRF Scores (k=60):
    - chunk_dual: 1/(60+2) + 1/(60+1) = 0.03252247
    - chunk_top_dense: 1/(60+1) = 0.01639344
    - chunk_top_sparse: 1/(60+2) = 0.01612903
    
    chunk_dual ranks #1 overall due to dual-method agreement boost!
    """
    dense_results = [
        RetrievalResult(chunk_id="chunk_top_dense", score=0.99, metadata={"source_text": "Dense top"}),
        RetrievalResult(chunk_id="chunk_dual", score=0.80, metadata={"source_text": "Dual match"}),
    ]
    sparse_results = [
        RetrievalResult(chunk_id="chunk_dual", score=15.0, metadata={"source_text": "Dual match"}),
        RetrievalResult(chunk_id="chunk_top_sparse", score=10.0, metadata={"source_text": "Sparse top"}),
    ]

    fused = reciprocal_rank_fusion([dense_results, sparse_results], k=60)
    assert fused[0].chunk_id == "chunk_dual"
    assert fused[0].score > fused[1].score


def test_rrf_deduplication():
    """Verify that chunks present in multiple result lists appear exactly once in the fused output."""
    res1 = [
        RetrievalResult(chunk_id="c1", score=0.9, metadata={"meta": 1}),
        RetrievalResult(chunk_id="c2", score=0.8, metadata={"meta": 2}),
    ]
    res2 = [
        RetrievalResult(chunk_id="c2", score=10.0, metadata={"meta": 2}),
        RetrievalResult(chunk_id="c1", score=5.0, metadata={"meta": 1}),
    ]
    res3 = [
        RetrievalResult(chunk_id="c1", score=0.5, metadata={"meta": 1}),
    ]

    fused = reciprocal_rank_fusion([res1, res2, res3], k=60)
    chunk_ids = [r.chunk_id for r in fused]
    assert len(chunk_ids) == 2
    assert set(chunk_ids) == {"c1", "c2"}


def test_rrf_single_list_chunk_handling():
    """Verify chunks appearing in only one list get valid contribution without requiring presence in both."""
    dense = [RetrievalResult(chunk_id="dense_only", score=0.9, metadata={"source": "dense"})]
    sparse = [RetrievalResult(chunk_id="sparse_only", score=10.0, metadata={"source": "sparse"})]

    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    assert len(fused) == 2
    
    # Both rank 1 in their respective lists, so score = 1/(60+1) for both
    expected_score = 1.0 / 61.0
    for res in fused:
        assert res.score == pytest.approx(expected_score, rel=1e-6)


def test_rrf_metadata_preservation_and_verification():
    """Verify metadata is correctly preserved and identical across dense and sparse instances."""
    dense_meta = {"source_text": "Shared text", "document_id": "doc1", "page_number": 5, "filename": "test.pdf"}
    sparse_meta = {"source_text": "Shared text", "document_id": "doc1", "page_number": 5, "filename": "test.pdf"}

    dense_results = [RetrievalResult(chunk_id="shared_chunk", score=0.88, metadata=dense_meta)]
    sparse_results = [RetrievalResult(chunk_id="shared_chunk", score=14.2, metadata=sparse_meta)]

    fused = reciprocal_rank_fusion([dense_results, sparse_results], k=60)
    assert len(fused) == 1
    assert fused[0].metadata == dense_meta
    assert fused[0].metadata["source_text"] == "Shared text"
    assert fused[0].metadata["page_number"] == 5


def test_rrf_edge_cases():
    """Test boundary conditions for reciprocal_rank_fusion."""
    # 1. Empty input lists
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []

    # 2. Single list
    single_list = [
        RetrievalResult(chunk_id="c1", score=0.9, metadata={}),
        RetrievalResult(chunk_id="c2", score=0.7, metadata={}),
    ]
    fused = reciprocal_rank_fusion([single_list], k=60)
    assert len(fused) == 2
    assert fused[0].chunk_id == "c1"
    assert fused[0].score == pytest.approx(1.0 / 61.0)
    assert fused[1].chunk_id == "c2"
    assert fused[1].score == pytest.approx(1.0 / 62.0)

    # 3. Invalid k value
    with pytest.raises(ValueError, match="RRF parameter k must be positive"):
        reciprocal_rank_fusion([single_list], k=0)
    with pytest.raises(ValueError, match="RRF parameter k must be positive"):
        reciprocal_rank_fusion([single_list], k=-10)


@pytest.fixture
def mock_registry(tmp_path: Path) -> DocumentRegistry:
    """Provide isolated DocumentRegistry fixture."""
    return DocumentRegistry(upload_dir=tmp_path)


def test_e2e_hybrid_retrieval_fusion(mock_registry: DocumentRegistry):
    """End-to-end integration test of dense_search() + sparse_search() + RRF fusion on ingested chunks."""
    doc_id = "doc_rrf_e2e"
    pdf_path = mock_registry.upload_dir / doc_id / "sample.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-dummy")

    mock_registry.save_document(doc_id, "sample.pdf", b"%PDF-dummy")

    # Ingest mock pages with distinct content
    pages = [
        PageText(page_number=1, text="Self-Correcting RAG uses LangGraph state machines for hallucination correction. " * 30, char_count=2400),
        PageText(page_number=2, text="Reciprocal Rank Fusion combines vector cosine distance and BM25 scores. " * 30, char_count=2300),
        PageText(page_number=3, text="Pinecone handles dense vector similarity queries with MiniLM embeddings. " * 30, char_count=2300),
    ]

    mock_pc = MagicMock(spec=PineconeClient)
    mock_pc.delete_vectors.return_value = None
    mock_pc.upsert_vectors.return_value = {"upserted_count": 3, "successful_ids": ["v1", "v2", "v3"], "failed_ids": []}

    # Mock Pinecone query returning top matches for vector search
    mock_pc.query_vectors.return_value = [
        {"id": "chunk_page2", "score": 0.92, "metadata": {"source_text": "Reciprocal Rank Fusion combines...", "document_id": doc_id, "page_number": 2, "filename": "sample.pdf"}},
        {"id": "chunk_page3", "score": 0.81, "metadata": {"source_text": "Pinecone handles dense vector...", "document_id": doc_id, "page_number": 3, "filename": "sample.pdf"}},
    ]

    with patch("app.ingestion.pipeline.extract_raw_pages", return_value=pages), \
         patch("app.ingestion.pipeline.get_pinecone_client", return_value=mock_pc), \
         patch("app.retrieval.dense_search.embed_query", return_value=[0.1] * 384):
        
        pipeline_result = run_ingestion_pipeline(doc_id, registry=mock_registry, verify_upsert=False)
        assert pipeline_result.status == DocumentStatus.READY

        # Build BM25 index
        bm25_idx = build_bm25_index(registry=mock_registry)
        assert bm25_idx.chunk_count >= 1

        # Execute dense and sparse search
        query = "Reciprocal Rank Fusion BM25 scores"
        dense_results = dense_search(query, top_k=2, pinecone_client=mock_pc)
        sparse_results = sparse_search(query, top_k=2, bm25_index=bm25_idx)

        assert len(dense_results) > 0
        assert len(sparse_results) > 0

        # Execute RRF fusion
        fused_results = reciprocal_rank_fusion([dense_results, sparse_results], k=60)

        assert len(fused_results) > 0
        # Fused scores should be valid RRF scores
        for item in fused_results:
            assert 0.0 < item.score < 1.0
            assert "source_text" in item.metadata
            assert item.metadata["filename"] == "sample.pdf"
