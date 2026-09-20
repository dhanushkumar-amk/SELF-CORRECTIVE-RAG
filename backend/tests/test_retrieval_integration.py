"""
Integration tests and edge case sweep for Phase 21: Retrieval Layer Testing.

Tests:
1. End-to-End Integration Flow: PDF ingestion -> index build -> hybrid_search retrieval.
2. Multi-Document Retrieval & Scoping: query relevance across documents and filter={"document_id": ...} isolation.
3. Retrieval Consistency: identical query results across 3 repeated runs.
4. Edge Case Sweep:
   - Empty/whitespace queries
   - Extremely long queries (>256 token embedding context limit)
   - Special characters & unicode in query strings
   - top_k greater than total available document chunks
   - Graceful degradation fallback when dense or sparse search fails
5. Retrieval Quality Spot-Check & Latency Baseline (5 QA pairs).
"""

import time
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from app.ingestion.chunk_metadata import create_vector_id
from app.ingestion.pipeline import run_ingestion_pipeline
from app.ingestion.storage import DocumentRegistry
from app.models.schemas import Chunk, DocumentStatus, PageText, RetrievalResult
from app.retrieval import (
    BM25Index,
    build_bm25_index,
    dense_search,
    get_bm25_index,
    hybrid_search,
    reciprocal_rank_fusion,
    sparse_search,
)
from app.retrieval.pinecone_client import PineconeClient


@pytest.fixture
def mock_registry(tmp_path: Path) -> DocumentRegistry:
    """Provide isolated DocumentRegistry fixture."""
    return DocumentRegistry(upload_dir=tmp_path)


def test_e2e_full_retrieval_pipeline(mock_registry: DocumentRegistry):
    """Test full retrieval flow: ingest PDF -> build BM25 -> query hybrid_search()."""
    doc_id = "doc_e2e_full"
    pdf_path = mock_registry.upload_dir / doc_id / "document.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4 dummy")
    mock_registry.save_document(doc_id, "document.pdf", b"%PDF-1.4 dummy")

    pages = [
        PageText(page_number=1, text="The primary revenue model of Anthropic relies on Claude API subscriptions. " * 20, char_count=1500),
        PageText(page_number=2, text="Google DeepMind develops AlphaFold and Gemini multimodal foundation models. " * 20, char_count=1600),
    ]

    mock_pc = MagicMock(spec=PineconeClient)
    mock_pc.delete_vectors.return_value = None
    mock_pc.upsert_vectors.return_value = {"upserted_count": 2, "successful_ids": ["v1", "v2"], "failed_ids": []}

    with patch("app.ingestion.pipeline.extract_raw_pages", return_value=pages), \
         patch("app.ingestion.pipeline.get_pinecone_client", return_value=mock_pc), \
         patch("app.retrieval.dense_search.embed_query", return_value=[0.05] * 384):

        result = run_ingestion_pipeline(doc_id, registry=mock_registry, verify_upsert=False)
        assert result.status == DocumentStatus.READY

        doc_chunks = mock_registry.get_chunks(doc_id)
        assert len(doc_chunks) >= 1
        c1 = doc_chunks[0]
        c1_vector_id = create_vector_id(doc_id, c1.chunk_id)

        mock_pc.query_vectors.return_value = [
            {"id": c1_vector_id, "score": 0.89, "metadata": c1.to_pinecone_metadata(filename="document.pdf")},
        ]

        bm25_idx = build_bm25_index(registry=mock_registry)
        assert bm25_idx.chunk_count >= 1

        results = hybrid_search(
            query="Anthropic Claude API subscription revenue",
            top_k=2,
            pinecone_client=mock_pc,
            bm25_index=bm25_idx,
        )

        assert len(results) > 0
        top_match = results[0]
        assert top_match.chunk_id == c1_vector_id
        assert top_match.score > 0.0
        assert "document_id" in top_match.metadata
        assert top_match.metadata["filename"] == "document.pdf"


def test_multi_document_retrieval_and_scoping(mock_registry: DocumentRegistry):
    """Test retrieval across multiple documents and document_id metadata filtering isolation."""
    doc1_id = "doc_quantum"
    doc2_id = "doc_chemistry"
    doc3_id = "doc_astronomy"

    mock_registry.save_document(doc1_id, "quantum.pdf", b"%PDF-quantum")
    mock_registry.save_document(doc2_id, "chemistry.pdf", b"%PDF-chemistry")
    mock_registry.save_document(doc3_id, "astronomy.pdf", b"%PDF-astronomy")

    chunk1 = Chunk(
        chunk_id="chunk_q1",
        document_id=doc1_id,
        chunk_index=0,
        text="Quantum superposition and entanglement govern qubit state coherence.",
        token_count=14,
        page_number=1,
        page_number_end=1,
        char_start=0,
        char_end=68,
    )
    chunk2 = Chunk(
        chunk_id="chunk_c1",
        document_id=doc2_id,
        chunk_index=0,
        text="Covalent electron pair sharing forms stable organic hydrocarbon chains.",
        token_count=13,
        page_number=1,
        page_number_end=1,
        char_start=0,
        char_end=71,
    )
    chunk3 = Chunk(
        chunk_id="chunk_a1",
        document_id=doc3_id,
        chunk_index=0,
        text="Stellar nucleosynthesis fuses hydrogen nuclei into helium inside stellar cores.",
        token_count=12,
        page_number=1,
        page_number_end=1,
        char_start=0,
        char_end=79,
    )

    mock_registry.save_chunks(doc1_id, [chunk1])
    mock_registry.save_chunks(doc2_id, [chunk2])
    mock_registry.save_chunks(doc3_id, [chunk3])

    for d_id in (doc1_id, doc2_id, doc3_id):
        mock_registry.update_document_metadata(d_id, status=DocumentStatus.PROCESSING)
        mock_registry.update_document_metadata(d_id, status=DocumentStatus.READY)

    c1_vid = create_vector_id(doc1_id, chunk1.chunk_id)
    c2_vid = create_vector_id(doc2_id, chunk2.chunk_id)
    c3_vid = create_vector_id(doc3_id, chunk3.chunk_id)

    mock_pc = MagicMock(spec=PineconeClient)

    def mock_query_vectors(vector, top_k=5, filter=None, include_metadata=True, namespace=""):
        all_matches = [
            {"id": c1_vid, "score": 0.95, "metadata": chunk1.to_pinecone_metadata(filename="quantum.pdf")},
            {"id": c2_vid, "score": 0.85, "metadata": chunk2.to_pinecone_metadata(filename="chemistry.pdf")},
            {"id": c3_vid, "score": 0.75, "metadata": chunk3.to_pinecone_metadata(filename="astronomy.pdf")},
        ]
        if filter and "document_id" in filter:
            target_doc = filter["document_id"]
            return [m for m in all_matches if m["metadata"]["document_id"] == target_doc][:top_k]
        return all_matches[:top_k]

    mock_pc.query_vectors.side_effect = mock_query_vectors

    bm25_idx = build_bm25_index(registry=mock_registry)
    assert bm25_idx.chunk_count == 3

    with patch("app.retrieval.dense_search.embed_query", return_value=[0.1] * 384):
        # 1. Unscoped query returns top match from relevant doc (doc1 ranks #1 in dense & sparse)
        unscoped = hybrid_search("quantum qubit superposition", top_k=5, pinecone_client=mock_pc, bm25_index=bm25_idx)
        assert len(unscoped) >= 1
        assert unscoped[0].metadata["document_id"] == doc1_id, f"Expected {doc1_id}, got {unscoped[0]}"

        # 2. Scoped query to doc1_id MUST exclude other document chunks
        scoped_doc1 = hybrid_search(
            "hydrocarbon covalent",
            top_k=5,
            filter={"document_id": doc1_id},
            pinecone_client=mock_pc,
            bm25_index=bm25_idx,
        )
        assert len(scoped_doc1) == 1
        for res in scoped_doc1:
            assert res.metadata["document_id"] == doc1_id
            assert res.metadata["document_id"] != doc2_id

        # 3. Scoped query to doc2_id MUST exclude other document chunks
        scoped_doc2 = hybrid_search(
            "quantum qubit",
            top_k=5,
            filter={"document_id": doc2_id},
            pinecone_client=mock_pc,
            bm25_index=bm25_idx,
        )
        assert len(scoped_doc2) == 1
        for res in scoped_doc2:
            assert res.metadata["document_id"] == doc2_id
            assert res.metadata["document_id"] != doc1_id


def test_retrieval_determinism_and_consistency(mock_registry: DocumentRegistry):
    """Verify that executing the exact same query 3 times produces identical results each time."""
    mock_pc = MagicMock(spec=PineconeClient)
    mock_pc.query_vectors.return_value = [
        {"id": "chunk_det_1", "score": 0.95, "metadata": {"source_text": "Deterministic result 1", "document_id": "d1"}},
        {"id": "chunk_det_2", "score": 0.85, "metadata": {"source_text": "Deterministic result 2", "document_id": "d1"}},
    ]

    bm25_mock = MagicMock(spec=BM25Index)
    bm25_mock.search.return_value = [
        {"id": "chunk_det_1", "score": 12.0, "metadata": {"source_text": "Deterministic result 1", "document_id": "d1"}},
        {"id": "chunk_det_2", "score": 8.0, "metadata": {"source_text": "Deterministic result 2", "document_id": "d1"}},
    ]

    query = "Deterministic test query consistency"

    with patch("app.retrieval.dense_search.embed_query", return_value=[0.2] * 384):
        run1 = hybrid_search(query, top_k=2, pinecone_client=mock_pc, bm25_index=bm25_mock)
        run2 = hybrid_search(query, top_k=2, pinecone_client=mock_pc, bm25_index=bm25_mock)
        run3 = hybrid_search(query, top_k=2, pinecone_client=mock_pc, bm25_index=bm25_mock)

    assert len(run1) == len(run2) == len(run3) == 2
    assert [r.chunk_id for r in run1] == [r.chunk_id for r in run2] == [r.chunk_id for r in run3]
    assert [r.score for r in run1] == [r.score for r in run2] == [r.score for r in run3]


def test_edge_cases_sweep():
    """Verify hybrid_search handles edge cases gracefully."""
    mock_pc = MagicMock(spec=PineconeClient)
    mock_pc.query_vectors.return_value = []
    bm25_mock = MagicMock(spec=BM25Index)
    bm25_mock.search.return_value = []

    # 1. Empty / whitespace query string
    assert hybrid_search("", top_k=5, pinecone_client=mock_pc, bm25_index=bm25_mock) == []
    assert hybrid_search("   \n\t  ", top_k=5, pinecone_client=mock_pc, bm25_index=bm25_mock) == []

    # 2. Extremely long query string (> 500 words, exceeding 256 token embedding model context)
    long_query = "quantum " * 600
    with patch("app.retrieval.dense_search.embed_query", return_value=[0.1] * 384):
        long_res = hybrid_search(long_query, top_k=5, pinecone_client=mock_pc, bm25_index=bm25_mock)
        assert isinstance(long_res, list)

    # 3. Special characters & Unicode in query
    unicode_query = "Self-Correcting RAG @#$! %^&* 🤖 𝛂 + 𝛃 = 𝛄 test query"
    with patch("app.retrieval.dense_search.embed_query", return_value=[0.1] * 384):
        unicode_res = hybrid_search(unicode_query, top_k=5, pinecone_client=mock_pc, bm25_index=bm25_mock)
        assert isinstance(unicode_res, list)

    # 4. top_k larger than available chunks (asking top_k=50 when only 2 chunks exist)
    mock_pc_small = MagicMock(spec=PineconeClient)
    mock_pc_small.query_vectors.return_value = [
        {"id": "c1", "score": 0.9, "metadata": {"source_text": "Chunk 1"}},
        {"id": "c2", "score": 0.8, "metadata": {"source_text": "Chunk 2"}},
    ]
    bm25_small = MagicMock(spec=BM25Index)
    bm25_small.search.return_value = [
        {"id": "c1", "score": 10.0, "metadata": {"source_text": "Chunk 1"}},
    ]

    with patch("app.retrieval.dense_search.embed_query", return_value=[0.1] * 384):
        over_k_res = hybrid_search("test query", top_k=50, pinecone_client=mock_pc_small, bm25_index=bm25_small)
        assert len(over_k_res) == 2  # Returns all available chunks without crashing or padding


def test_graceful_degradation_failures():
    """Verify hybrid_search degrades gracefully when dense or sparse backend fails."""
    # 1. Dense search throws exception -> Fallback to sparse BM25 search
    mock_pc_failing = MagicMock(spec=PineconeClient)
    mock_pc_failing.query_vectors.side_effect = Exception("Pinecone connection timeout")

    bm25_working = MagicMock(spec=BM25Index)
    bm25_working.search.return_value = [
        {"id": "bm25_fallback", "score": 15.0, "metadata": {"source_text": "Sparse fallback text"}},
    ]

    with patch("app.retrieval.dense_search.embed_query", return_value=[0.1] * 384):
        fallback_res = hybrid_search("test fallback", top_k=5, pinecone_client=mock_pc_failing, bm25_index=bm25_working)
        assert len(fallback_res) == 1
        assert fallback_res[0].chunk_id == "bm25_fallback"

    # 2. Sparse search throws exception -> Fallback to dense Pinecone search
    mock_pc_working = MagicMock(spec=PineconeClient)
    mock_pc_working.query_vectors.return_value = [
        {"id": "pinecone_fallback", "score": 0.91, "metadata": {"source_text": "Dense fallback text"}},
    ]
    bm25_failing = MagicMock(spec=BM25Index)
    bm25_failing.search.side_effect = Exception("BM25 index error")

    with patch("app.retrieval.dense_search.embed_query", return_value=[0.1] * 384):
        fallback_res2 = hybrid_search("test fallback 2", top_k=5, pinecone_client=mock_pc_working, bm25_index=bm25_failing)
        assert len(fallback_res2) == 1
        assert fallback_res2[0].chunk_id == "pinecone_fallback"

    # 3. Both fail -> returns empty list gracefully
    with patch("app.retrieval.dense_search.embed_query", return_value=[0.1] * 384):
        both_failed = hybrid_search("test both fail", top_k=5, pinecone_client=mock_pc_failing, bm25_index=bm25_failing)
        assert both_failed == []


def test_retrieval_spot_check_and_latency_baseline(mock_registry: DocumentRegistry):
    """Informal 5 QA spot-check evaluation and latency baseline measurement."""
    doc_id = "doc_rag_eval"
    pdf_path = mock_registry.upload_dir / doc_id / "rag_guide.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-rag")
    mock_registry.save_document(doc_id, "rag_guide.pdf", b"%PDF-rag")

    pages = [
        PageText(page_number=1, text="The embedding generator uses sentence-transformers all-MiniLM-L6-v2 producing 384 dimensional vectors for semantic indexing. " * 10, char_count=1400),
        PageText(page_number=2, text="BM25 search uses rank_bm25 BM25Okapi for keyword matching over tokenized document terms. " * 10, char_count=1100),
        PageText(page_number=3, text="Reciprocal Rank Fusion calculates chunk relevance using k=60 rank decay constant across retrieval lists. " * 10, char_count=1200),
        PageText(page_number=4, text="Pinecone manages cloud vector storage and cosine similarity vector queries. " * 10, char_count=900),
        PageText(page_number=5, text="LangGraph state machine executes self-correction when NLI verification detects hallucinated claims. " * 10, char_count=1300),
    ]

    mock_pc = MagicMock(spec=PineconeClient)
    mock_pc.delete_vectors.return_value = None
    mock_pc.upsert_vectors.return_value = {"upserted_count": 5, "successful_ids": ["v1", "v2", "v3", "v4", "v5"], "failed_ids": []}

    with patch("app.ingestion.pipeline.extract_raw_pages", return_value=pages), \
         patch("app.ingestion.pipeline.get_pinecone_client", return_value=mock_pc), \
         patch("app.retrieval.dense_search.embed_query", return_value=[0.1] * 384):

        run_ingestion_pipeline(doc_id, registry=mock_registry, verify_upsert=False)
        bm25_idx = build_bm25_index(registry=mock_registry)
        chunks = mock_registry.get_chunks(doc_id)
        assert len(chunks) >= 1

        def mock_dense_query(vector, top_k=5, filter=None, include_metadata=True, namespace=""):
            matches = []
            for idx, c in enumerate(chunks):
                vid = create_vector_id(doc_id, c.chunk_id)
                score = round(0.95 - (idx * 0.05), 2)
                matches.append({"id": vid, "score": score, "metadata": c.to_pinecone_metadata(filename="rag_guide.pdf")})
            return matches[:top_k]

        mock_pc.query_vectors.side_effect = mock_dense_query

        qa_pairs = [
            ("What model does sentence-transformers use for dense embeddings?", "all-MiniLM-L6-v2"),
            ("How does BM25 calculate keyword relevance scores?", "BM25Okapi"),
            ("What constant parameter k is used in Reciprocal Rank Fusion?", "k=60"),
            ("What database vector index stores chunk embeddings?", "Pinecone"),
            ("How does LangGraph self-correction handle hallucinated answers?", "LangGraph"),
        ]

        latencies_ms = []
        hits = 0

        for query_text, expected_keyword in qa_pairs:
            start_t = time.perf_counter()
            results = hybrid_search(query_text, top_k=3, pinecone_client=mock_pc, bm25_index=bm25_idx)
            duration_ms = (time.perf_counter() - start_t) * 1000.0
            latencies_ms.append(duration_ms)

            # Check if expected keyword appears in top 3 results
            top_texts = [r.metadata.get("source_text", "") for r in results]
            if any(expected_keyword in text for text in top_texts):
                hits += 1

        avg_latency = sum(latencies_ms) / len(latencies_ms)
        print(f"\n[Spot-Check Results] Accuracy Score: {hits}/5 ({(hits/5)*100:.0f}%) | Average Latency: {avg_latency:.2f} ms")

        assert hits == 5
        assert avg_latency < 500.0  # Latency under 500ms benchmark
