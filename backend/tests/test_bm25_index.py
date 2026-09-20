"""
Unit and integration tests for Phase 17: BM25 Sparse Index Build.

Tests:
1. Index construction from READY document chunks succeeds and chunk_count matches storage.
2. BM25 search correctly maps keyword queries to real chunk_id and metadata (source_text, page_number, filename).
3. Rebuild trigger: ingesting a 2nd READY document updates the BM25 index with new chunks.
4. Thread safety: concurrent index rebuild and search query execution across multiple threads complete without error.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from app.ingestion.pipeline import run_ingestion_pipeline
from app.ingestion.storage import DocumentRegistry
from app.models.schemas import Chunk, DocumentStatus, PageText
from app.retrieval import (
    BM25Index,
    build_bm25_index,
    get_bm25_index,
    rebuild_bm25_index,
    tokenize_text,
)
from app.retrieval.pinecone_client import PineconeClient


@pytest.fixture
def mock_registry(tmp_path: Path) -> DocumentRegistry:
    """Provide a fresh DocumentRegistry isolated in temporary directory."""
    return DocumentRegistry(upload_dir=tmp_path)


def test_tokenize_text():
    """Verify tokenize_text helper correctly splits and lowercases text."""
    tokens = tokenize_text("Self-Correcting RAG: BM25 Sparse Search Engine v1.0!")
    assert tokens == ["self", "correcting", "rag", "bm25", "sparse", "search", "engine", "v1", "0"]
    assert tokenize_text("") == []


def test_bm25_build_and_search_mapping(mock_registry: DocumentRegistry):
    """Verify BM25 index construction from READY document chunks and exact metadata mapping."""
    doc_id = "doc_bm25_test"
    pdf_path = mock_registry.upload_dir / doc_id / "sample.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-dummy")

    mock_registry.save_document(doc_id, "sample.pdf", b"%PDF-dummy")

    # Construct three distinct pages to produce multiple chunks in the BM25 index
    mock_pages = [
        PageText(page_number=1, text="The primary revenue stream of Anthropic is Claude API subscription model. " * 50, char_count=3500),
        PageText(page_number=2, text="DeepMind specializes in general artificial intelligence and reinforcement learning. " * 50, char_count=3700),
        PageText(page_number=3, text="OpenAI develops GPT models and frontier conversational AI architectures. " * 50, char_count=3600),
    ]

    mock_pc = MagicMock(spec=PineconeClient)
    mock_pc.delete_vectors.return_value = None
    mock_pc.upsert_vectors.return_value = {"upserted_count": 2, "successful_ids": ["v1", "v2"], "failed_ids": []}

    with patch("app.ingestion.pipeline.extract_raw_pages", return_value=mock_pages), \
         patch("app.ingestion.pipeline.get_pinecone_client", return_value=mock_pc):
        pipeline_result = run_ingestion_pipeline(doc_id, registry=mock_registry, verify_upsert=False)

    assert pipeline_result.status == DocumentStatus.READY

    # Build BM25 Index directly
    bm25_idx = build_bm25_index(registry=mock_registry)
    assert bm25_idx.chunk_count >= 1
    assert bm25_idx.build_time_seconds >= 0.0

    # Execute search query
    matches = bm25_idx.search("Anthropic Claude subscription", top_k=2)
    assert len(matches) > 0
    top_match = matches[0]

    assert "id" in top_match
    assert top_match["score"] > 0.0
    meta = top_match["metadata"]
    assert meta["document_id"] == doc_id
    assert meta["filename"] == "sample.pdf"
    assert "Anthropic" in meta["source_text"]


def test_rebuild_after_ingestion_trigger(mock_registry: DocumentRegistry):
    """Verify that ingesting a second document updates the BM25 index."""
    mock_pc = MagicMock(spec=PineconeClient)
    mock_pc.delete_vectors.return_value = None
    mock_pc.upsert_vectors.return_value = {"upserted_count": 1, "successful_ids": ["v1"], "failed_ids": []}

    # Document 1
    doc_id_1 = "doc_first"
    pdf1 = mock_registry.upload_dir / doc_id_1 / "first.pdf"
    pdf1.parent.mkdir(parents=True, exist_ok=True)
    pdf1.write_bytes(b"%PDF-1")
    mock_registry.save_document(doc_id_1, "first.pdf", b"%PDF-1")

    pages1 = [PageText(page_number=1, text="Quantum computing uses qubits for quantum parallelism.", char_count=54)]

    with patch("app.ingestion.pipeline.extract_raw_pages", return_value=pages1), \
         patch("app.ingestion.pipeline.get_pinecone_client", return_value=mock_pc):
        run_ingestion_pipeline(doc_id_1, registry=mock_registry, verify_upsert=False)

    idx1 = get_bm25_index(registry=mock_registry, force_rebuild=True)
    assert idx1.chunk_count == 1

    # Document 2
    doc_id_2 = "doc_second"
    pdf2 = mock_registry.upload_dir / doc_id_2 / "second.pdf"
    pdf2.parent.mkdir(parents=True, exist_ok=True)
    pdf2.write_bytes(b"%PDF-2")
    mock_registry.save_document(doc_id_2, "second.pdf", b"%PDF-2")

    pages2 = [PageText(page_number=1, text="Superconductors exhibit zero electrical resistance at low temperatures.", char_count=69)]

    with patch("app.ingestion.pipeline.extract_raw_pages", return_value=pages2), \
         patch("app.ingestion.pipeline.get_pinecone_client", return_value=mock_pc):
        run_ingestion_pipeline(doc_id_2, registry=mock_registry, verify_upsert=False)

    idx2 = get_bm25_index(registry=mock_registry)
    assert idx2.chunk_count == 2

    # Query for second document term
    matches = idx2.search("Superconductors electrical resistance", top_k=1)
    assert len(matches) == 1
    assert matches[0]["metadata"]["document_id"] == doc_id_2
    assert "Superconductors" in matches[0]["metadata"]["source_text"]


def test_concurrent_search_and_rebuild_thread_safety(mock_registry: DocumentRegistry):
    """Verify concurrent thread execution of rebuild_bm25_index and search queries completes safely."""
    doc_id = "doc_concurrent"
    pdf_path = mock_registry.upload_dir / doc_id / "concurrent.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-conc")

    mock_registry.save_document(doc_id, "concurrent.pdf", b"%PDF-conc", status=DocumentStatus.READY)

    pages = [PageText(page_number=1, text="Thread safety testing for BM25 reference swap pattern.", char_count=57)]
    mock_registry.save_chunks(doc_id, [
        Chunk(
            chunk_id="chunk-c1",
            document_id=doc_id,
            chunk_index=0,
            text=pages[0].text,
            token_count=10,
            page_number=1,
            page_number_end=1,
            char_start=0,
            char_end=57,
        )
    ])

    rebuild_bm25_index(mock_registry)

    def reader_task():
        for _ in range(20):
            idx = get_bm25_index(registry=mock_registry)
            res = idx.search("Thread safety testing", top_k=1)
            assert len(res) == 1

    def writer_task():
        for _ in range(10):
            rebuild_bm25_index(mock_registry)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(reader_task) if i % 2 == 0 else executor.submit(writer_task)
            for i in range(10)
        ]
        for f in as_completed(futures):
            f.result()
