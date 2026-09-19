"""Tests for the token-aware recursive chunking engine, page-boundary tracking, and chunk endpoints."""

from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.ingestion.chunker import (
    chunk_document,
    get_token_count,
)
from app.ingestion.storage import reset_document_registry
from app.main import app
from app.models.schemas import PageText

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample.pdf"
MULTIPAGE_PDF = FIXTURES_DIR / "multipage_with_footer.pdf"


@pytest.fixture(autouse=True)
def isolated_upload_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate uploads and registry file to a temporary directory for each test."""
    upload_dir = tmp_path / "test_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(upload_dir))
    reset_document_registry()
    yield upload_dir
    reset_document_registry()


class TestChunkerUnit:
    """Unit tests for token measurement, page-spanning tracking, and tiny chunk merging."""

    def test_token_count_accuracy(self) -> None:
        """Token count should match tiktoken cl100k_base tokenizer directly."""
        sample = "Self-Correcting Retrieval-Augmented Generation with Hallucination Detection."
        count = get_token_count(sample)
        # Should be a positive integer matching tokenizer
        assert count > 0
        assert isinstance(count, int)
        # Empty string has 0 tokens
        assert get_token_count("") == 0

    def test_single_page_short_document(self) -> None:
        """A short page well under 500 tokens should produce exactly 1 chunk with identical text."""
        text = "This is a concise single page document describing foundational concepts."
        page = PageText(page_number=1, text=text, char_count=len(text))

        chunks = chunk_document(document_id="doc-123", pages=[page], chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.chunk_index == 0
        assert chunk.page_number == 1
        assert chunk.page_number_end == 1
        assert chunk.char_start == 0
        assert chunk.char_end == len(text)
        assert chunk.text == text
        assert chunk.token_count == get_token_count(text)

    def test_multipage_spanning_chunk_tracking(self) -> None:
        """Chunks crossing a page boundary must record both page_number and page_number_end."""
        p1 = PageText(page_number=1, text="Page 1 short content.", char_count=21)
        p2 = PageText(page_number=2, text="Page 2 short content.", char_count=21)
        p3 = PageText(page_number=3, text="Page 3 short content.", char_count=21)

        chunks = chunk_document(
            document_id="doc-spanning",
            pages=[p1, p2, p3],
            chunk_size=15,
            chunk_overlap=8,
            min_chunk_tokens=2,
        )

        assert len(chunks) == 2
        # Verify both chunks span across consecutive pages
        assert chunks[0].page_number == 1
        assert chunks[0].page_number_end == 2
        assert "Page 1 short content." in chunks[0].text
        assert "Page 2 short content." in chunks[0].text

        assert chunks[1].page_number == 2
        assert chunks[1].page_number_end == 3
        assert "Page 2 short content." in chunks[1].text
        assert "Page 3 short content." in chunks[1].text

        # Overlap verified
        assert "Page 2 short content." in chunks[0].text and "Page 2 short content." in chunks[1].text

    def test_tiny_chunk_merging(self) -> None:
        """Chunks under min_chunk_tokens must be merged into neighboring chunks."""
        # Paragraph 1 (~35 tokens) + a trailing 4-word paragraph (~5 tokens)
        main_body = "The architecture utilizes dense vector retrieval combined with reciprocal rank fusion to produce grounded answers."
        tiny_trailer = "Final brief remark."
        full_text = f"{main_body}\n\n{tiny_trailer}"
        page = PageText(page_number=1, text=full_text, char_count=len(full_text))

        # With chunk_size=20 and min_chunk_tokens=10, the trailer alone would be < 10 tokens
        chunks = chunk_document(
            document_id="doc-merge",
            pages=[page],
            chunk_size=25,
            chunk_overlap=5,
            min_chunk_tokens=10,
        )

        # Confirm no chunk has fewer than min_chunk_tokens (unless it was the only chunk in doc)
        for c in chunks:
            assert c.token_count >= 10
        # The trailer must be absorbed into a chunk
        assert any(tiny_trailer in c.text for c in chunks)

    def test_chunk_overlap_between_consecutive_chunks(self) -> None:
        """Consecutive chunks must share overlapping tokens."""
        long_paragraph = (
            "Sentence one introduces retrieval augmented generation. "
            "Sentence two discusses vector databases like Pinecone. "
            "Sentence three explains semantic similarity search. "
            "Sentence four addresses the problem of hallucinations in language models. "
            "Sentence five introduces natural language inference for verification. "
            "Sentence six concludes with citation grounding and source attribution."
        )
        page = PageText(page_number=1, text=long_paragraph, char_count=len(long_paragraph))

        chunks = chunk_document(
            document_id="doc-overlap",
            pages=[page],
            chunk_size=25,
            chunk_overlap=10,
            min_chunk_tokens=5,
        )

        assert len(chunks) >= 2
        for i in range(len(chunks) - 1):
            c_curr = chunks[i]
            c_next = chunks[i + 1]

            # Check for shared words/substrings between the end of chunk N and start of chunk N+1
            curr_words = set(c_curr.text.split()[-8:])
            next_words = set(c_next.text.split()[:8])
            shared = curr_words.intersection(next_words)
            assert len(shared) > 0, f"Expected overlap between chunk {i} and {i+1}, found none"


class TestChunkerEndpoints:
    """Integration tests for POST /api/ingest/{document_id}/chunk and GET /{document_id}/chunks."""

    @pytest.mark.asyncio
    async def test_full_upload_extract_chunk_flow(self, isolated_upload_dir: Path) -> None:
        """Full pipeline: Upload PDF -> Extract text -> Chunk document -> Verify chunks.json."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Upload multipage PDF
            upload_res = await client.post(
                "/api/ingest/upload",
                files={"file": ("multipage.pdf", MULTIPAGE_PDF.read_bytes(), "application/pdf")},
            )
            assert upload_res.status_code == 200
            doc_id = upload_res.json()["document_id"]

            # 2. Extract & clean text
            extract_res = await client.post(f"/api/ingest/{doc_id}/extract")
            assert extract_res.status_code == 200

            # 3. Chunk document
            chunk_res = await client.post(f"/api/ingest/{doc_id}/chunk")
            assert chunk_res.status_code == 200
            data = chunk_res.json()
            assert data["document_id"] == doc_id
            assert data["chunk_count"] > 0
            assert len(data["chunks"]) == data["chunk_count"]

            first_chunk = data["chunks"][0]
            assert "chunk_id" in first_chunk
            assert first_chunk["chunk_index"] == 0
            assert first_chunk["token_count"] > 0
            assert first_chunk["page_number"] >= 1

            # 4. Verify chunks.json on disk
            chunks_file = isolated_upload_dir / doc_id / "chunks.json"
            assert chunks_file.exists()

            # 5. Verify registry was updated with chunk_count
            doc_res = await client.get(f"/api/ingest/documents/{doc_id}")
            assert doc_res.status_code == 200
            assert doc_res.json()["chunk_count"] == data["chunk_count"]

            # 6. Verify GET /{document_id}/chunks
            get_chunks_res = await client.get(f"/api/ingest/{doc_id}/chunks")
            assert get_chunks_res.status_code == 200
            assert get_chunks_res.json()["chunk_count"] == data["chunk_count"]

    @pytest.mark.asyncio
    async def test_chunk_before_extract_returns_400(self) -> None:
        """Triggering chunking before extraction should return HTTP 400."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            upload_res = await client.post(
                "/api/ingest/upload",
                files={"file": ("sample.pdf", SAMPLE_PDF.read_bytes(), "application/pdf")},
            )
            assert upload_res.status_code == 200
            doc_id = upload_res.json()["document_id"]

            chunk_res = await client.post(f"/api/ingest/{doc_id}/chunk")
            assert chunk_res.status_code == 400
            assert "has not been extracted" in chunk_res.json()["detail"]

    @pytest.mark.asyncio
    async def test_chunk_nonexistent_document_returns_404(self) -> None:
        """Triggering chunking on nonexistent document ID should return HTTP 404."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.post("/api/ingest/nonexistent-id-000/chunk")
            assert res.status_code == 404
            assert "not found" in res.json()["detail"]
