"""Tests for PDF text extraction, page-number preservation, artifact stripping, and extraction endpoints."""

from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.ingestion.pdf_extractor import (
    PDFExtractionError,
    detect_recurring_headers_footers,
    extract_text_by_page,
)
from app.ingestion.storage import reset_document_registry
from app.main import app

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample.pdf"
PROTECTED_PDF = FIXTURES_DIR / "protected.pdf"
MULTIPAGE_PDF = FIXTURES_DIR / "multipage_with_footer.pdf"
SCANNED_BLANK_PDF = FIXTURES_DIR / "scanned_blank.pdf"


@pytest.fixture(autouse=True)
def isolated_upload_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate uploads and registry file to a temporary directory for each test."""
    upload_dir = tmp_path / "test_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(upload_dir))
    reset_document_registry()
    yield upload_dir
    reset_document_registry()


class TestPdfExtractorUnit:
    """Unit tests for the core extract_text_by_page function and heuristics."""

    def test_extract_valid_sample_pdf(self) -> None:
        """Extracting text from a valid single-page PDF returns 1 page with clean text."""
        pages = extract_text_by_page(SAMPLE_PDF)
        assert len(pages) == 1
        page = pages[0]
        assert page.page_number == 1
        assert "Self-Correcting RAG" in page.text
        assert page.char_count == len(page.text)
        assert page.char_count > 0

    def test_extract_multipage_with_footer_heuristic(self) -> None:
        """Repeated running footers appearing on >50% of pages must be stripped."""
        pages = extract_text_by_page(MULTIPAGE_PDF)
        assert len(pages) == 3

        # Confirm content from each page is retained
        assert "Chapter 1: Scaffolding RAG Systems" in pages[0].text
        assert "Chapter 2: Vector Search" in pages[1].text
        assert "Chapter 3: Verification & Hallucination Guardrails" in pages[2].text

        # Confirm running footer 'CONFIDENTIAL - INTERNAL ONLY' was stripped from all pages
        for page in pages:
            assert "CONFIDENTIAL - INTERNAL ONLY" not in page.text
            assert page.page_number in (1, 2, 3)

    def test_header_footer_detector_heuristic(self) -> None:
        """Unit test for detect_recurring_headers_footers frequency logic."""
        raw_pages = [
            ["Running Header", "Content A", "Page 1 Footer"],
            ["Running Header", "Content B", "Unique Footer"],
            ["Running Header", "Content C", "Page 3 Footer"],
        ]
        recurring = detect_recurring_headers_footers(raw_pages)
        assert "Running Header" in recurring
        assert "Content A" not in recurring
        assert "Page 1 Footer" not in recurring

    def test_extract_protected_pdf_raises_graceful_error(self) -> None:
        """Password-protected PDFs must raise PDFExtractionError with specific message."""
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_by_page(PROTECTED_PDF)
        assert "password-protected PDF not supported" in str(exc_info.value)

    def test_extract_scanned_zero_text_pdf_raises_graceful_error(self) -> None:
        """Scanned or image-only PDFs with 0 extractable characters must fail gracefully."""
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_by_page(SCANNED_BLANK_PDF)
        assert "no extractable text - scanned PDF not supported" in str(exc_info.value)

    def test_extract_corrupted_pdf_raises_graceful_error(self, tmp_path: Path) -> None:
        """Corrupted or truncated PDF bytes must raise PDFExtractionError, not crash."""
        corrupted_file = tmp_path / "corrupted.pdf"
        # Truncated PDF header with garbage body
        corrupted_file.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nEOF_TRUNCATED")

        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_by_page(corrupted_file)
        assert "corrupted or unreadable PDF file" in str(exc_info.value)


class TestPdfExtractionEndpoints:
    """Integration tests for the /api/ingest/{document_id}/extract endpoint."""

    @pytest.mark.asyncio
    async def test_full_upload_and_extract_flow(self, isolated_upload_dir: Path) -> None:
        """Full happy path: Upload valid PDF -> trigger extract -> verify ready status and extracted.json."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Upload valid PDF
            upload_res = await client.post(
                "/api/ingest/upload",
                files={"file": ("sample.pdf", SAMPLE_PDF.read_bytes(), "application/pdf")},
            )
            assert upload_res.status_code == 200
            doc_id = upload_res.json()["document_id"]

            # 2. Trigger text extraction
            extract_res = await client.post(f"/api/ingest/{doc_id}/extract")
            assert extract_res.status_code == 200
            extract_data = extract_res.json()

            assert extract_data["document_id"] == doc_id
            assert extract_data["status"] == "ready"
            assert extract_data["page_count"] == 1
            assert extract_data["total_char_count"] > 0
            assert len(extract_data["pages"]) == 1
            assert extract_data["pages"][0]["page_number"] == 1
            assert "Self-Correcting RAG" in extract_data["pages"][0]["text"]

            # 3. Verify extracted.json is created on disk
            extracted_json_path = isolated_upload_dir / doc_id / "extracted.json"
            assert extracted_json_path.exists()

            # 4. Check registry metadata was updated
            doc_meta_res = await client.get(f"/api/ingest/documents/{doc_id}")
            assert doc_meta_res.status_code == 200
            meta = doc_meta_res.json()
            assert meta["status"] == "ready"
            assert meta["page_count"] == 1
            assert meta["total_char_count"] > 0
            assert meta["failure_reason"] is None

            # 5. Verify GET /{document_id}/pages endpoint
            pages_res = await client.get(f"/api/ingest/{doc_id}/pages")
            assert pages_res.status_code == 200
            pages_list = pages_res.json()
            assert len(pages_list) == 1
            assert pages_list[0]["page_number"] == 1

    @pytest.mark.asyncio
    async def test_extract_password_protected_flow(self) -> None:
        """Uploading and extracting a password-protected PDF marks status failed with clear reason."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            upload_res = await client.post(
                "/api/ingest/upload",
                files={"file": ("protected.pdf", PROTECTED_PDF.read_bytes(), "application/pdf")},
            )
            assert upload_res.status_code == 200
            doc_id = upload_res.json()["document_id"]

            extract_res = await client.post(f"/api/ingest/{doc_id}/extract")
            assert extract_res.status_code == 200
            data = extract_res.json()
            assert data["status"] == "failed"
            assert "password-protected PDF not supported" in data["failure_reason"]

            # Verify registry status also reflects failure
            doc_res = await client.get(f"/api/ingest/documents/{doc_id}")
            assert doc_res.json()["status"] == "failed"
            assert "password-protected PDF not supported" in doc_res.json()["failure_reason"]

    @pytest.mark.asyncio
    async def test_extract_corrupted_pdf_flow(self, tmp_path: Path) -> None:
        """Extracting a corrupted PDF file handles exception and marks document failed."""
        corrupted_bytes = b"%PDF-1.4\ncorrupted_body_without_xref_or_trailer"
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            upload_res = await client.post(
                "/api/ingest/upload",
                files={"file": ("corrupted.pdf", corrupted_bytes, "application/pdf")},
            )
            assert upload_res.status_code == 200
            doc_id = upload_res.json()["document_id"]

            extract_res = await client.post(f"/api/ingest/{doc_id}/extract")
            assert extract_res.status_code == 200
            data = extract_res.json()
            assert data["status"] == "failed"
            assert "corrupted or unreadable PDF file" in data["failure_reason"]

    @pytest.mark.asyncio
    async def test_extract_nonexistent_document_returns_404(self) -> None:
        """Triggering extraction on a missing document ID returns HTTP 404."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            extract_res = await client.post("/api/ingest/nonexistent-id-000/extract")
            assert extract_res.status_code == 404
            assert "not found" in extract_res.json()["detail"]
