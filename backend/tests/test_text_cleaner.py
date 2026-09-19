"""Tests for text cleaning, hyphenation rejoining, artifact removal, and diagnostic reporting."""

import logging
from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.ingestion.storage import reset_document_registry
from app.ingestion.text_cleaner import (
    clean_document_pages,
    clean_page_text,
    get_cleaning_report,
)
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


class TestTextCleanerUnit:
    """Unit tests for individual text normalization functions and edge cases."""

    def test_whitespace_normalization(self) -> None:
        """Collapse multiple spaces/tabs and excessive 3+ newlines into 2."""
        messy_text = "Word1   \t   Word2.\n\n\n\n\nParagraph two   here.\t\tEnd."
        input_page = PageText(page_number=1, text=messy_text, char_count=len(messy_text))

        cleaned_page = clean_page_text(input_page)
        expected = "Word1 Word2.\n\nParagraph two here. End."
        assert cleaned_page.text == expected
        assert cleaned_page.char_count == len(expected)
        assert cleaned_page.page_number == 1

    def test_hyphenated_word_rejoining(self) -> None:
        """Broken line-end hyphenated words must be rejoined without hyphens or line breaks."""
        text = (
            "We present an informa-\ntion retrieval and self-reflec-\ntive "
            "methodology for multi-\n  national organizations."
        )
        input_page = PageText(page_number=1, text=text, char_count=len(text))
        cleaned = clean_page_text(input_page)

        assert "information" in cleaned.text
        assert "self-reflective" in cleaned.text
        assert "multinational" in cleaned.text
        assert "informa-\ntion" not in cleaned.text

    def test_isolated_page_number_removal(self) -> None:
        """Isolated page numbers on their own lines must be removed without touching sentence numbers."""
        text = (
            "Section 1. Background\n"
            "In 2023, 42 employees participated in the experiment.\n"
            "42\n"
            "The model achieved 98.5% accuracy.\n"
            "- 15 -\n"
            "— 99 —\n"
            "Page 7\n"
            "Conclusion follows."
        )
        input_page = PageText(page_number=1, text=text, char_count=len(text))
        cleaned = clean_page_text(input_page)

        # Isolated page numbers stripped
        lines = cleaned.text.splitlines()
        assert "42" not in lines
        assert "- 15 -" not in lines
        assert "— 99 —" not in lines
        assert "Page 7" not in lines

        # Numbers inside sentences and section headers preserved
        assert "Section 1. Background" in cleaned.text
        assert "In 2023, 42 employees participated in the experiment." in cleaned.text
        assert "The model achieved 98.5% accuracy." in cleaned.text
        assert "Conclusion follows." in cleaned.text

    def test_unicode_nfkc_normalization(self) -> None:
        """Ligatures (ﬁ, ﬂ) and fullwidth/compatibility characters are normalized to standard ASCII."""
        # \uFB01 is 'fi', \uFB02 is 'fl'
        text_with_ligatures = "The proﬁle and the ﬂow of the model."
        input_page = PageText(page_number=1, text=text_with_ligatures, char_count=len(text_with_ligatures))
        cleaned = clean_page_text(input_page)

        assert "profile" in cleaned.text
        assert "flow" in cleaned.text
        assert "ﬁ" not in cleaned.text
        assert "ﬂ" not in cleaned.text

    def test_control_characters_removal(self) -> None:
        """Control and non-printable characters are stripped while preserving tabs and newlines."""
        text_with_control = "Line 1\x00\x08\x0b.\nLine 2\x0c with \x1ftab\there."
        input_page = PageText(page_number=1, text=text_with_control, char_count=len(text_with_control))
        cleaned = clean_page_text(input_page)

        assert cleaned.text == "Line 1.\nLine 2 with tab here."

    def test_reduction_warning_triggered_on_overcleaned_page(self, caplog: pytest.LogCaptureFixture) -> None:
        """If cleaning removes more than 40% of characters, a warning must be logged and flagged."""
        # Create a page that is mostly whitespace, control chars, and standalone numbers
        raw_text = "Important content.\n" + ("42\n" * 50) + ("\x00" * 200)
        before = PageText(page_number=3, text=raw_text, char_count=len(raw_text))

        with caplog.at_level(logging.WARNING):
            after = clean_page_text(before)
            report = get_cleaning_report(before, after)

        assert report["flagged_for_review"] is True
        assert report["percent_reduction"] > 40.0
        assert "character count reduced by" in caplog.text
        assert "Flagged for review" in caplog.text

    def test_full_pipeline_on_realistic_page(self) -> None:
        """Full pipeline execution on a complex page with mixed artifacts."""
        raw_text = (
            "   1. Introduction to Retrieval\n\n\n\n"
            "Modern large language models often struggle with fac-\ntual hallucinations.\n"
            "In our evaluation of 150 benchmark queries, accuracy was 94.2%.\n"
            "\x00\x08"
            "42\n"
            "A second paragraph detailing the veriﬁcation pipeline.\n"
        )
        page = PageText(page_number=1, text=raw_text, char_count=len(raw_text))
        cleaned = clean_page_text(page)

        expected = (
            "1. Introduction to Retrieval\n\n"
            "Modern large language models often struggle with factual hallucinations.\n"
            "In our evaluation of 150 benchmark queries, accuracy was 94.2%.\n"
            "A second paragraph detailing the verification pipeline."
        )
        assert cleaned.text == expected


class TestTextCleanerIntegration:
    """Integration tests verifying document-level batch cleaning and API endpoints."""

    def test_clean_document_pages_batch(self) -> None:
        """Batch cleaning multi-page document strips recurring headers and footers."""
        p1 = PageText(page_number=1, text="HEADER\nContent P1\nFOOTER", char_count=23)
        p2 = PageText(page_number=2, text="HEADER\nContent P2\nFOOTER", char_count=23)
        p3 = PageText(page_number=3, text="HEADER\nContent P3\nFOOTER", char_count=23)

        cleaned_pages, reports = clean_document_pages([p1, p2, p3])
        assert len(cleaned_pages) == 3
        assert len(reports) == 3

        for cp in cleaned_pages:
            assert "HEADER" not in cp.text
            assert "FOOTER" not in cp.text
            assert "Content P" in cp.text

    @pytest.mark.asyncio
    async def test_extract_endpoint_generates_raw_and_clean_files(
        self, isolated_upload_dir: Path
    ) -> None:
        """Endpoint saves extracted_raw.json, extracted_clean.json, and canonical extracted.json."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Upload valid multipage PDF
            upload_res = await client.post(
                "/api/ingest/upload",
                files={"file": ("multipage.pdf", MULTIPAGE_PDF.read_bytes(), "application/pdf")},
            )
            assert upload_res.status_code == 200
            doc_id = upload_res.json()["document_id"]

            # 2. Extract & Clean
            extract_res = await client.post(f"/api/ingest/{doc_id}/extract")
            assert extract_res.status_code == 200
            data = extract_res.json()
            assert data["status"] == "ready"
            assert "cleaning_reports" in data
            assert len(data["cleaning_reports"]) == 3

            # 3. Check physical files on disk
            raw_file = isolated_upload_dir / doc_id / "extracted_raw.json"
            clean_file = isolated_upload_dir / doc_id / "extracted_clean.json"
            main_file = isolated_upload_dir / doc_id / "extracted.json"

            assert raw_file.exists()
            assert clean_file.exists()
            assert main_file.exists()

            # Raw should contain the footer, cleaned should have it stripped
            raw_content = raw_file.read_text(encoding="utf-8")
            clean_content = clean_file.read_text(encoding="utf-8")
            assert "CONFIDENTIAL - INTERNAL ONLY" in raw_content
            assert "CONFIDENTIAL - INTERNAL ONLY" not in clean_content
