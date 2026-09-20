"""
PDF text extraction engine with page-number preservation and robust error handling.

Design Notes:
- Library Choice: Uses `pypdf` (pure-Python, lightweight, fast, no external C/poppler runtime).
- Page Attribution: Extracts text per-page into `PageText` objects (page_number, text, char_count).
  Preserving page numbers is required for NLI verification citations (Phase 30+).
- Layout & Reading Order: `pypdf` extracts text blocks in logical PDF reading order.
- Clean Separation: Raw PDF stream extraction is decoupled from text normalization,
  which is centralized in `app.ingestion.text_cleaner`.
- Graceful Failures: Explicitly handles corrupted PDFs, password-protected PDFs, and
  scanned/image-only PDFs (OCR is out of scope).
"""

from pathlib import Path
import pypdf
from pypdf.errors import PdfReadError, PdfStreamError

from app.core.logging import get_logger
from app.ingestion.text_cleaner import (
    clean_document_pages,
    detect_recurring_headers_footers,
)
from app.models.schemas import PageText

from app.ingestion.exceptions import PermanentIngestionError

logger = get_logger(__name__)

# Re-export detect_recurring_headers_footers for backward compatibility
__all__ = [
    "PDFExtractionError",
    "extract_raw_pages",
    "extract_text_by_page",
    "detect_recurring_headers_footers",
]


class PDFExtractionError(PermanentIngestionError):
    """Raised when PDF extraction cannot complete due to document-level issues."""

    def __init__(self, message: str) -> None:
        super().__init__(message, stage="extracting")
        self.message = message


def extract_raw_pages(pdf_path: str | Path) -> list[PageText]:
    """Extract raw, uncleaned text from each page of a stored PDF.

    Args:
        pdf_path: Path to the target PDF file on disk.

    Returns:
        List of raw PageText objects containing 1-indexed page number, text, and char count.

    Raises:
        PDFExtractionError: If PDF is encrypted, corrupted, empty, or has no extractable text layer.
    """
    target = Path(pdf_path)
    if not target.exists():
        logger.error("Extraction failed: File not found at %s", target)
        raise PDFExtractionError("PDF file does not exist on disk")

    try:
        reader = pypdf.PdfReader(str(target))
    except (PdfReadError, PdfStreamError, ValueError, OSError, Exception) as exc:
        logger.warning("Corrupted or unreadable PDF (%s): %s", target.name, exc)
        raise PDFExtractionError("corrupted or unreadable PDF file") from exc

    # 1. Handle Password Protection / Encryption
    if reader.is_encrypted:
        try:
            decrypt_result = reader.decrypt("")
            if decrypt_result == 0 or reader.is_encrypted:
                logger.warning("PDF %s is password protected.", target.name)
                raise PDFExtractionError("password-protected PDF not supported")
        except PDFExtractionError:
            raise
        except Exception as exc:
            logger.warning("Decryption failed for %s: %s", target.name, exc)
            raise PDFExtractionError("password-protected PDF not supported") from exc

    total_pages = len(reader.pages)
    if total_pages == 0:
        logger.warning("PDF %s contains 0 pages.", target.name)
        raise PDFExtractionError("no extractable text - scanned PDF not supported")

    # 2. Extract Raw Text Per Page
    raw_pages: list[PageText] = []
    for page_idx in range(total_pages):
        try:
            page = reader.pages[page_idx]
            text = page.extract_text() or ""
            raw_pages.append(
                PageText(
                    page_number=page_idx + 1,
                    text=text,
                    char_count=len(text),
                )
            )
        except Exception as exc:
            logger.warning(
                "Error extracting page %d from %s: %s",
                page_idx + 1,
                target.name,
                exc,
            )
            raise PDFExtractionError("corrupted or unreadable PDF file") from exc

    # 3. Check for Scanned / Image-Only PDF (Zero Extractable Text Layer)
    total_chars = sum(len(p.text.strip()) for p in raw_pages)
    if total_chars == 0:
        logger.warning(
            "PDF %s has 0 extractable text characters across %d pages (likely scanned).",
            target.name,
            total_pages,
        )
        raise PDFExtractionError("no extractable text - scanned PDF not supported")

    logger.info(
        "Successfully extracted raw text from %d pages (%d total chars) for '%s'",
        len(raw_pages),
        total_chars,
        target.name,
    )
    return raw_pages


def extract_text_by_page(pdf_path: str | Path) -> list[PageText]:
    """Extract and clean text from each page of a stored PDF.

    Executes raw extraction followed by the full text cleaning and normalization pipeline.
    """
    raw_pages = extract_raw_pages(pdf_path)
    clean_pages, _ = clean_document_pages(raw_pages)
    return clean_pages
