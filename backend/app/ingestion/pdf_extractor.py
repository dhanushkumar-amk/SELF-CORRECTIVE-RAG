"""
PDF text extraction engine with page-number preservation and artifact stripping.

Design Notes:
- Library Choice: Uses `pypdf` (pure-Python, lightweight, fast, no external C/poppler runtime).
- Page Attribution: Extracts text per-page into `PageText` objects (page_number, text, char_count).
  Preserving page numbers is required for NLI verification citations (Phase 30+).
- Layout & Reading Order: `pypdf` extracts text blocks in logical PDF reading order.
- Artifact Removal: Strips running headers/footers appearing on >50% of pages and cleans
  excessive intra-page whitespace.
- Graceful Failures: Explicitly handles corrupted PDFs, password-protected PDFs, and
  scanned/image-only PDFs (OCR is out of scope).
"""

from collections import defaultdict
from pathlib import Path
import re

import pypdf
from pypdf.errors import PdfReadError, PdfStreamError

from app.core.logging import get_logger
from app.models.schemas import PageText

logger = get_logger(__name__)


class PDFExtractionError(Exception):
    """Raised when PDF extraction cannot complete due to document-level issues."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def detect_recurring_headers_footers(pages_lines: list[list[str]]) -> set[str]:
    """Detect recurring running headers and footers across multiple pages.

    Heuristic:
    A short line (<= 120 characters) appearing on strictly more than 50% of pages
    in a multi-page document is treated as a running header or footer.
    """
    total_pages = len(pages_lines)
    if total_pages <= 1:
        return set()

    line_page_presence: dict[str, set[int]] = defaultdict(set)
    for page_idx, lines in enumerate(pages_lines):
        for line in lines:
            normalized = line.strip()
            # Only consider candidate header/footer lines (non-empty and concise)
            if normalized and len(normalized) <= 120:
                line_page_presence[normalized].add(page_idx)

    threshold = total_pages * 0.5
    recurring = {
        line
        for line, pages in line_page_presence.items()
        if len(pages) > threshold
    }

    if recurring:
        logger.debug(
            "Identified %d recurring header/footer artifact(s): %s",
            len(recurring),
            recurring,
        )
    return recurring


def clean_page_lines(
    lines: list[str],
    recurring_artifacts: set[str],
) -> str:
    """Filter out recurring header/footer lines and normalize whitespace."""
    filtered_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped in recurring_artifacts:
            continue
        filtered_lines.append(stripped)

    # Join lines and collapse excessive whitespace
    joined = "\n".join(filtered_lines)
    # Collapse 3 or more consecutive newlines into double newlines
    cleaned = re.sub(r"\n{3,}", "\n\n", joined).strip()
    return cleaned


def extract_text_by_page(pdf_path: str | Path) -> list[PageText]:
    """Extract clean, structured text from each page of a stored PDF.

    Args:
        pdf_path: Path to the target PDF file on disk.

    Returns:
        List of PageText objects containing 1-indexed page number, text, and char count.

    Raises:
        PDFExtractionError: If PDF is encrypted, corrupted, empty, or has no extractable text.
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
            # Attempt blank password decrypt (common for permission-restricted PDFs)
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
    pages_raw_lines: list[list[str]] = []
    for page_idx in range(total_pages):
        try:
            page = reader.pages[page_idx]
            text = page.extract_text() or ""
            # Split into individual lines
            lines = [line.strip() for line in text.splitlines()]
            pages_raw_lines.append(lines)
        except Exception as exc:
            logger.warning(
                "Error extracting page %d from %s: %s",
                page_idx + 1,
                target.name,
                exc,
            )
            # If reading an individual page fails severely, consider file corrupted
            raise PDFExtractionError("corrupted or unreadable PDF file") from exc

    # 3. Detect and Strip Recurring Headers/Footers across pages
    recurring_artifacts = detect_recurring_headers_footers(pages_raw_lines)

    # 4. Assemble Cleaned PageText List
    extracted_pages: list[PageText] = []
    for page_idx, lines in enumerate(pages_raw_lines):
        cleaned_text = clean_page_lines(lines, recurring_artifacts)
        extracted_pages.append(
            PageText(
                page_number=page_idx + 1,
                text=cleaned_text,
                char_count=len(cleaned_text),
            )
        )

    # 5. Check for Scanned / Image-Only PDF (Zero Extractable Text Layer)
    total_chars = sum(p.char_count for p in extracted_pages)
    if total_chars == 0:
        logger.warning(
            "PDF %s has 0 extractable text characters across %d pages (likely scanned).",
            target.name,
            total_pages,
        )
        raise PDFExtractionError("no extractable text - scanned PDF not supported")

    logger.info(
        "Successfully extracted %d pages (%d total chars) from '%s'",
        len(extracted_pages),
        total_chars,
        target.name,
    )
    return extracted_pages
