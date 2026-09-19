"""
Text cleaning and normalization pipeline for extracted PDF page text.

Design Philosophy:
- Citation Traceability: Text cleaning removes extraction noise (broken hyphenations,
  ligatures, isolated page numbers, excess spacing) while strictly preserving sentence
  structures, numbers, section headings, and content required for accurate source grounding.
- Single Source of Truth: Consolidates all text normalization and header/footer removal
  into this module.
- Quality Safeguard: Tracks character reduction per page and logs a loud warning if cleaning
  removes > 40% of characters (indicating potential content destruction).
"""

from collections import defaultdict
import re
import unicodedata
from typing import Any

from app.core.logging import get_logger
from app.models.schemas import PageText

logger = get_logger(__name__)

# Maximum acceptable character reduction ratio before flagging a page for manual review
MAX_REDUCTION_THRESHOLD_PERCENT = 40.0

# Regex matching isolated page-number lines (e.g. "42", "- 42 -", "— 12 —", "Page 5")
ISOLATED_PAGE_NUMBER_REGEX = re.compile(
    r"^\s*(?:page\s+)?[-–—]?\s*\d+\s*[-–—]?\s*$", re.IGNORECASE
)

# Regex matching non-printable/control characters (preserves standard \n and \t)
CONTROL_CHAR_REGEX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Regex matching hyphenated words split across a newline
# E.g. "informa-\ntion" -> "information", "multi-\n  national" -> "multinational"
HYPHENATED_LINEBREAK_REGEX = re.compile(
    r"(\b[A-Za-z]+)-\n[ \t]*([a-z]+)\b"
)


def detect_recurring_headers_footers(pages_lines: list[list[str]]) -> set[str]:
    """Identify running headers and footers appearing on >50% of pages in a multi-page document.

    Heuristic:
    A concise line (<= 120 chars) appearing on strictly more than half of the pages
    is classified as a recurring running header or footer artifact.
    """
    total_pages = len(pages_lines)
    if total_pages <= 1:
        return set()

    line_page_presence: dict[str, set[int]] = defaultdict(set)
    for page_idx, lines in enumerate(pages_lines):
        for line in lines:
            normalized = line.strip()
            if normalized and len(normalized) <= 120:
                line_page_presence[normalized].add(page_idx)

    threshold = total_pages * 0.5
    recurring = {
        line
        for line, pages in line_page_presence.items()
        if len(pages) > threshold
    }
    return recurring


def clean_page_text(
    page_text: PageText,
    recurring_headers_footers: set[str] | None = None,
) -> PageText:
    """Clean and normalize a single page's text while preserving citation integrity.

    Operations executed in deliberate order:
    1. Unicode Normalization (NFKC): Standardize ligatures, special spaces, and symbols
       so downstream regex can reliably match word and line boundaries.
    2. Control Character Removal: Strip invisible/corrupt ASCII and C1 control characters
       while preserving essential layout delimiters (\\n, \\t).
    3. Hyphenated Word Rejoining: Mend words split across line breaks ("informa-\\ntion" -> "information").
       Must run before newline collapsing so the line-break hyphen delimiter is intact.
    4. Isolated Page Number Removal: Strip standalone page numbers on their own line.
       Does NOT touch numbers embedded in sentences, statistics, or section headers.
    5. Recurring Header/Footer Removal: Strip repeated running headers/footers identified
       across the document.
    6. Whitespace Normalization: Collapse multi-spaces/tabs to one space, trim line whitespace,
       and collapse 3+ consecutive newlines down to 2 (\\n\\n) to preserve paragraph breaks.

    Args:
        page_text: Input PageText object.
        recurring_headers_footers: Optional set of recurring lines to remove.

    Returns:
        New PageText instance with cleaned text and updated char_count.
    """
    raw = page_text.text
    if not raw:
        return PageText(
            page_number=page_text.page_number, text="", char_count=0
        )

    # 1. Unicode NFKC Normalization
    text = unicodedata.normalize("NFKC", raw)

    # 2. Control Character Removal (keep \n and \t)
    text = CONTROL_CHAR_REGEX.sub("", text)

    # 3. Fix Hyphenated Line-Break Words
    text = HYPHENATED_LINEBREAK_REGEX.sub(r"\1\2", text)

    # 4 & 5. Filter Isolated Page Numbers and Recurring Headers/Footers Line-by-Line
    recurring = recurring_headers_footers or set()
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped_line = line.strip()

        # Check for isolated page number line (e.g. "42", "- 12 -", "Page 3")
        if ISOLATED_PAGE_NUMBER_REGEX.fullmatch(stripped_line):
            continue

        # Check for recurring header/footer artifact
        if stripped_line in recurring:
            continue

        cleaned_lines.append(line)

    rejoined = "\n".join(cleaned_lines)

    # 6. Whitespace Normalization
    # Collapse multiple horizontal spaces and tabs into a single space
    rejoined = re.sub(r"[ \t]+", " ", rejoined)

    # Strip trailing whitespace on each line
    trimmed_lines = [l.rstrip() for l in rejoined.splitlines()]
    rejoined = "\n".join(trimmed_lines)

    # Collapse 3 or more consecutive newlines into double newlines (\n\n) to preserve paragraphs
    cleaned = re.sub(r"\n{3,}", "\n\n", rejoined).strip()

    return PageText(
        page_number=page_text.page_number,
        text=cleaned,
        char_count=len(cleaned),
    )


def get_cleaning_report(before: PageText, after: PageText) -> dict[str, Any]:
    """Generate diagnostic metrics and sanity checks comparing before/after cleaning.

    Flags pages with >40% character reduction for manual review.
    """
    chars_before = before.char_count
    chars_after = after.char_count
    chars_removed = chars_before - chars_after
    percent_reduction = (
        (chars_removed / chars_before * 100.0) if chars_before > 0 else 0.0
    )
    flagged = percent_reduction > MAX_REDUCTION_THRESHOLD_PERCENT

    if flagged:
        logger.warning(
            "Page %d character count reduced by %.1f%% (> %.0f%%) during cleaning "
            "(%d -> %d chars). Flagged for review.",
            before.page_number,
            percent_reduction,
            MAX_REDUCTION_THRESHOLD_PERCENT,
            chars_before,
            chars_after,
        )

    return {
        "page_number": before.page_number,
        "chars_before": chars_before,
        "chars_after": chars_after,
        "chars_removed": chars_removed,
        "percent_reduction": round(percent_reduction, 2),
        "flagged_for_review": flagged,
        "operations_applied": [
            "unicode_nfkc_normalization",
            "control_character_removal",
            "hyphenated_linebreak_rejoining",
            "isolated_page_number_removal",
            "recurring_header_footer_removal",
            "whitespace_normalization",
        ],
    }


def clean_document_pages(
    pages: list[PageText],
) -> tuple[list[PageText], list[dict[str, Any]]]:
    """Clean all pages of a document and return cleaned PageText list with diagnostic reports.

    Args:
        pages: List of raw extracted PageText objects.

    Returns:
        Tuple of (cleaned_pages, cleaning_reports).
    """
    # Extract line representations to detect multi-page recurring headers/footers
    pages_lines = [[line.strip() for line in p.text.splitlines()] for p in pages]
    recurring = detect_recurring_headers_footers(pages_lines)

    cleaned_pages: list[PageText] = []
    reports: list[dict[str, Any]] = []

    for raw_page in pages:
        clean_page = clean_page_text(raw_page, recurring_headers_footers=recurring)
        report = get_cleaning_report(raw_page, clean_page)
        cleaned_pages.append(clean_page)
        reports.append(report)

    return cleaned_pages, reports
