"""
Document chunking engine with token awareness, recursive splitting, and page-boundary traceability.

Design Notes:
- Token-Aware Splitting: Uses `tiktoken` (cl100k_base encoding) to measure chunk sizes in tokens
  rather than characters. Embedding models (e.g. text-embedding-3-small, BGE) and downstream LLMs
  enforce token context limits, making token-based sizing essential.
- Sane Defaults: Chunk size is 500 tokens with 50 tokens (10%) overlap.
- Split Hierarchy: Recursively splits on paragraph breaks (`\\n\\n`), single newlines (`\\n`),
  sentence boundaries (`. `), word spaces (` `), and characters (`""`).
- Page-Spanning Traceability: Rather than forcing arbitrary cuts at page boundaries (which corrupts
  sentences), chunks are allowed to cross page breaks. The chunker tracks both `page_number` (start)
  and `page_number_end` (end), along with local character offsets into each respective page's text.
- Tiny Chunk Merging: Any chunk resulting in fewer than `MIN_CHUNK_TOKENS` (default 20) is merged
  into its neighboring chunk to avoid low-information fragments.
"""

from typing import Any
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

from app.core.logging import get_logger
from app.models.schemas import Chunk, PageText

logger = get_logger(__name__)

# Standard defaults
DEFAULT_CHUNK_SIZE_TOKENS = 500
DEFAULT_CHUNK_OVERLAP_TOKENS = 50
MIN_CHUNK_TOKENS = 20
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Global tiktoken tokenizer singleton (cl100k_base used across modern OpenAI/LangChain stacks)
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def get_token_count(text: str) -> int:
    """Return the exact number of tokens in text using the cl100k_base tokenizer."""
    return len(_TOKENIZER.encode(text))


class PageSpan:
    """Represents a page's global character interval within the concatenated document text."""

    def __init__(self, page_number: int, start: int, end: int, text: str) -> None:
        self.page_number = page_number
        self.start = start
        self.end = end
        self.text = text


def _build_page_spans(pages: list[PageText]) -> tuple[str, list[PageSpan]]:
    """Concatenate page texts with double newlines while recording exact global offsets."""
    concat_text = ""
    spans: list[PageSpan] = []

    for idx, page in enumerate(pages):
        if idx > 0:
            concat_text += "\n\n"
        start = len(concat_text)
        concat_text += page.text
        end = len(concat_text)
        spans.append(PageSpan(page_number=page.page_number, start=start, end=end, text=page.text))

    return concat_text, spans


def _map_global_to_page_offset(
    global_pos: int, spans: list[PageSpan], is_end: bool = False
) -> tuple[int, int]:
    """Map a global character offset back to a specific page number and local character offset.

    Args:
        global_pos: Character offset in concatenated document text.
        spans: Ordered list of PageSpan objects.
        is_end: If True, indicates mapping the ending boundary of a chunk.

    Returns:
        Tuple of (page_number, local_char_offset).
    """
    if not spans:
        return 1, 0

    # Bounds handling
    if global_pos <= 0:
        return spans[0].page_number, 0
    if global_pos >= spans[-1].end:
        last = spans[-1]
        return last.page_number, len(last.text)

    for i, span in enumerate(spans):
        # Position lies within this page
        if span.start <= global_pos <= span.end:
            local = global_pos - span.start
            return span.page_number, local

        # Position lies in delimiter between this page and the next
        if i + 1 < len(spans) and span.end < global_pos < spans[i + 1].start:
            if is_end:
                return span.page_number, len(span.text)
            return spans[i + 1].page_number, 0

    last = spans[-1]
    return last.page_number, len(last.text)


def chunk_document(
    document_id: str,
    pages: list[PageText],
    chunk_size: int = DEFAULT_CHUNK_SIZE_TOKENS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    min_chunk_tokens: int = MIN_CHUNK_TOKENS,
) -> list[Chunk]:
    """Split a list of cleaned pages into semantically cohesive, token-bounded chunks.

    Args:
        document_id: Parent document identifier.
        pages: Cleaned PageText objects (sorted by page_number).
        chunk_size: Target maximum tokens per chunk (default 500).
        chunk_overlap: Overlapping tokens between consecutive chunks (default 50).
        min_chunk_tokens: Minimum tokens below which chunks are merged into neighbors (default 20).

    Returns:
        List of well-formed Chunk objects with verified page and character offset tracking.
    """
    # Filter out empty pages
    valid_pages = [p for p in pages if p.text.strip()]
    if not valid_pages:
        logger.warning("No text found across pages for document %s.", document_id)
        return []

    concat_text, spans = _build_page_spans(valid_pages)
    if not concat_text.strip():
        return []

    # Configure RecursiveCharacterTextSplitter with tokenizer length function
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=get_token_count,
        separators=DEFAULT_SEPARATORS,
        strip_whitespace=True,
    )

    raw_splits = splitter.split_text(concat_text)
    if not raw_splits:
        return []

    # 1. Locate character boundaries for each split using sequential cursor tracking
    intermediate_spans: list[dict[str, Any]] = []
    cursor = 0

    for raw_text in raw_splits:
        # Find raw_text starting from current cursor
        pos = concat_text.find(raw_text, cursor)
        if pos == -1:
            # Fallback in rare edge case: search from 0
            pos = concat_text.find(raw_text)
            if pos == -1:
                logger.error("Could not locate split text in document: %.40s...", raw_text)
                continue

        end_pos = pos + len(raw_text)
        intermediate_spans.append({
            "text": raw_text,
            "global_start": pos,
            "global_end": end_pos,
            "token_count": get_token_count(raw_text),
        })
        # Advance cursor past start of current chunk to prevent matching identical prefixes
        cursor = pos + 1

    if not intermediate_spans:
        return []

    # 2. Merge tiny chunks (< min_chunk_tokens)
    merged_spans: list[dict[str, Any]] = []
    for item in intermediate_spans:
        if not merged_spans:
            merged_spans.append(item)
            continue

        # If current item is too small, merge it into the previous chunk
        if item["token_count"] < min_chunk_tokens:
            prev = merged_spans[-1]
            new_global_end = max(prev["global_end"], item["global_end"])
            merged_text = concat_text[prev["global_start"]:new_global_end].strip()
            prev["global_end"] = new_global_end
            prev["text"] = merged_text
            prev["token_count"] = get_token_count(merged_text)
            logger.debug(
                "Merged tiny chunk (%d tokens) into preceding chunk. New token count: %d",
                item["token_count"],
                prev["token_count"],
            )
        else:
            merged_spans.append(item)

    # If the first chunk ended up < min_chunk_tokens and there is a subsequent chunk, merge forward
    if len(merged_spans) > 1 and merged_spans[0]["token_count"] < min_chunk_tokens:
        first = merged_spans.pop(0)
        second = merged_spans[0]
        new_global_start = min(first["global_start"], second["global_start"])
        merged_text = concat_text[new_global_start:second["global_end"]].strip()
        second["global_start"] = new_global_start
        second["text"] = merged_text
        second["token_count"] = get_token_count(merged_text)

    # 3. Assemble final Chunk instances with projected page numbers & offsets
    final_chunks: list[Chunk] = []
    for idx, span in enumerate(merged_spans):
        g_start = span["global_start"]
        g_end = span["global_end"]
        chunk_text = span["text"]
        token_count = span["token_count"]

        # Quality warning if chunk exceeds limit significantly (> 10% tolerance)
        if token_count > int(chunk_size * 1.15):
            logger.warning(
                "Chunk %d has %d tokens, exceeding configured target %d tokens.",
                idx,
                token_count,
                chunk_size,
            )

        p_start, c_start = _map_global_to_page_offset(g_start, spans, is_end=False)
        p_end, c_end = _map_global_to_page_offset(g_end, spans, is_end=True)

        # Sanity check: page_number_end must be >= page_number
        if p_end < p_start:
            p_end = p_start

        final_chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                chunk_index=idx,
                text=chunk_text,
                token_count=token_count,
                page_number=p_start,
                page_number_end=p_end,
                char_start=c_start,
                char_end=c_end,
            )
        )

    logger.info(
        "Document %s produced %d chunks across %d pages (avg tokens: %.1f).",
        document_id,
        len(final_chunks),
        len(valid_pages),
        sum(c.token_count for c in final_chunks) / len(final_chunks) if final_chunks else 0,
    )
    return final_chunks
