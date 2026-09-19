"""
Pinecone chunk metadata schema formatting, validation, and vector ID management.

This module formalizes chunk metadata strictly against Pinecone's vector storage constraints:
1. Allowed metadata value types: str, int, float, bool, and list[str].
2. Strictly prohibited: null/None values, nested dicts/objects, and non-string lists.
3. Size limits: 40 KB (40,960 bytes) maximum metadata payload per vector.

Traceability & Attribution:
- source_text is retained directly in Pinecone metadata to power downstream NLI claim verification
  (Phase 34) and context generation without needing an external document lookup database.
- page_number and page_number_end provide honest citations across single or page-spanning chunks.
- filename and document_title provide human-readable citation attribution in the UI.
- char_start and char_end are explicitly EXCLUDED from Pinecone metadata to preserve vector payload
  headroom, as exact character offsets remain permanently traceable on disk via
  uploads/{document_id}/chunks.json keyed by document_id and chunk_id.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.models.schemas import Chunk, ChunkMetadata

logger = get_logger(__name__)

# Pinecone enforces a hard limit of 40 KB (40,960 bytes) per vector metadata payload.
PINECONE_MAX_METADATA_BYTES: int = 40960
TRUNCATION_MARKER: str = " ...[truncated]"
VECTOR_ID_SEPARATOR: str = "::"

__all__ = [
    "PINECONE_MAX_METADATA_BYTES",
    "TRUNCATION_MARKER",
    "VECTOR_ID_SEPARATOR",
    "PineconeMetadataValidationError",
    "create_vector_id",
    "parse_vector_id",
    "estimate_metadata_size_bytes",
    "validate_pinecone_metadata",
    "truncate_metadata_source_text",
    "to_pinecone_metadata",
]


class PineconeMetadataValidationError(ValueError):
    """Raised when metadata payload violates Pinecone vector storage constraints."""

    pass


def create_vector_id(document_id: str, chunk_id: str) -> str:
    """Generate a globally unique, document-scoped Pinecone vector ID.

    Format: "{document_id}::{chunk_id}"
    Example: "d9e1...::c3a2..."

    Benefits:
    - Provides immediate visual and programmatic document scoping directly from vector query results.
    - Enables prefix-based vector deletion and inspection without metadata lookups.
    - Preserves UUID4 collision-free guarantees while remaining deterministic and parseable.
    """
    if not document_id or not chunk_id:
        raise ValueError("Both document_id and chunk_id must be non-empty strings.")
    return f"{document_id}{VECTOR_ID_SEPARATOR}{chunk_id}"


def parse_vector_id(vector_id: str) -> tuple[str, str]:
    """Parse a document-scoped vector ID back into (document_id, chunk_id).

    Raises:
        ValueError: If the vector_id does not contain the expected separator.
    """
    if VECTOR_ID_SEPARATOR not in vector_id:
        raise ValueError(
            f"Invalid vector ID '{vector_id}': missing separator '{VECTOR_ID_SEPARATOR}'"
        )
    parts = vector_id.split(VECTOR_ID_SEPARATOR, 1)
    return parts[0], parts[1]


def estimate_metadata_size_bytes(metadata: dict[str, Any]) -> int:
    """Calculate the exact UTF-8 serialized byte size of a metadata dictionary."""
    try:
        serialized = json.dumps(metadata, ensure_ascii=False)
        return len(serialized.encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise PineconeMetadataValidationError(
            f"Metadata cannot be serialized to JSON: {exc}"
        ) from exc


def validate_pinecone_metadata(
    metadata: dict[str, Any],
    max_bytes: int = PINECONE_MAX_METADATA_BYTES,
) -> list[str]:
    """Validate a metadata dictionary against Pinecone constraints.

    Returns:
        list[str]: A list of validation error descriptions. Empty if valid.
    """
    errors: list[str] = []

    if not isinstance(metadata, dict):
        return ["Metadata must be a dictionary."]

    for key, value in metadata.items():
        if not isinstance(key, str):
            errors.append(
                f"Metadata key {key!r} is not a string (type: {type(key).__name__})."
            )

        # Pinecone strictly disallows null/None values - keys must be omitted
        if value is None:
            errors.append(
                f"Key '{key}' has None/null value. Pinecone requires null fields to be omitted."
            )
            continue

        # Check for nested dictionaries
        if isinstance(value, dict):
            errors.append(
                f"Key '{key}' contains a nested dictionary. Pinecone metadata must be flat."
            )
            continue

        # Booleans (must check before int because bool subclasses int in Python)
        if isinstance(value, bool):
            continue

        # Standard numbers and strings
        if isinstance(value, (int, float, str)):
            continue

        # List of strings
        if isinstance(value, list):
            non_strings = [item for item in value if not isinstance(item, str)]
            if non_strings:
                errors.append(
                    f"Key '{key}' is a list containing non-string items: {non_strings[:3]}. "
                    f"Pinecone only permits list[str]."
                )
            continue

        # Any other type (e.g. set, tuple, custom object)
        errors.append(
            f"Key '{key}' has unsupported type '{type(value).__name__}'. "
            f"Allowed types: str, int, float, bool, list[str]."
        )

    # Size check
    try:
        size_bytes = estimate_metadata_size_bytes(metadata)
        if size_bytes > max_bytes:
            errors.append(
                f"Total metadata size ({size_bytes} bytes) exceeds Pinecone limit "
                f"of {max_bytes} bytes."
            )
    except PineconeMetadataValidationError as err:
        errors.append(str(err))

    return errors


def truncate_metadata_source_text(
    metadata: dict[str, Any],
    max_bytes: int = PINECONE_MAX_METADATA_BYTES,
    suffix: str = TRUNCATION_MARKER,
) -> dict[str, Any]:
    """Truncate the source_text in a metadata dict so the payload fits within max_bytes.

    Maintains valid UTF-8 encoding and appends a clear truncation marker.
    """
    result = dict(metadata)
    current_size = estimate_metadata_size_bytes(result)
    if current_size <= max_bytes:
        return result

    chunk_id = result.get("chunk_id", "unknown")
    source_text = str(result.get("source_text", ""))
    if not source_text:
        return result

    logger.warning(
        "Chunk %s metadata exceeded Pinecone size limit (%d > %d bytes). Truncating source_text.",
        chunk_id,
        current_size,
        max_bytes,
    )

    # Compute target byte allowance for source_text
    suffix_bytes = len(suffix.encode("utf-8"))
    excess_bytes = current_size - max_bytes + suffix_bytes + 64  # 64-byte safety margin
    text_bytes = source_text.encode("utf-8")

    if excess_bytes >= len(text_bytes):
        # Edge case: non-text metadata dominates; reduce text to minimum
        result["source_text"] = suffix.strip()
    else:
        truncated_bytes = text_bytes[: len(text_bytes) - excess_bytes]
        # Decode ignoring cut characters at multi-byte boundaries
        safe_text = truncated_bytes.decode("utf-8", errors="ignore")
        result["source_text"] = safe_text + suffix

    # Final iterative trim if multi-byte or JSON escaping differences kept it slightly over
    while estimate_metadata_size_bytes(result) > max_bytes and len(result["source_text"]) > len(suffix):
        current_text = result["source_text"][: -len(suffix)]
        shorter = current_text[: max(0, len(current_text) - 100)]
        result["source_text"] = shorter + suffix

    return result


def _coerce_primitive_value(val: Any) -> Any:
    """Safely coerce primitive values to Pinecone-compliant types."""
    if val is None:
        return None
    if isinstance(val, bool):
        return bool(val)

    # Handle numpy types if present without hard numpy import dependency
    type_name = type(val).__name__
    if "int" in type_name and hasattr(val, "item"):
        return int(val.item())
    if "float" in type_name and hasattr(val, "item"):
        return float(val.item())
    if "bool" in type_name and hasattr(val, "item"):
        return bool(val.item())

    if isinstance(val, int):
        return int(val)
    if isinstance(val, float):
        return float(val)
    if isinstance(val, str):
        return str(val)
    if isinstance(val, (list, tuple, set)):
        return [str(item) for item in val if item is not None]

    return val


def to_pinecone_metadata(
    chunk_or_meta: Any,
    filename: str = "",
    document_title: str | None = None,
    max_bytes: int = PINECONE_MAX_METADATA_BYTES,
    auto_truncate: bool = True,
) -> dict[str, Any]:
    """Convert an internal Chunk or ChunkMetadata model into a Pinecone-safe metadata dictionary.

    Operations:
    1. Extracts core fields required for vector indexing, citation rendering, and NLI verification.
    2. Excludes internal coordinates (char_start, char_end).
    3. Performs strict type coercion (converting numpy numbers to native Python types).
    4. Omits None values completely (Pinecone disallows nulls).
    5. Applies defensive truncation if metadata exceeds Pinecone's 40KB limit.
    6. Validates the output against all Pinecone constraints.

    Args:
        chunk_or_meta: A Chunk instance, ChunkMetadata instance, or dictionary.
        filename: Original PDF filename for citation display.
        document_title: Display title of the document. Defaults to filename.
        max_bytes: Maximum allowed metadata size in bytes (default 40,960).
        auto_truncate: If True, truncates oversized source_text with a warning.

    Returns:
        dict[str, Any]: A validated Pinecone-compatible metadata dictionary.

    Raises:
        PineconeMetadataValidationError: If the resulting dictionary violates constraints.
    """
    # Extract values based on input type
    if hasattr(chunk_or_meta, "model_dump"):
        raw_dict = chunk_or_meta.model_dump()
    elif isinstance(chunk_or_meta, dict):
        raw_dict = dict(chunk_or_meta)
    else:
        raw_dict = vars(chunk_or_meta)

    # Determine core field values
    document_id = str(raw_dict.get("document_id", ""))
    chunk_id = str(raw_dict.get("chunk_id", ""))
    chunk_index = int(raw_dict.get("chunk_index", 0))
    page_number = int(raw_dict.get("page_number", 1))
    page_number_end = int(raw_dict.get("page_number_end", page_number))

    # Source text can come from 'text' (Chunk) or 'source_text' (ChunkMetadata/dict)
    source_text = str(raw_dict.get("text") or raw_dict.get("source_text") or "")
    token_count = int(raw_dict.get("token_count", 0))

    # Filename and document title resolution
    resolved_filename = str(filename or raw_dict.get("filename") or "")
    resolved_title = str(
        document_title
        or raw_dict.get("document_title")
        or resolved_filename
        or (f"Document {document_id}" if document_id else "")
    )

    # Assemble base dictionary
    # Note: char_start and char_end are explicitly omitted here
    meta: dict[str, Any] = {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "page_number": page_number,
        "page_number_end": page_number_end,
        "source_text": source_text,
        "token_count": token_count,
        "filename": resolved_filename,
        "document_title": resolved_title,
    }

    # Include any extra custom metadata passed in, excluding internal offset keys
    excluded_keys = {"char_start", "char_end", "text"}
    for k, v in raw_dict.items():
        if k not in meta and k not in excluded_keys:
            coerced = _coerce_primitive_value(v)
            if coerced is not None:
                meta[k] = coerced

    # Clean and coerce all primary fields, dropping any None values
    clean_meta: dict[str, Any] = {}
    for k, v in meta.items():
        coerced = _coerce_primitive_value(v)
        if coerced is not None:
            clean_meta[k] = coerced

    # Handle size limits
    if auto_truncate and estimate_metadata_size_bytes(clean_meta) > max_bytes:
        clean_meta = truncate_metadata_source_text(clean_meta, max_bytes=max_bytes)

    # Validate against Pinecone constraints
    errors = validate_pinecone_metadata(clean_meta, max_bytes=max_bytes)
    if errors:
        raise PineconeMetadataValidationError(
            f"Pinecone metadata validation failed with {len(errors)} error(s): "
            + "; ".join(errors)
        )

    return clean_meta
