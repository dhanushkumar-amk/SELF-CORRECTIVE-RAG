"""
Document storage management and registry tracking for uploaded PDFs.

Design Notes:
- Files are saved to a local disk directory organized as `uploads/{document_id}/{filename}`.
- Production Note: In a cloud production deployment (Phase 50), this local storage
  implementation should be replaced with an object store such as AWS S3 or GCP Cloud Storage,
  paired with a database (e.g. PostgreSQL) for document records. For this single-node
  portfolio setup, local storage with atomic JSON registry persistence is used.
- Registry Concurrency: Uses a threading.Lock and atomic temp-file replacement
  to guarantee `registry.json` is never corrupted during concurrent upload requests.
"""

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import (
    Chunk,
    DocumentMetadata,
    DocumentStatus,
    PageText,
    validate_transition,
)

logger = get_logger(__name__)

# Re-entrant thread lock for synchronizing registry file reads and atomic writes
_REGISTRY_LOCK = threading.Lock()


def check_registry_consistency(doc: DocumentMetadata) -> tuple[bool, list[str]]:
    """Audit the logical consistency of a DocumentMetadata record.

    Consistency Rules:
    1. READY status must NOT have a failure_reason set.
    2. FAILED status MUST have a non-empty failure_reason set.
    3. FAILED status MUST have a boolean retryable flag (True or False).
    4. Non-failed statuses must NOT have a failure_reason set.
    5. READY status must have non-negative chunk_count and upserted_count if present.
    6. Page count / total tokens must be non-negative if set.

    Returns:
        Tuple of (is_consistent: bool, violations: list[str]).
    """
    violations: list[str] = []
    s_val = doc.status.value if isinstance(doc.status, DocumentStatus) else str(doc.status)

    if s_val == "ready":
        if doc.failure_reason:
            violations.append(f"READY document has failure_reason='{doc.failure_reason}'")
        if doc.chunk_count is not None and doc.chunk_count < 0:
            violations.append(f"READY document has negative chunk_count={doc.chunk_count}")
        if doc.upserted_count is not None and doc.upserted_count < 0:
            violations.append(f"READY document has negative upserted_count={doc.upserted_count}")
    elif s_val == "failed":
        if not doc.failure_reason or not doc.failure_reason.strip():
            violations.append("FAILED document missing failure_reason")
        if doc.retryable is None:
            violations.append("FAILED document missing retryable boolean flag")
    else:
        if doc.failure_reason:
            violations.append(f"Document in status '{s_val}' has failure_reason='{doc.failure_reason}'")

    if doc.page_count is not None and doc.page_count < 0:
        violations.append(f"Document has negative page_count={doc.page_count}")
    if doc.total_tokens is not None and doc.total_tokens < 0:
        violations.append(f"Document has negative total_tokens={doc.total_tokens}")

    return (len(violations) == 0, violations)


def get_upload_dir() -> Path:
    """Return the absolute Path to the configured upload storage directory."""
    base_dir = Path(settings.UPLOAD_DIR)
    if not base_dir.is_absolute():
        base_dir = Path.cwd() / base_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def sanitize_filename(filename: str | None) -> str:
    """Sanitize the uploaded filename to prevent directory traversal or invalid path characters."""
    if not filename:
        return "document.pdf"
    # Strip directory components
    clean_name = Path(filename).name
    # Filter non-safe characters
    clean_name = re.sub(r"[^\w\s\.-]", "_", clean_name).strip()
    if not clean_name.lower().endswith(".pdf"):
        clean_name = f"{clean_name}.pdf"
    return clean_name or "document.pdf"


class DocumentRegistry:
    """Thread-safe JSON-backed document registry."""

    def __init__(self, upload_dir: Path | None = None) -> None:
        self._upload_dir = upload_dir

    @property
    def upload_dir(self) -> Path:
        """Return target upload directory, defaulting dynamically to configured directory."""
        return self._upload_dir if self._upload_dir is not None else get_upload_dir()

    @property
    def registry_file(self) -> Path:
        """Return the Path to registry.json inside the upload directory."""
        return self.upload_dir / "registry.json"

    def _load_registry_unlocked(self) -> dict[str, dict[str, Any]]:
        """Read registry file from disk. Caller must hold _REGISTRY_LOCK."""
        if not self.registry_file.exists():
            return {}
        try:
            content = self.registry_file.read_text(encoding="utf-8")
            if not content.strip():
                return {}
            return json.loads(content)
        except Exception as exc:
            logger.warning(
                "Could not parse %s: %s. Re-initializing empty registry.",
                self.registry_file,
                exc,
            )
            return {}

    def _save_registry_unlocked(self, data: dict[str, dict[str, Any]]) -> None:
        """Atomic write to registry using a temporary file and atomic rename. Caller must hold _REGISTRY_LOCK."""
        temp_file = (
            self.upload_dir
            / f"registry_{os.getpid()}_{threading.get_ident()}.tmp"
        )
        try:
            temp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temp_file.replace(self.registry_file)
        except Exception as exc:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            logger.error("Failed to atomically write registry: %s", exc)
            raise

    def save_document(
        self,
        document_id: str,
        filename: str,
        file_bytes: bytes,
        status: DocumentStatus = DocumentStatus.UPLOADED,
    ) -> DocumentMetadata:
        """Store the document file to disk and record its metadata in the registry."""
        doc_dir = self.upload_dir / document_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        target_path = doc_dir / filename

        # Write binary content
        target_path.write_bytes(file_bytes)

        # Record metadata
        timestamp = datetime.now(timezone.utc).isoformat()
        metadata = DocumentMetadata(
            document_id=document_id,
            filename=filename,
            upload_timestamp=timestamp,
            size_bytes=len(file_bytes),
            status=status,
            file_path=str(target_path.resolve()),
            current_stage="uploaded",
        )

        with _REGISTRY_LOCK:
            current = self._load_registry_unlocked()
            current[document_id] = metadata.model_dump()
            self._save_registry_unlocked(current)

        logger.info(
            "Registered document '%s' (%s, %d bytes)",
            document_id,
            filename,
            len(file_bytes),
        )
        return metadata

    def get_document(self, document_id: str) -> DocumentMetadata | None:
        """Retrieve document metadata by document_id."""
        with _REGISTRY_LOCK:
            current = self._load_registry_unlocked()
            raw = current.get(document_id)
            if not raw:
                return None
            return DocumentMetadata(**raw)

    def list_documents(self) -> list[DocumentMetadata]:
        """List all documents stored in the registry, sorted newest first."""
        with _REGISTRY_LOCK:
            current = self._load_registry_unlocked()
            docs = [DocumentMetadata(**item) for item in current.values()]
            docs.sort(key=lambda d: d.upload_timestamp, reverse=True)
            return docs

    def update_document_metadata(
        self, document_id: str, **kwargs: Any
    ) -> DocumentMetadata | None:
        """Atomically update arbitrary metadata fields on a document record."""
        with _REGISTRY_LOCK:
            current = self._load_registry_unlocked()
            if document_id not in current:
                return None
            record = current[document_id]
            if "status" in kwargs and kwargs["status"] is not None:
                new_status = kwargs["status"]
                curr_status = record.get("status", "uploaded")
                if not validate_transition(curr_status, new_status):
                    err_msg = f"Invalid status transition for doc '{document_id}': '{curr_status}' -> '{new_status}'"
                    logger.error(err_msg)
                    raise ValueError(err_msg)

            for key, val in kwargs.items():
                if isinstance(val, DocumentStatus):
                    record[key] = val.value
                else:
                    record[key] = val
            self._save_registry_unlocked(current)
            return DocumentMetadata(**record)

    def update_status(
        self, document_id: str, status: DocumentStatus
    ) -> DocumentMetadata | None:
        """Update the processing status of a document."""
        return self.update_document_metadata(document_id, status=status)

    def save_extracted_pages(
        self, document_id: str, pages: list[PageText]
    ) -> Path:
        """Store extracted per-page text to uploads/{document_id}/extracted.json."""
        doc_dir = self.upload_dir / document_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        extracted_file = doc_dir / "extracted.json"

        serialized = [p.model_dump() for p in pages]
        temp_file = doc_dir / f"extracted_{os.getpid()}_{threading.get_ident()}.tmp"
        try:
            temp_file.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
            temp_file.replace(extracted_file)
        except Exception:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            raise
        logger.info(
            "Saved extracted text for document '%s' to %s (%d pages)",
            document_id,
            extracted_file,
            len(pages),
        )
        return extracted_file

    def save_raw_and_cleaned_pages(
        self,
        document_id: str,
        raw_pages: list[PageText],
        clean_pages: list[PageText],
    ) -> tuple[Path, Path, Path]:
        """Save extracted_raw.json, extracted_clean.json, and canonical extracted.json.

        During development, preserving both versions enables direct comparison and auditing
        to verify that cleaning never damages citation-critical text.
        """
        doc_dir = self.upload_dir / document_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        raw_file = doc_dir / "extracted_raw.json"
        clean_file = doc_dir / "extracted_clean.json"
        main_file = doc_dir / "extracted.json"

        raw_serialized = [p.model_dump() for p in raw_pages]
        clean_serialized = [p.model_dump() for p in clean_pages]

        # 1. Write raw extracted pages
        raw_file.write_text(json.dumps(raw_serialized, indent=2), encoding="utf-8")

        # 2. Write cleaned extracted pages
        clean_file.write_text(json.dumps(clean_serialized, indent=2), encoding="utf-8")

        # 3. Write canonical extracted.json with atomic replace
        temp_file = doc_dir / f"extracted_{os.getpid()}_{threading.get_ident()}.tmp"
        try:
            temp_file.write_text(json.dumps(clean_serialized, indent=2), encoding="utf-8")
            temp_file.replace(main_file)
        except Exception:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            raise

        logger.info(
            "Stored raw (%d pages) and cleaned (%d pages) text for doc '%s'",
            len(raw_pages),
            len(clean_pages),
            document_id,
        )
        return raw_file, clean_file, main_file

    def get_extracted_pages(self, document_id: str) -> list[PageText] | None:
        """Load extracted per-page text from uploads/{document_id}/extracted.json (or clean fallback)."""
        extracted_file = self.upload_dir / document_id / "extracted.json"
        if not extracted_file.exists():
            extracted_file = self.upload_dir / document_id / "extracted_clean.json"
        if not extracted_file.exists():
            return None
        try:
            content = extracted_file.read_text(encoding="utf-8")
            raw_pages = json.loads(content)
            return [PageText(**p) for p in raw_pages]
        except Exception as exc:
            logger.error("Failed to read %s: %s", extracted_file, exc)
            return None

    def get_raw_extracted_pages(self, document_id: str) -> list[PageText] | None:
        """Load raw uncleaned per-page text from uploads/{document_id}/extracted_raw.json."""
        raw_file = self.upload_dir / document_id / "extracted_raw.json"
        if not raw_file.exists():
            return None
        try:
            content = raw_file.read_text(encoding="utf-8")
            raw_pages = json.loads(content)
            return [PageText(**p) for p in raw_pages]
        except Exception as exc:
            logger.error("Failed to read %s: %s", raw_file, exc)
            return None

    def save_chunks(self, document_id: str, chunks: list[Chunk]) -> Path:
        """Save chunk list to uploads/{document_id}/chunks.json."""
        doc_dir = self.upload_dir / document_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        chunks_file = doc_dir / "chunks.json"

        serialized = [c.model_dump() for c in chunks]
        temp_file = doc_dir / f"chunks_{os.getpid()}_{threading.get_ident()}.tmp"
        try:
            temp_file.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
            temp_file.replace(chunks_file)
        except Exception:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            raise

        logger.info(
            "Saved %d chunks for document '%s' to %s",
            len(chunks),
            document_id,
            chunks_file,
        )
        return chunks_file

    def get_chunks(self, document_id: str) -> list[Chunk] | None:
        """Load chunks from uploads/{document_id}/chunks.json."""
        chunks_file = self.upload_dir / document_id / "chunks.json"
        if not chunks_file.exists():
            return None
        try:
            content = chunks_file.read_text(encoding="utf-8")
            raw_chunks = json.loads(content)
            return [Chunk(**c) for c in raw_chunks]
        except Exception as exc:
            logger.error("Failed to read %s: %s", chunks_file, exc)
            return None


_registry_instance: DocumentRegistry | None = None


def get_document_registry() -> DocumentRegistry:
    """Return the global DocumentRegistry singleton."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = DocumentRegistry()
    return _registry_instance


def reset_document_registry() -> None:
    """Reset the global DocumentRegistry singleton (primarily for test isolation)."""
    global _registry_instance
    _registry_instance = None
