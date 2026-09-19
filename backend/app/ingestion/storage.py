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
from app.models.schemas import DocumentMetadata, DocumentStatus

logger = get_logger(__name__)

# Re-entrant thread lock for synchronizing registry file reads and atomic writes
_REGISTRY_LOCK = threading.Lock()


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

    def update_status(
        self, document_id: str, status: DocumentStatus
    ) -> DocumentMetadata | None:
        """Update the processing status of a document."""
        with _REGISTRY_LOCK:
            current = self._load_registry_unlocked()
            if document_id not in current:
                return None
            current[document_id]["status"] = status.value
            self._save_registry_unlocked(current)
            return DocumentMetadata(**current[document_id])


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
