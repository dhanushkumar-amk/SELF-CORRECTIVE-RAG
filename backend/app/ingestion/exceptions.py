"""
Ingestion pipeline exception taxonomy.

Classifies errors into two distinct categories:
1. PermanentIngestionError: Non-retriable failures caused by invalid content, unsupported formats,
   corrupted files, missing inputs, or invalid metadata constraints. Processing stops immediately.
2. TransientIngestionError: Retriable failures caused by temporary network blips, Pinecone rate limits,
   connection timeouts, or transient model inference interruptions. Safe for automatic retry.
"""

from __future__ import annotations


class IngestionPipelineError(Exception):
    """Base exception class for all document ingestion pipeline errors."""

    def __init__(self, message: str, stage: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.stage = stage

    def __str__(self) -> str:
        if self.stage:
            return f"[{self.stage}] {self.message}"
        return self.message


class PermanentIngestionError(IngestionPipelineError):
    """Exception raised when document processing fails permanently and cannot succeed on retry.

    Examples:
    - PDF file missing on disk
    - Corrupted or unreadable PDF streams
    - Password-protected/encrypted PDFs
    - Image-only / scanned PDFs with zero extractable text layer
    - Zero valid text chunks generated
    - Unparseable or invalid metadata
    """

    pass


class TransientIngestionError(IngestionPipelineError):
    """Exception raised when document processing fails due to temporary infrastructure blips.

    Examples:
    - Pinecone network connection timeouts
    - Pinecone HTTP 429 rate limit errors
    - Pinecone server 5xx errors
    - Temporary network timeouts during local model operations
    """

    pass
