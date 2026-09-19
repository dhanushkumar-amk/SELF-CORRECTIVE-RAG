"""
Pydantic schemas for request/response models.

This file will be populated in later phases as endpoints are implemented.
"""

from enum import Enum

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response schema for the health check endpoint."""

    status: str


class DocumentStatus(str, Enum):
    """Lifecycle status of an ingested document."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DocumentUploadResponse(BaseModel):
    """Response schema returned after a successful document upload."""

    document_id: str = Field(description="Unique UUID4 identifier for the document")
    filename: str = Field(description="Original sanitized filename")
    status: DocumentStatus = Field(default=DocumentStatus.UPLOADED)
    size_bytes: int = Field(description="File size in bytes")


class DocumentMetadata(BaseModel):
    """Metadata schema representing an uploaded document record in the registry."""

    document_id: str
    filename: str
    upload_timestamp: str
    size_bytes: int
    status: DocumentStatus = DocumentStatus.UPLOADED
    file_path: str | None = None


class DocumentListResponse(BaseModel):
    """Response schema for listing all registered documents."""

    documents: list[DocumentMetadata]
    total: int



class ChunkMetadata(BaseModel):
    """Metadata associated with each text chunk stored in Pinecone.

    Pinecone metadata values are strictly restricted to strings, numbers (int/float),
    booleans, or lists of strings. All fields in this model conform directly to these types.
    """

    document_id: str
    chunk_id: str
    page_number: int
    source_text: str
    char_start: int
    char_end: int

    def to_pinecone_metadata(self) -> dict[str, str | int]:
        """Serialize model to a Pinecone-compatible metadata dictionary.

        Ensures all keys and values conform strictly to Pinecone's metadata value constraints.
        """
        return self.model_dump()

    @classmethod
    def from_pinecone_metadata(cls, metadata: dict) -> "ChunkMetadata":
        """Deserialize a Pinecone metadata dictionary back into a validated ChunkMetadata instance."""
        return cls(**metadata)

