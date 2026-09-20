"""
Pydantic schemas for request/response models.

This file will be populated in later phases as endpoints are implemented.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


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
    failure_reason: str | None = None
    retryable: bool | None = None
    page_count: int | None = None
    total_char_count: int | None = None
    chunk_count: int | None = None
    upserted_count: int | None = None
    current_stage: str | None = None
    total_tokens: int | None = None
    processing_time_seconds: float | None = None
    stage_timings: dict[str, float] | None = None


class PageText(BaseModel):
    """Extracted text and character metadata for a single PDF page."""

    page_number: int = Field(description="1-indexed page number within the PDF document")
    text: str = Field(description="Cleaned, structured extracted text from the page")
    char_count: int = Field(description="Total number of characters extracted on this page")


class DocumentExtractionResponse(BaseModel):
    """Response schema returned by the document extraction endpoint."""

    document_id: str
    status: DocumentStatus
    failure_reason: str | None = None
    page_count: int | None = None
    total_char_count: int | None = None
    pages: list[PageText] = Field(default_factory=list)
    cleaning_reports: list[dict[str, Any]] = Field(default_factory=list)


class Chunk(BaseModel):
    """A semantic text chunk with exact document, page, and character offset traceability."""

    chunk_id: str = Field(description="Unique UUID4 identifier for the chunk")
    document_id: str = Field(description="ID of the parent source document")
    chunk_index: int = Field(description="Sequential order index of chunk within the document")
    text: str = Field(description="Text content of the chunk")
    token_count: int = Field(description="Total token count measured by tokenizer")
    page_number: int = Field(description="Starting page number of the chunk (1-indexed)")
    page_number_end: int = Field(description="Ending page number of the chunk (1-indexed)")
    char_start: int = Field(description="Character offset on the starting page's cleaned text")
    char_end: int = Field(description="Character offset on the ending page's cleaned text")
    embedding: list[float] | None = Field(
        default=None,
        description="Dense vector embedding of chunk text (e.g. 384-dimensional)",
    )

    def to_pinecone_metadata(
        self,
        filename: str = "",
        document_title: str | None = None,
        max_bytes: int = 40960,
        auto_truncate: bool = True,
    ) -> dict[str, Any]:
        """Convert chunk into a Pinecone-safe metadata dictionary."""
        from app.ingestion.chunk_metadata import to_pinecone_metadata as _convert

        return _convert(
            self,
            filename=filename,
            document_title=document_title,
            max_bytes=max_bytes,
            auto_truncate=auto_truncate,
        )


class ChunkListResponse(BaseModel):
    """Response schema returned by the document chunking endpoint."""

    document_id: str
    chunk_count: int
    chunks: list[Chunk]


class DocumentListResponse(BaseModel):
    """Response schema for listing all registered documents."""

    documents: list[DocumentMetadata]
    total: int


class ChunkMetadata(BaseModel):
    """Metadata associated with each text chunk stored in Pinecone vector index."""

    document_id: str = Field(description="Unique UUID4 identifier of the parent document")
    chunk_id: str = Field(description="Unique UUID4 identifier of the chunk")
    chunk_index: int = Field(default=0, description="Sequential order index of chunk within document")
    page_number: int = Field(description="Starting page number of the chunk (1-indexed)")
    page_number_end: int | None = Field(default=None, description="Ending page number of the chunk (1-indexed)")
    source_text: str = Field(description="Full text content of the chunk for retrieval and verification")
    token_count: int = Field(default=0, description="Tokenizer token count of the chunk")
    filename: str = Field(default="", description="Original PDF filename for display in citations")
    document_title: str = Field(default="", description="Display title of document (defaults to filename)")

    # Internal coordinates (excluded from Pinecone metadata dictionary)
    char_start: int | None = Field(default=None, exclude=True, description="Internal char start offset")
    char_end: int | None = Field(default=None, exclude=True, description="Internal char end offset")

    @model_validator(mode="after")
    def populate_defaults(self) -> "ChunkMetadata":
        if self.page_number_end is None:
            self.page_number_end = self.page_number
        if not self.document_title:
            self.document_title = self.filename or (f"Document {self.document_id}" if self.document_id else "")
        return self

    def to_pinecone_metadata(
        self,
        max_bytes: int = 40960,
        auto_truncate: bool = True,
    ) -> dict[str, Any]:
        """Serialize model to a validated Pinecone-compatible metadata dictionary."""
        from app.ingestion.chunk_metadata import to_pinecone_metadata as _convert

        return _convert(
            self,
            filename=self.filename,
            document_title=self.document_title,
            max_bytes=max_bytes,
            auto_truncate=auto_truncate,
        )

    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        filename: str = "",
        document_title: str | None = None,
    ) -> "ChunkMetadata":
        """Create a ChunkMetadata instance from an ingestion Chunk."""
        return cls(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            page_number_end=chunk.page_number_end,
            source_text=chunk.text,
            token_count=chunk.token_count,
            filename=filename,
            document_title=document_title or filename,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
        )

    @classmethod
    def from_pinecone_metadata(cls, metadata: dict[str, Any]) -> "ChunkMetadata":
        """Deserialize a Pinecone metadata dictionary back into a ChunkMetadata instance."""
        source_text = str(metadata.get("source_text") or metadata.get("text") or "")
        doc_id = str(metadata.get("document_id", ""))
        chunk_id = str(metadata.get("chunk_id", ""))
        p_start = int(metadata.get("page_number", 1))
        p_end = int(metadata.get("page_number_end", p_start))
        idx = int(metadata.get("chunk_index", 0))
        tokens = int(metadata.get("token_count", 0))
        fname = str(metadata.get("filename", ""))
        title = str(metadata.get("document_title", fname))
        c_start = metadata.get("char_start")
        c_end = metadata.get("char_end")
        return cls(
            document_id=doc_id,
            chunk_id=chunk_id,
            chunk_index=idx,
            page_number=p_start,
            page_number_end=p_end,
            source_text=source_text,
            token_count=tokens,
            filename=fname,
            document_title=title,
            char_start=int(c_start) if c_start is not None else None,
            char_end=int(c_end) if c_end is not None else None,
        )


class IngestionResult(BaseModel):
    """Result returned after running the complete document ingestion pipeline."""

    document_id: str = Field(description="Unique UUID4 identifier for the document")
    status: DocumentStatus = Field(description="Final lifecycle status of the document")
    chunk_count: int = Field(default=0, description="Total number of chunks produced")
    upserted_count: int = Field(default=0, description="Total vectors successfully upserted to Pinecone")
    total_tokens: int = Field(default=0, description="Total number of tokens across all chunks")
    processing_time_seconds: float = Field(default=0.0, description="Total pipeline execution time in seconds")
    failure_reason: str | None = Field(default=None, description="Detailed failure reason if pipeline failed")
    retryable: bool | None = Field(default=None, description="Whether the failure is eligible for retry")
    current_stage: str | None = Field(default=None, description="Current or terminal pipeline stage")
    stage_timings: dict[str, float] = Field(default_factory=dict, description="Execution duration in seconds per stage")


class DocumentStatusResponse(BaseModel):
    """Response schema for polling document ingestion status and stage progress."""

    document_id: str = Field(description="Unique UUID4 identifier for the document")
    status: DocumentStatus = Field(description="Current status (uploaded, processing, ready, failed)")
    current_stage: str | None = Field(default=None, description="Current stage (extracting, cleaning, chunking, embedding, upserting, ready, failed)")
    failure_reason: str | None = Field(default=None, description="Detailed failure message if status is failed")
    retryable: bool | None = Field(default=None, description="Whether a failed document is eligible for retry")
    chunk_count: int | None = Field(default=None, description="Total chunks if chunking completed")
    upserted_count: int | None = Field(default=None, description="Total vectors upserted to Pinecone")
    page_count: int | None = Field(default=None, description="Total extracted pages if extraction completed")
    total_tokens: int | None = Field(default=None, description="Total token count if chunking completed")
    total_char_count: int | None = Field(default=None, description="Total extracted character count")
    processing_time_seconds: float | None = Field(default=None, description="Elapsed processing time")
    stage_timings: dict[str, float] | None = Field(default=None, description="Breakdown of timing per stage")


class DocumentVerificationResponse(BaseModel):
    """Response schema for on-demand Pinecone vector verification."""

    document_id: str = Field(description="Unique UUID4 identifier for the document")
    status: DocumentStatus = Field(description="Current document status")
    verified: bool = Field(description="Whether all sampled vectors matched on-disk chunk source text")
    chunk_count: int = Field(default=0, description="Total chunks in local document storage")
    sampled_count: int = Field(default=0, description="Number of chunks sampled and fetched from Pinecone")
    matched_count: int = Field(default=0, description="Number of sampled chunks whose source text matched exactly")
    mismatches: list[str] = Field(default_factory=list, description="List of mismatch descriptions if any")
    verified_chunks: list[dict[str, Any]] = Field(default_factory=list, description="Sampled chunk verification details")
    message: str = Field(description="Human-readable verification result summary")




