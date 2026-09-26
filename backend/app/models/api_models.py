"""
Public API Contract Models (Phase 44) — app/models/api_models.py.

This module is the versioned public contract exposed through the REST API
(surface: ``/api/v1/*``). It is deliberately kept separate from the internal
domain models in ``app/models/schemas.py`` (ClaimWithSource, RetrievalResult,
Chunk, ...), which are implementation details of the pipeline and may evolve
independently.

Design decisions:
1. Single source of truth for every request/response DTO consumed by FastAPI.
2. Uniform PascalCase naming (e.g. ``QueryRequest``, ``DocumentUploadResponse``),
   and every response model is named ``...Response``.
3. Key models carry realistic ``examples`` (exposed in the OpenAPI ``/docs`` UI)
   so the API is self-documenting.
4. Backward-compatible API evolution is handled via the ``/api/v1`` route prefix
   registered in ``app.main``; a breaking change would introduce ``/api/v2``
   alongside it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.schemas import (
    Chunk,
    ClaimWithSource,
    DocumentStatus,
    PageText,
    RetrievalResult,
)

__all__ = [
    "ChunkListResponse",
    "DocumentExtractionResponse",
    "DocumentListResponse",
    "DocumentMetadata",
    "DocumentStatusResponse",
    "DocumentUploadResponse",
    "DocumentVerificationResponse",
    "ErrorResponse",
    "HealthResponse",
    "IngestionResult",
    "QueryRequest",
    "QueryResponse",
]


class HealthResponse(BaseModel):
    """Response schema for the health check endpoint."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "ok"}]},
    )

    status: str = Field(examples=["ok"], description="Health status indicator")


class ErrorResponse(BaseModel):
    """Standardized API error response payload returned by every endpoint on failure."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error": "RATE_LIMIT_EXCEEDED",
                    "detail": "Rate limit exceeded: maximum 10 queries per minute per client. Retry after 32 seconds.",
                    "status_code": 429,
                }
            ]
        },
    )

    error: str = Field(description="High-level machine-readable error classification code")
    detail: str = Field(description="Detailed human-readable error explanation")
    status_code: int = Field(description="HTTP status code")


class QueryRequest(BaseModel):
    """Request payload for executing a RAG query through the self-correction graph."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "What were Q3 revenues and key growth metrics?",
                    "document_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "stream": True,
                }
            ]
        },
    )

    query: str = Field(
        min_length=1,
        examples=["What were Q3 revenues and key growth metrics?"],
        description="Natural language user question",
    )
    document_id: str | None = Field(
        default=None,
        examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
        description="Optional document UUID for single-document retrieval scoping",
    )
    stream: bool = Field(
        default=True,
        examples=[True],
        description="True to stream state machine progress via Server-Sent Events (SSE), False for a single JSON response",
    )

    @field_validator("query", mode="before")
    @classmethod
    def _sanitize_query(cls, v: str) -> str:
        if isinstance(v, str):
            v = v.strip()
        if not v:
            raise ValueError("Query string cannot be empty or whitespace-only.")
        return v


class QueryResponse(BaseModel):
    """Structured response payload returned by non-streaming query execution."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "What were Q3 revenues and key growth metrics?",
                    "final_status": "fully_verified",
                    "final_answer_text": "Q3 revenue was $12.4M, up 18% quarter-over-quarter, driven by enterprise adoption.",
                    "claims": [
                        {
                            "claim_id": "9c7b6a55-3d1e-4c8f-9a2b-1e5d4c3b2a19",
                            "claim_text": "Q3 revenue was $12.4M.",
                            "source_chunk_id": "chunk-uuid-1",
                            "verification_status": "entailed",
                            "confidence": 0.97,
                        }
                    ],
                    "retry_count": 0,
                    "retrieved_chunks": [],
                    "latency_ms": 1834.5,
                }
            ]
        },
    )

    query: str = Field(description="Original user query")
    final_status: str = Field(
        description="Aggregate system verification status ('fully_verified', 'partially_verified', 'unverifiable')"
    )
    final_answer_text: str = Field(description="Synthesized final user-facing answer text")
    claims: list[ClaimWithSource] = Field(
        default_factory=list,
        description="List of atomic claims with verification statuses and confidence scores",
    )
    retry_count: int = Field(default=0, description="Total self-correction retries executed")
    retrieved_chunks: list[RetrievalResult] = Field(
        default_factory=list,
        description="Final context chunks used for answer generation",
    )
    latency_ms: float = Field(default=0.0, description="Total execution latency in milliseconds")


class DocumentUploadResponse(BaseModel):
    """Response schema returned after a successful document upload."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "document_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "filename": "Q3-earnings-report.pdf",
                    "status": "uploaded",
                    "size_bytes": 2482136,
                }
            ]
        },
    )

    document_id: str = Field(description="Unique UUID4 identifier for the document")
    filename: str = Field(description="Original sanitized filename")
    status: DocumentStatus = Field(default=DocumentStatus.UPLOADED)
    size_bytes: int = Field(description="File size in bytes")


class DocumentMetadata(BaseModel):
    """Metadata schema representing an uploaded document record in the registry."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "document_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "filename": "Q3-earnings-report.pdf",
                    "upload_timestamp": "2025-09-01T14:22:10.512Z",
                    "size_bytes": 2482136,
                    "status": "ready",
                    "chunk_count": 42,
                }
            ]
        },
    )

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


class DocumentExtractionResponse(BaseModel):
    """Response schema returned by the document extraction endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "document_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "status": "ready",
                    "failure_reason": None,
                    "page_count": 12,
                    "total_char_count": 45821,
                    "pages": [],
                    "cleaning_reports": [],
                }
            ]
        },
    )

    document_id: str
    status: DocumentStatus
    failure_reason: str | None = None
    page_count: int | None = None
    total_char_count: int | None = None
    pages: list[PageText] = Field(default_factory=list)
    cleaning_reports: list[dict[str, Any]] = Field(default_factory=list)


class ChunkListResponse(BaseModel):
    """Response schema returned by the document chunking endpoint."""

    document_id: str
    chunk_count: int
    chunks: list[Chunk]


class DocumentListResponse(BaseModel):
    """Response schema for listing all registered documents."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "documents": [
                        {
                            "document_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                            "filename": "Q3-earnings-report.pdf",
                            "upload_timestamp": "2025-09-01T14:22:10.512Z",
                            "size_bytes": 2482136,
                            "status": "ready",
                        }
                    ],
                    "total": 1,
                }
            ]
        },
    )

    documents: list[DocumentMetadata]
    total: int


class IngestionResult(BaseModel):
    """Result returned after running the complete document ingestion pipeline."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "document_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "status": "ready",
                    "chunk_count": 42,
                    "upserted_count": 42,
                    "total_tokens": 21408,
                    "processing_time_seconds": 61.3,
                    "failure_reason": None,
                    "retryable": None,
                    "current_stage": "ready",
                    "stage_timings": {"extracting": 1.2, "chunking": 2.1, "embedding": 55.0, "upserting": 3.0},
                }
            ]
        },
    )

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

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "document_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "filename": "Q3-earnings-report.pdf",
                    "status": "ready",
                    "current_stage": "ready",
                    "progress_percent": 100,
                    "failure_reason": None,
                    "chunk_count": 42,
                    "upserted_count": 42,
                }
            ]
        },
    )

    document_id: str = Field(description="Unique UUID4 identifier for the document")
    filename: str = Field(default="", description="Original PDF filename")
    status: DocumentStatus = Field(description="Current status (uploaded, processing, ready, failed)")
    current_stage: str | None = Field(default=None, description="Current stage (extracting, cleaning, chunking, embedding, upserting, ready, failed)")
    progress_percent: int = Field(default=0, description="Approximate processing percentage (0-100%) for UI progress display")
    failure_reason: str | None = Field(default=None, description="Detailed failure message if status is failed")
    retryable: bool | None = Field(default=None, description="Whether a failed document is eligible for retry")
    chunk_count: int | None = Field(default=None, description="Total chunks if chunking completed")
    upserted_count: int | None = Field(default=None, description="Total vectors upserted to Pinecone")
    page_count: int | None = Field(default=None, description="Total extracted pages if extraction completed")
    total_tokens: int | None = Field(default=None, description="Total token count if chunking completed")
    total_char_count: int | None = Field(default=None, description="Total extracted character count")
    uploaded_at: str = Field(default="", description="ISO-8601 upload timestamp")
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
