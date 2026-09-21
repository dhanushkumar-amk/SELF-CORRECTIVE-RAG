"""
Pydantic schemas for request/response models.

This file will be populated in later phases as endpoints are implemented.
"""

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class HealthResponse(BaseModel):
    """Response schema for the health check endpoint."""

    status: str


class DocumentStatus(str, Enum):
    """Lifecycle status of an ingested document across pipeline stages."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    EXTRACTING = "extracting"
    CLEANING = "cleaning"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    UPSERTING = "upserting"
    READY = "ready"
    FAILED = "failed"


class VerificationStatus(str, Enum):
    """Classification status of a claim after NLI premise-hypothesis verification."""

    PENDING = "pending"
    ENTAILED = "entailed"
    CONTRADICTED = "contradicted"
    NEUTRAL = "neutral"
    UNVERIFIABLE = "unverifiable"


# Map of valid status transitions in the document ingestion state machine
VALID_STATUS_TRANSITIONS: dict[DocumentStatus, set[DocumentStatus]] = {
    DocumentStatus.UPLOADED: {
        DocumentStatus.PROCESSING,
        DocumentStatus.EXTRACTING,
        DocumentStatus.FAILED,
    },
    DocumentStatus.PROCESSING: {
        DocumentStatus.EXTRACTING,
        DocumentStatus.CLEANING,
        DocumentStatus.CHUNKING,
        DocumentStatus.EMBEDDING,
        DocumentStatus.UPSERTING,
        DocumentStatus.READY,
        DocumentStatus.FAILED,
    },
    DocumentStatus.EXTRACTING: {
        DocumentStatus.CLEANING,
        DocumentStatus.FAILED,
    },
    DocumentStatus.CLEANING: {
        DocumentStatus.CHUNKING,
        DocumentStatus.FAILED,
    },
    DocumentStatus.CHUNKING: {
        DocumentStatus.EMBEDDING,
        DocumentStatus.FAILED,
    },
    DocumentStatus.EMBEDDING: {
        DocumentStatus.UPSERTING,
        DocumentStatus.FAILED,
    },
    DocumentStatus.UPSERTING: {
        DocumentStatus.READY,
        DocumentStatus.FAILED,
    },
    DocumentStatus.READY: {
        DocumentStatus.UPLOADED,
        DocumentStatus.PROCESSING,
        DocumentStatus.EXTRACTING,
    },
    DocumentStatus.FAILED: {
        DocumentStatus.UPLOADED,
        DocumentStatus.PROCESSING,
        DocumentStatus.EXTRACTING,
    },
}


def validate_transition(current: DocumentStatus | str, target: DocumentStatus | str) -> bool:
    """Validate whether transitioning from current status to target status is permitted.

    Args:
        current: Current DocumentStatus enum or string value.
        target: Target DocumentStatus enum or string value to transition into.

    Returns:
        True if transition is valid or idempotent, False otherwise.
    """
    try:
        curr_enum = DocumentStatus(current)
        targ_enum = DocumentStatus(target)
    except ValueError:
        return False

    if curr_enum == targ_enum:
        return True

    valid_targets = VALID_STATUS_TRANSITIONS.get(curr_enum, set())
    return targ_enum in valid_targets


STAGE_PROGRESS_PERCENT: dict[str, int] = {
    "uploaded": 0,
    "extracting": 15,
    "cleaning": 30,
    "chunking": 45,
    "embedding": 70,
    "upserting": 90,
    "ready": 100,
}


def calculate_progress_percent(
    status: DocumentStatus | str, current_stage: str | None = None
) -> int:
    """Calculate approximate percentage completion (0-100%) for UI progress display.

    Progress Mapping:
    - UPLOADED: 0%
    - EXTRACTING: 15%
    - CLEANING: 30%
    - CHUNKING: 45%
    - EMBEDDING: 70%
    - UPSERTING: 90%
    - READY: 100%
    - FAILED: Retains progress percentage of stage at which processing halted.
    """
    s_val = status.value if isinstance(status, DocumentStatus) else str(status).lower()
    stage_val = (current_stage or "").lower()

    if s_val == "ready":
        return 100
    if stage_val in STAGE_PROGRESS_PERCENT:
        return STAGE_PROGRESS_PERCENT[stage_val]
    if s_val in STAGE_PROGRESS_PERCENT:
        return STAGE_PROGRESS_PERCENT[s_val]
    if s_val == "processing":
        return 15
    return 0


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


class RetrievalResult(BaseModel):
    """Unified retrieval result container returned by dense, sparse, and fused hybrid search."""

    chunk_id: str = Field(description="Unique UUID4 identifier of the chunk (or vector id)")
    score: float = Field(description="Relevance or RRF score")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary (source_text, document_id, page_number, filename, etc.)")


class RerankResult(BaseModel):
    """Structured response object returned by threshold-based reranking."""

    chunks: list[RetrievalResult] = Field(default_factory=list, description="Filtered and top-n truncated list of relevant chunks")
    reason: str | None = Field(default=None, description="Reason flag if filtering excluded all candidates (e.g. 'no_relevant_chunks_found')")
    total_candidates: int = Field(default=0, description="Total candidate chunks received before threshold filtering")
    relevant_count: int = Field(default=0, description="Number of candidate chunks passing min relevance score threshold")


class LLMResponse(BaseModel):
    """Unified response object returned by the LLM generation provider interface."""

    content: str = Field(description="Generated text output from the LLM provider")
    provider: str = Field(description="Name of provider serving request ('groq' or 'gemini')")
    model_name: str = Field(description="Name of specific LLM model executed")
    latency_ms: float = Field(default=0.0, description="Execution duration in milliseconds")
    fallback_triggered: bool = Field(default=False, description="Whether fallback mechanism was triggered")
    error: str | None = Field(default=None, description="Error message if generation failed")


class Claim(BaseModel):
    """An individual factual claim/sentence bound to its source chunk_id citation."""

    claim_text: str = Field(description="Factual statement or sentence in the generated answer")
    source_chunk_id: str = Field(description="Exact chunk_id supporting this claim")


class GeneratedAnswer(BaseModel):
    """Structured response schema returned by the citation-forced answer generator."""

    claims: list[Claim] = Field(default_factory=list, description="List of factual claims tagged with source_chunk_id citations")
    insufficient_information: bool = Field(default=False, description="True if provided chunks do not contain enough info to answer the query")
    unverified_claims: list[Claim] = Field(default_factory=list, description="Claims whose source_chunk_id did not match any input chunk_id")
    raw_response: str | None = Field(default=None, description="Original raw LLM response text before JSON parsing")
    provider: str | None = Field(default=None, description="LLM provider name ('groq' or 'gemini')")
    model_name: str | None = Field(default=None, description="Name of LLM model executed")
    latency_ms: float = Field(default=0.0, description="Generation latency in milliseconds")


class ParseError(BaseModel):
    """Details of a failed LLM JSON parsing or validation attempt."""

    raw_text: str = Field(description="Original raw text response from LLM provider")
    error_type: str = Field(description="Classification of error (empty_response, json_decode_error, contradictory_response, invalid_schema)")
    error_message: str = Field(description="Detailed error message describing parsing failure")
    is_recoverable: bool = Field(default=False, description="True if output could be recovered via sanitization, False if unrecoverable and requires LLM retry")


class GenerationError(Exception):
    """Custom exception raised when answer generation fails across all retry and fallback attempts."""

    def __init__(
        self,
        message: str = "Unable to generate a reliable answer, please try rephrasing your question.",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ClaimWithSource(BaseModel):
    """An atomic factual claim resolved against its underlying source chunk context."""

    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique UUID for referencing individual claim in correction loop")
    claim_text: str = Field(description="Atomic factual statement or sentence")
    source_chunk_id: str = Field(description="Cited source chunk ID")
    source_text: str | None = Field(default=None, description="Full source chunk text for NLI premise verification")
    page_number: int | None = Field(default=None, description="Starting page number of source chunk (1-indexed)")
    page_number_end: int | None = Field(default=None, description="Ending page number of source chunk (1-indexed)")
    is_valid_source: bool = Field(default=True, description="False if source_chunk_id is missing or hallucinated in retrieved context")
    document_id: str | None = Field(default=None, description="Parent document ID if resolved")
    filename: str | None = Field(default=None, description="Source PDF filename for citations")
    verification_status: VerificationStatus | None = Field(default=None, description="Populated in Phase 34-35 by NLI verification logic")
    confidence: float | None = Field(default=None, description="Populated in Phase 34-35 with NLI prediction confidence")


class NLIScore(BaseModel):
    """Raw logits and softmax probabilities returned by NLI DeBERTa CrossEncoder model."""

    contradiction: float = Field(description="Probability/score for contradiction label")
    entailment: float = Field(description="Probability/score for entailment label")
    neutral: float = Field(description="Probability/score for neutral label")
    predicted_label: VerificationStatus = Field(description="VerificationStatus enum label corresponding to highest NLI score")


class ErrorResponse(BaseModel):
    """Standardized API error response payload across all endpoints."""

    error: str = Field(description="High-level error classification code")
    detail: str = Field(description="Detailed human-readable error explanation")
    status_code: int = Field(description="HTTP status code")


class QueryRequest(BaseModel):
    """Request payload for executing a RAG query through the self-correction graph."""

    query: str = Field(min_length=1, description="Natural language user question")
    document_id: str | None = Field(default=None, description="Optional document UUID for single-document retrieval scoping")
    stream: bool = Field(default=True, description="True to stream state machine progress via Server-Sent Events (SSE), False for JSON response")

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

    query: str = Field(description="Original user query")
    final_status: str = Field(description="Aggregate system verification status ('fully_verified', 'partially_verified', 'unverifiable')")
    final_answer_text: str = Field(description="Synthesized final user-facing answer text")
    claims: list[ClaimWithSource] = Field(default_factory=list, description="List of atomic claims with verification statuses and confidence scores")
    retry_count: int = Field(default=0, description="Total self-correction retries executed")
    retrieved_chunks: list[RetrievalResult] = Field(default_factory=list, description="Final context chunks used for answer generation")
    latency_ms: float = Field(default=0.0, description="Total execution latency in milliseconds")








