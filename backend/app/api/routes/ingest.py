"""
Document ingestion endpoints for receiving, validating, and managing uploaded PDFs.

Phase 6: File reception, magic byte validation, storage, and registry tracking.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status

from pathlib import Path
import threading

from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.chunker import chunk_document
from app.ingestion.pdf_extractor import (
    PDFExtractionError,
    extract_raw_pages,
    extract_text_by_page,
)
from app.ingestion.pipeline import run_ingestion_pipeline, verify_document_upsert
from app.ingestion.storage import get_document_registry, sanitize_filename
from app.ingestion.text_cleaner import clean_document_pages
from app.models.schemas import (
    Chunk,
    ChunkListResponse,
    DocumentExtractionResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentStatus,
    DocumentStatusResponse,
    DocumentUploadResponse,
    DocumentVerificationResponse,
    IngestionResult,
    PageText,
)

logger = get_logger(__name__)
router = APIRouter()

# Magic bytes identifying a valid PDF document (%PDF-)
PDF_MAGIC_BYTES = b"%PDF-"


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and validate a PDF document",
)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF document to upload")],
) -> DocumentUploadResponse:
    """Accept multipart/form-data PDF upload, perform strict validation, and store file.

    Validation Rules:
    1. Rejects empty files (0 bytes) with HTTP 400.
    2. Rejects files that do not start with `%PDF-` magic bytes with HTTP 400.
    3. Reject files larger than `MAX_UPLOAD_SIZE_MB` with HTTP 400.

    Returns:
        DocumentUploadResponse with generated document_id, filename, status, and size.
    """
    max_bytes = settings.max_upload_size_bytes

    # 1. Read first 5 bytes to verify PDF magic bytes
    header = await file.read(5)
    if len(header) == 0:
        logger.warning("Upload rejected: File is empty (0 bytes).")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes). Please upload a valid PDF document.",
        )

    if not header.startswith(PDF_MAGIC_BYTES):
        logger.warning("Upload rejected: Invalid magic bytes (%r).", header)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid PDF. File header does not match '%PDF-'.",
        )

    # 2. Read the remainder in chunks and enforce size limits
    chunks: list[bytes] = [header]
    total_size = len(header)
    chunk_size = 64 * 1024  # 64 KB

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > max_bytes:
            logger.warning(
                "Upload rejected: File size %d bytes exceeds max limit %d bytes (%d MB).",
                total_size,
                max_bytes,
                settings.MAX_UPLOAD_SIZE_MB,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"File size exceeds maximum allowed limit of "
                    f"{settings.MAX_UPLOAD_SIZE_MB}MB."
                ),
            )
        chunks.append(chunk)

    file_bytes = b"".join(chunks)

    # 3. Generate secure UUID4 and sanitize filename
    document_id = str(uuid.uuid4())
    filename = sanitize_filename(file.filename)

    # 4. Save to local storage and register metadata
    registry = get_document_registry()
    try:
        registry.save_document(
            document_id=document_id,
            filename=filename,
            file_bytes=file_bytes,
            status=DocumentStatus.UPLOADED,
        )
    except Exception as exc:
        logger.error("Failed to save uploaded document %s to disk: %s", document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store uploaded document to disk storage.",
        ) from exc

    return DocumentUploadResponse(
        document_id=document_id,
        filename=filename,
        status=DocumentStatus.UPLOADED,
        size_bytes=len(file_bytes),
    )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all registered documents",
)
async def list_documents() -> DocumentListResponse:
    """Retrieve all uploaded documents recorded in the document registry."""
    registry = get_document_registry()
    docs = registry.list_documents()
    return DocumentListResponse(documents=docs, total=len(docs))


@router.get(
    "/documents/{document_id}",
    response_model=DocumentMetadata,
    summary="Get document details by ID",
)
@router.get(
    "/{document_id}",
    response_model=DocumentMetadata,
    include_in_schema=False,
)
async def get_document(document_id: str) -> DocumentMetadata:
    """Retrieve metadata and ingestion status for a specific document ID."""
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        logger.warning("Document with ID '%s' not found.", document_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )
    return doc


@router.post(
    "/{document_id}/extract",
    response_model=DocumentExtractionResponse,
    summary="Extract text by page from an uploaded PDF document",
)
@router.post(
    "/documents/{document_id}/extract",
    response_model=DocumentExtractionResponse,
    include_in_schema=False,
)
async def extract_document(document_id: str) -> DocumentExtractionResponse:
    """Trigger PDF text extraction for an uploaded document.

    Updates document status:
    1. Sets status to 'processing'.
    2. Runs pypdf extraction, page-number preservation, and running header/footer stripping.
    3. If successful, writes extracted text to `uploads/{document_id}/extracted.json`
       and marks status 'ready'.
    4. If document fails extraction (password protected, scanned/no text, corrupted),
       marks status 'failed' with a detailed `failure_reason`.
    """
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        logger.warning("Extraction requested for missing document ID: %s", document_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    # 1. Mark status as processing
    registry.update_document_metadata(document_id, status=DocumentStatus.PROCESSING)

    # 2. Locate PDF file on disk
    pdf_path = (
        Path(doc.file_path)
        if doc.file_path
        else registry.upload_dir / document_id / doc.filename
    )

    if not pdf_path.exists():
        logger.error("Stored PDF file not found at %s", pdf_path)
        failure_msg = "PDF file does not exist on disk"
        registry.update_document_metadata(
            document_id,
            status=DocumentStatus.FAILED,
            failure_reason=failure_msg,
        )
        return DocumentExtractionResponse(
            document_id=document_id,
            status=DocumentStatus.FAILED,
            failure_reason=failure_msg,
            pages=[],
        )

    # 3. Perform text extraction and cleaning with edge case handling
    try:
        raw_pages = extract_raw_pages(pdf_path)
    except PDFExtractionError as exc:
        failure_msg = exc.message
        logger.warning("Extraction failed for doc %s: %s", document_id, failure_msg)
        registry.update_document_metadata(
            document_id,
            status=DocumentStatus.FAILED,
            failure_reason=failure_msg,
        )
        return DocumentExtractionResponse(
            document_id=document_id,
            status=DocumentStatus.FAILED,
            failure_reason=failure_msg,
            pages=[],
            cleaning_reports=[],
        )
    except Exception as exc:
        logger.error("Unexpected failure extracting doc %s: %s", document_id, exc)
        failure_msg = "corrupted or unreadable PDF file"
        registry.update_document_metadata(
            document_id,
            status=DocumentStatus.FAILED,
            failure_reason=failure_msg,
        )
        return DocumentExtractionResponse(
            document_id=document_id,
            status=DocumentStatus.FAILED,
            failure_reason=failure_msg,
            pages=[],
            cleaning_reports=[],
        )

    # 4. Clean and normalize extracted text
    clean_pages, cleaning_reports = clean_document_pages(raw_pages)

    # 5. Save raw and cleaned per-page results to disk
    try:
        registry.save_raw_and_cleaned_pages(document_id, raw_pages, clean_pages)
    except Exception as exc:
        logger.error("Failed to write extracted files for doc %s: %s", document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist extracted text to storage.",
        ) from exc

    # 6. Mark ready and record page/char counts
    total_chars = sum(p.char_count for p in clean_pages)
    page_count = len(clean_pages)
    registry.update_document_metadata(
        document_id,
        status=DocumentStatus.READY,
        page_count=page_count,
        total_char_count=total_chars,
        failure_reason=None,
    )

    logger.info(
        "Document %s extraction and cleaning completed: %d pages, %d chars.",
        document_id,
        page_count,
        total_chars,
    )
    return DocumentExtractionResponse(
        document_id=document_id,
        status=DocumentStatus.READY,
        page_count=page_count,
        total_char_count=total_chars,
        failure_reason=None,
        pages=clean_pages,
        cleaning_reports=cleaning_reports,
    )


@router.get(
    "/{document_id}/pages",
    response_model=list[PageText],
    summary="Get extracted pages for a document",
)
@router.get(
    "/documents/{document_id}/pages",
    response_model=list[PageText],
    include_in_schema=False,
)
async def get_document_pages(document_id: str) -> list[PageText]:
    """Retrieve the extracted per-page text list for an ingested document."""
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    pages = registry.get_extracted_pages(document_id)
    if pages is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No extracted text found for document '{document_id}'. "
                f"Status is '{doc.status.value}'."
            ),
        )
    return pages


@router.post(
    "/{document_id}/chunk",
    response_model=ChunkListResponse,
    summary="Split document text into semantic chunks with page tracking",
)
@router.post(
    "/documents/{document_id}/chunk",
    response_model=ChunkListResponse,
    include_in_schema=False,
)
async def chunk_document_endpoint(document_id: str) -> ChunkListResponse:
    """Split cleaned extracted text into token-bounded, overlapping chunks with position tracking.

    - Splits text into ~500 token chunks with ~50 token overlap using recursive separators.
    - Preserves exact page numbers (including multi-page spanning: page_number -> page_number_end).
    - Preserves exact character offsets for source citation grounding.
    - Merges tiny trailing chunks (< 20 tokens) into neighboring chunks.
    - Saves output to `uploads/{document_id}/chunks.json` and updates document metadata.
    """
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        logger.warning("Chunking requested for nonexistent document ID: %s", document_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    pages = registry.get_extracted_pages(document_id)
    if not pages:
        logger.warning(
            "Chunking requested for document %s before text extraction.", document_id
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Document '{document_id}' has not been extracted or has no text. "
                f"Please run POST /api/ingest/{document_id}/extract first."
            ),
        )

    # 1. Run recursive chunking with page and character tracking
    chunks = chunk_document(document_id=document_id, pages=pages)

    # 2. Persist chunks to uploads/{document_id}/chunks.json
    try:
        registry.save_chunks(document_id, chunks)
    except Exception as exc:
        logger.error("Failed to save chunks.json for doc %s: %s", document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist chunks to disk storage.",
        ) from exc

    # 3. Update document registry metadata with chunk_count
    registry.update_document_metadata(document_id, chunk_count=len(chunks))

    logger.info(
        "Successfully chunked document %s into %d chunks.",
        document_id,
        len(chunks),
    )
    return ChunkListResponse(
        document_id=document_id,
        chunk_count=len(chunks),
        chunks=chunks,
    )


@router.get(
    "/{document_id}/chunks",
    response_model=ChunkListResponse,
    summary="Get all generated chunks for a document",
)
@router.get(
    "/documents/{document_id}/chunks",
    response_model=ChunkListResponse,
    include_in_schema=False,
)
async def get_document_chunks_endpoint(document_id: str) -> ChunkListResponse:
    """Retrieve the generated chunks for a document from storage."""
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    chunks = registry.get_chunks(document_id)
    if chunks is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No chunks found for document '{document_id}'. "
                f"Run POST /api/ingest/{document_id}/chunk first."
            ),
        )
    return ChunkListResponse(
        document_id=document_id,
        chunk_count=len(chunks),
        chunks=chunks,
    )


@router.post(
    "/{document_id}/process",
    response_model=IngestionResult,
    status_code=status.HTTP_200_OK,
    summary="Trigger end-to-end ingestion pipeline (extract -> clean -> chunk -> embed)",
)
@router.post(
    "/documents/{document_id}/process",
    response_model=IngestionResult,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
async def process_document_endpoint(
    document_id: str,
    background_tasks: BackgroundTasks,
    sync: bool = False,
) -> IngestionResult:
    """Run the complete document ingestion pipeline in a single call.

    Pipeline stages:
    1. Extract per-page text from stored PDF (Phase 7)
    2. Clean and normalize extracted text (Phase 8)
    3. Split text into token-bounded chunks (Phase 9)
    4. Generate local dense vector embeddings (Phase 11)
    5. Attach embeddings to chunks and persist to disk (Phase 12)

    Concurrency Model:
    - By default (`sync=False`), processing runs in the background using FastAPI
      BackgroundTasks + worker thread, returning immediately with status 'processing'.
    - Clients can poll `GET /api/ingest/{document_id}/status` to track progress.
    - If `sync=True`, blocks and returns the completed IngestionResult synchronously.
    """
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        logger.warning("Process requested for nonexistent document ID: %s", document_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    if sync:
        logger.info("Executing synchronous ingestion pipeline for document %s...", document_id)
        result = run_ingestion_pipeline(document_id)
        return result

    # Update status to processing immediately so initial poll sees 'processing'
    registry.update_document_metadata(
        document_id,
        status=DocumentStatus.PROCESSING,
        current_stage="extracting",
        failure_reason=None,
    )

    def _execute_in_background(doc_id: str) -> None:
        worker = threading.Thread(
            target=run_ingestion_pipeline,
            args=(doc_id,),
            daemon=True,
            name=f"ingestion-pipeline-{doc_id}",
        )
        worker.start()

    background_tasks.add_task(_execute_in_background, document_id)

    return IngestionResult(
        document_id=document_id,
        status=DocumentStatus.PROCESSING,
        current_stage="extracting",
        chunk_count=0,
        total_tokens=0,
        processing_time_seconds=0.0,
        failure_reason=None,
        stage_timings={},
    )


@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
    summary="Get document ingestion status, current stage, and progress metrics",
)
@router.get(
    "/documents/{document_id}/status",
    response_model=DocumentStatusResponse,
    include_in_schema=False,
)
async def get_document_status_endpoint(document_id: str) -> DocumentStatusResponse:
    """Retrieve the current ingestion status, stage progression, and metrics for a document.

    Used by the frontend to poll during processing and display live progress:
    uploaded -> extracting -> cleaning -> chunking -> embedding -> ready (or failed).
    """
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        logger.warning("Status requested for nonexistent document ID: %s", document_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    return DocumentStatusResponse(
        document_id=doc.document_id,
        status=doc.status,
        current_stage=doc.current_stage,
        failure_reason=doc.failure_reason,
        chunk_count=doc.chunk_count,
        upserted_count=getattr(doc, "upserted_count", None),
        page_count=doc.page_count,
        total_tokens=doc.total_tokens,
        total_char_count=doc.total_char_count,
        processing_time_seconds=doc.processing_time_seconds,
        stage_timings=doc.stage_timings,
    )


@router.get(
    "/{document_id}/verify",
    response_model=DocumentVerificationResponse,
    summary="Verify document vectors in Pinecone index against local chunk text",
)
@router.get(
    "/documents/{document_id}/verify",
    response_model=DocumentVerificationResponse,
    include_in_schema=False,
)
async def verify_document_endpoint(document_id: str) -> DocumentVerificationResponse:
    """Trigger on-demand post-upsert verification for a document.

    Picks 2-3 sample chunks from disk, fetches them back from Pinecone by vector ID,
    and confirms that the stored source_text metadata matches the original chunk content.
    """
    registry = get_document_registry()
    doc = registry.get_document(document_id)
    if not doc:
        logger.warning("Verification requested for nonexistent document ID: %s", document_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    verification_result = verify_document_upsert(
        document_id=document_id,
        registry=registry,
    )
    return verification_result




