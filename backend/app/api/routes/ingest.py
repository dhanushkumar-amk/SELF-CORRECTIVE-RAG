"""
Document ingestion endpoints for receiving, validating, and managing uploaded PDFs.

Phase 6: File reception, magic byte validation, storage, and registry tracking.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.pdf_extractor import (
    PDFExtractionError,
    extract_raw_pages,
    extract_text_by_page,
)
from app.ingestion.storage import get_document_registry, sanitize_filename
from app.ingestion.text_cleaner import clean_document_pages
from app.models.schemas import (
    DocumentExtractionResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentStatus,
    DocumentUploadResponse,
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

