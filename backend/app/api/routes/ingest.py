"""
Document ingestion endpoints for receiving, validating, and managing uploaded PDFs.

Phase 6: File reception, magic byte validation, storage, and registry tracking.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.storage import get_document_registry, sanitize_filename
from app.models.schemas import (
    DocumentListResponse,
    DocumentMetadata,
    DocumentStatus,
    DocumentUploadResponse,
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
