"""
Document ingestion pipeline orchestrator.

Chains the full document processing flow into a single unified execution:
1. Load stored PDF from disk (Phase 6)
2. Extract text by page using pypdf (Phase 7)
3. Clean and normalize per-page text (Phase 8)
4. Recursively split text into token-bounded chunks (Phase 9)
5. Generate dense vector embeddings via local sentence-transformers (Phase 11)
6. Attach embeddings to Chunk objects and persist to disk (Phase 12)

Stage Tracking & Observability:
- Updates registry.json at each stage transition:
  uploaded -> extracting -> cleaning -> chunking -> embedding -> ready (or failed)
- Records granular per-stage latency timings for performance monitoring and bottleneck profiling.
- Enforces strict failure isolation: an exception at any stage immediately halts execution,
  attributes the failure to that specific stage in failure_reason, and prevents wasted compute.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from app.core.logging import get_logger
from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import embed_texts
from app.ingestion.pdf_extractor import PDFExtractionError, extract_raw_pages
from app.ingestion.storage import DocumentRegistry, get_document_registry
from app.ingestion.text_cleaner import clean_document_pages
from app.models.schemas import Chunk, DocumentStatus, IngestionResult

logger = get_logger(__name__)

__all__ = ["run_ingestion_pipeline"]


def run_ingestion_pipeline(
    document_id: str,
    registry: DocumentRegistry | None = None,
) -> IngestionResult:
    """Execute the complete document ingestion pipeline for a given document_id.

    Args:
        document_id: UUID of the uploaded document in the registry.
        registry: Optional custom DocumentRegistry instance (defaults to global singleton).

    Returns:
        IngestionResult detailing final status, chunk counts, token totals,
        per-stage timing breakdown, and any failure reason.
    """
    reg = registry or get_document_registry()
    stage_timings: dict[str, float] = {}
    overall_start = time.perf_counter()

    # -------------------------------------------------------------------------
    # Stage 0: Pre-flight Validation
    # -------------------------------------------------------------------------
    doc = reg.get_document(document_id)
    if not doc:
        logger.error("Ingestion pipeline failed: Document '%s' not found in registry.", document_id)
        failure_reason = f"Document with ID '{document_id}' not found in registry."
        return IngestionResult(
            document_id=document_id,
            status=DocumentStatus.FAILED,
            current_stage="failed",
            chunk_count=0,
            total_tokens=0,
            processing_time_seconds=0.0,
            failure_reason=failure_reason,
            stage_timings={},
        )

    pdf_path = (
        Path(doc.file_path)
        if doc.file_path
        else reg.upload_dir / document_id / doc.filename
    )
    if not pdf_path.exists():
        logger.error("Ingestion pipeline failed: PDF file missing at '%s'", pdf_path)
        failure_reason = "failed at extraction: PDF file does not exist on disk"
        reg.update_document_metadata(
            document_id,
            status=DocumentStatus.FAILED,
            current_stage="failed",
            failure_reason=failure_reason,
        )
        return IngestionResult(
            document_id=document_id,
            status=DocumentStatus.FAILED,
            current_stage="failed",
            chunk_count=0,
            total_tokens=0,
            processing_time_seconds=0.0,
            failure_reason=failure_reason,
            stage_timings={},
        )

    # -------------------------------------------------------------------------
    # Stage 1: Extraction
    # -------------------------------------------------------------------------
    reg.update_document_metadata(
        document_id,
        status=DocumentStatus.PROCESSING,
        current_stage="extracting",
        failure_reason=None,
    )
    t_extract_start = time.perf_counter()
    try:
        raw_pages = extract_raw_pages(pdf_path)
        t_extract = time.perf_counter() - t_extract_start
        stage_timings["extraction_seconds"] = round(t_extract, 4)
        logger.info(
            "Stage 1/4 [Extracting] completed for doc '%s' in %.4fs (%d pages extracted)",
            document_id,
            t_extract,
            len(raw_pages),
        )
    except PDFExtractionError as exc:
        t_extract = time.perf_counter() - t_extract_start
        stage_timings["extraction_seconds"] = round(t_extract, 4)
        total_time = round(time.perf_counter() - overall_start, 4)
        stage_timings["total_seconds"] = total_time
        failure_reason = f"failed at extraction: {exc.message}"
        logger.warning("Ingestion halted: %s for doc '%s'", failure_reason, document_id)
        reg.update_document_metadata(
            document_id,
            status=DocumentStatus.FAILED,
            current_stage="failed",
            failure_reason=failure_reason,
            stage_timings=stage_timings,
            processing_time_seconds=total_time,
        )
        return IngestionResult(
            document_id=document_id,
            status=DocumentStatus.FAILED,
            current_stage="failed",
            chunk_count=0,
            total_tokens=0,
            processing_time_seconds=total_time,
            failure_reason=failure_reason,
            stage_timings=stage_timings,
        )
    except Exception as exc:
        t_extract = time.perf_counter() - t_extract_start
        stage_timings["extraction_seconds"] = round(t_extract, 4)
        total_time = round(time.perf_counter() - overall_start, 4)
        stage_timings["total_seconds"] = total_time
        failure_reason = f"failed at extraction: {exc}"
        logger.error("Unexpected error in extraction for doc '%s': %s", document_id, exc)
        reg.update_document_metadata(
            document_id,
            status=DocumentStatus.FAILED,
            current_stage="failed",
            failure_reason=failure_reason,
            stage_timings=stage_timings,
            processing_time_seconds=total_time,
        )
        return IngestionResult(
            document_id=document_id,
            status=DocumentStatus.FAILED,
            current_stage="failed",
            chunk_count=0,
            total_tokens=0,
            processing_time_seconds=total_time,
            failure_reason=failure_reason,
            stage_timings=stage_timings,
        )

    # -------------------------------------------------------------------------
    # Stage 2: Cleaning
    # -------------------------------------------------------------------------
    reg.update_document_metadata(
        document_id,
        status=DocumentStatus.PROCESSING,
        current_stage="cleaning",
    )
    t_clean_start = time.perf_counter()
    try:
        clean_pages, _ = clean_document_pages(raw_pages)
        reg.save_raw_and_cleaned_pages(document_id, raw_pages, clean_pages)
        total_chars = sum(p.char_count for p in clean_pages)
        reg.update_document_metadata(
            document_id,
            page_count=len(clean_pages),
            total_char_count=total_chars,
        )
        t_clean = time.perf_counter() - t_clean_start
        stage_timings["cleaning_seconds"] = round(t_clean, 4)
        logger.info(
            "Stage 2/4 [Cleaning] completed for doc '%s' in %.4fs (%d chars retained across %d pages)",
            document_id,
            t_clean,
            total_chars,
            len(clean_pages),
        )
    except Exception as exc:
        t_clean = time.perf_counter() - t_clean_start
        stage_timings["cleaning_seconds"] = round(t_clean, 4)
        total_time = round(time.perf_counter() - overall_start, 4)
        stage_timings["total_seconds"] = total_time
        failure_reason = f"failed at cleaning: {exc}"
        logger.error("Ingestion halted: %s for doc '%s'", failure_reason, document_id)
        reg.update_document_metadata(
            document_id,
            status=DocumentStatus.FAILED,
            current_stage="failed",
            failure_reason=failure_reason,
            stage_timings=stage_timings,
            processing_time_seconds=total_time,
        )
        return IngestionResult(
            document_id=document_id,
            status=DocumentStatus.FAILED,
            current_stage="failed",
            chunk_count=0,
            total_tokens=0,
            processing_time_seconds=total_time,
            failure_reason=failure_reason,
            stage_timings=stage_timings,
        )

    # -------------------------------------------------------------------------
    # Stage 3: Chunking
    # -------------------------------------------------------------------------
    reg.update_document_metadata(
        document_id,
        status=DocumentStatus.PROCESSING,
        current_stage="chunking",
    )
    t_chunk_start = time.perf_counter()
    try:
        chunks = chunk_document(document_id=document_id, pages=clean_pages)
        if not chunks:
            raise ValueError("No text chunks generated from cleaned document pages")

        reg.save_chunks(document_id, chunks)
        total_tokens = sum(c.token_count for c in chunks)
        reg.update_document_metadata(
            document_id,
            chunk_count=len(chunks),
            total_tokens=total_tokens,
        )
        t_chunk = time.perf_counter() - t_chunk_start
        stage_timings["chunking_seconds"] = round(t_chunk, 4)
        logger.info(
            "Stage 3/4 [Chunking] completed for doc '%s' in %.4fs (%d chunks, %d total tokens)",
            document_id,
            t_chunk,
            len(chunks),
            total_tokens,
        )
    except Exception as exc:
        t_chunk = time.perf_counter() - t_chunk_start
        stage_timings["chunking_seconds"] = round(t_chunk, 4)
        total_time = round(time.perf_counter() - overall_start, 4)
        stage_timings["total_seconds"] = total_time
        failure_reason = f"failed at chunking: {exc}"
        logger.error("Ingestion halted: %s for doc '%s'", failure_reason, document_id)
        reg.update_document_metadata(
            document_id,
            status=DocumentStatus.FAILED,
            current_stage="failed",
            failure_reason=failure_reason,
            stage_timings=stage_timings,
            processing_time_seconds=total_time,
        )
        return IngestionResult(
            document_id=document_id,
            status=DocumentStatus.FAILED,
            current_stage="failed",
            chunk_count=0,
            total_tokens=0,
            processing_time_seconds=total_time,
            failure_reason=failure_reason,
            stage_timings=stage_timings,
        )

    # -------------------------------------------------------------------------
    # Stage 4: Embedding Generation
    # -------------------------------------------------------------------------
    reg.update_document_metadata(
        document_id,
        status=DocumentStatus.PROCESSING,
        current_stage="embedding",
    )
    t_embed_start = time.perf_counter()
    try:
        chunk_texts = [c.text for c in chunks]
        embeddings = embed_texts(chunk_texts)
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedding vector count mismatch: generated {len(embeddings)} vectors for {len(chunks)} chunks"
            )

        # Attach dense vector embedding to each Chunk object
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # Persist chunks with embedded vectors to uploads/{document_id}/chunks.json
        reg.save_chunks(document_id, chunks)
        t_embed = time.perf_counter() - t_embed_start
        stage_timings["embedding_seconds"] = round(t_embed, 4)
        logger.info(
            "Stage 4/4 [Embedding] completed for doc '%s' in %.4fs (%d vectors generated, dim: %d)",
            document_id,
            t_embed,
            len(embeddings),
            len(embeddings[0]) if embeddings else 0,
        )
    except Exception as exc:
        t_embed = time.perf_counter() - t_embed_start
        stage_timings["embedding_seconds"] = round(t_embed, 4)
        total_time = round(time.perf_counter() - overall_start, 4)
        stage_timings["total_seconds"] = total_time
        failure_reason = f"failed at embedding: {exc}"
        logger.error("Ingestion halted: %s for doc '%s'", failure_reason, document_id)
        reg.update_document_metadata(
            document_id,
            status=DocumentStatus.FAILED,
            current_stage="failed",
            failure_reason=failure_reason,
            stage_timings=stage_timings,
            processing_time_seconds=total_time,
        )
        return IngestionResult(
            document_id=document_id,
            status=DocumentStatus.FAILED,
            current_stage="failed",
            chunk_count=len(chunks),
            total_tokens=total_tokens,
            processing_time_seconds=total_time,
            failure_reason=failure_reason,
            stage_timings=stage_timings,
        )

    # -------------------------------------------------------------------------
    # Stage 5: Completion & Metric Finalization
    # -------------------------------------------------------------------------
    total_time = round(time.perf_counter() - overall_start, 4)
    stage_timings["total_seconds"] = total_time

    reg.update_document_metadata(
        document_id,
        status=DocumentStatus.READY,
        current_stage="ready",
        failure_reason=None,
        chunk_count=len(chunks),
        total_tokens=total_tokens,
        processing_time_seconds=total_time,
        stage_timings=stage_timings,
    )

    logger.info(
        "Ingestion pipeline completed for document '%s': %d chunks, %d total tokens in %.4fs. "
        "Stage timing breakdown: extraction=%.4fs, cleaning=%.4fs, chunking=%.4fs, embedding=%.4fs",
        document_id,
        len(chunks),
        total_tokens,
        total_time,
        stage_timings["extraction_seconds"],
        stage_timings["cleaning_seconds"],
        stage_timings["chunking_seconds"],
        stage_timings["embedding_seconds"],
    )

    return IngestionResult(
        document_id=document_id,
        status=DocumentStatus.READY,
        current_stage="ready",
        chunk_count=len(chunks),
        total_tokens=total_tokens,
        processing_time_seconds=total_time,
        failure_reason=None,
        stage_timings=stage_timings,
    )
