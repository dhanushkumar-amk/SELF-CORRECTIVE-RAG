"""
Document ingestion pipeline orchestrator.

Chains the full document processing flow into a single unified execution:
1. Load stored PDF from disk (Phase 6)
2. Extract text by page using pypdf (Phase 7)
3. Clean and normalize per-page text (Phase 8)
4. Recursively split text into token-bounded chunks (Phase 9)
5. Generate dense vector embeddings via local sentence-transformers (Phase 11)
6. Attach embeddings to Chunk objects and persist to disk (Phase 12)
7. Format metadata, batch upsert to Pinecone with exponential backoff & verify (Phase 13)
8. Pipeline-level resilience, tenacity retries, transient/permanent error taxonomy & dead-letter handling (Phase 14)

Resilience & Error Taxonomy:
- Retries transient errors (Pinecone network blips, timeouts, 429 rate limits) up to 3 times (0s, 2s, 8s backoff) via tenacity.
- Immediately halts execution without retrying on permanent errors (corrupted PDF, password protection, scanned PDF, missing file).
- Records `retryable: bool` on dead-letter failed documents in registry.json.
- Logs structured JSON error payloads for observability.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.chunk_metadata import create_vector_id
from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import embed_texts
from app.ingestion.exceptions import (
    IngestionPipelineError,
    PermanentIngestionError,
    TransientIngestionError,
)
from app.ingestion.pdf_extractor import PDFExtractionError, extract_raw_pages
from app.ingestion.storage import DocumentRegistry, get_document_registry
from app.ingestion.text_cleaner import clean_document_pages
from app.models.api_models import (
    DocumentVerificationResponse,
    IngestionResult,
)
from app.models.schemas import (
    Chunk,
    DocumentStatus,
)
from app.retrieval.pinecone_client import (
    PineconeBatchUpsertError,
    PineconeClient,
    get_pinecone_client,
)

logger = get_logger(__name__)

__all__ = ["run_ingestion_pipeline", "verify_document_upsert"]


def _log_retry_attempt(retry_state: Any) -> None:
    """Tenacity callback triggered before sleeping between retry attempts."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    attempt = retry_state.attempt_number
    # Extract document_id from positional args if present
    doc_id = "unknown"
    if retry_state.args:
        doc_id = str(retry_state.args[0])

    stage = getattr(exc, "stage", "pipeline") if exc else "pipeline"
    error_type = exc.__class__.__name__ if exc else "UnknownError"
    error_msg = str(exc) if exc else "Transient error encountered"

    log_payload = {
        "event": "ingestion_pipeline_retry",
        "document_id": doc_id,
        "stage": stage,
        "error_type": error_type,
        "error_message": error_msg,
        "attempt_number": attempt,
        "retryable": True,
    }
    logger.warning("Pipeline transient error (attempt %d/3), retrying... %s", attempt, json.dumps(log_payload))


@retry(
    retry=retry_if_exception_type(TransientIngestionError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    before_sleep=_log_retry_attempt,
    reraise=True,
)
def _execute_pipeline_stages(
    document_id: str,
    reg: DocumentRegistry,
    pinecone_client: PineconeClient | None = None,
    verify_upsert: bool = True,
) -> IngestionResult:
    """Internal pipeline execution function wrapped with Tenacity retry logic.

    Only retries when TransientIngestionError is raised. PermanentIngestionError causes
    immediate termination without retry.
    """
    settings = get_settings()
    stage_timings: dict[str, float] = {}
    overall_start = time.perf_counter()

    # -------------------------------------------------------------------------
    # Stage 0: Pre-flight Validation
    # -------------------------------------------------------------------------
    doc = reg.get_document(document_id)
    if not doc:
        raise PermanentIngestionError(
            f"Document with ID '{document_id}' not found in registry.",
            stage="preflight",
        )

    pdf_path = (
        Path(doc.file_path)
        if doc.file_path
        else reg.upload_dir / document_id / doc.filename
    )
    if not pdf_path.exists():
        raise PermanentIngestionError(
            "failed at extraction: PDF file does not exist on disk",
            stage="extracting",
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
            "Stage 1/5 [Extracting] completed for doc '%s' in %.4fs (%d pages extracted)",
            document_id,
            t_extract,
            len(raw_pages),
        )
    except PDFExtractionError as exc:
        t_extract = time.perf_counter() - t_extract_start
        stage_timings["extraction_seconds"] = round(t_extract, 4)
        raise PermanentIngestionError(
            f"failed at extraction: {exc.message}",
            stage="extracting",
        ) from exc
    except (TimeoutError, ConnectionError, OSError) as exc:
        t_extract = time.perf_counter() - t_extract_start
        stage_timings["extraction_seconds"] = round(t_extract, 4)
        raise TransientIngestionError(
            f"failed at extraction: {exc}",
            stage="extracting",
        ) from exc
    except Exception as exc:
        t_extract = time.perf_counter() - t_extract_start
        stage_timings["extraction_seconds"] = round(t_extract, 4)
        raise PermanentIngestionError(
            f"failed at extraction: {exc}",
            stage="extracting",
        ) from exc

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
            "Stage 2/5 [Cleaning] completed for doc '%s' in %.4fs (%d chars retained across %d pages)",
            document_id,
            t_clean,
            total_chars,
            len(clean_pages),
        )
    except Exception as exc:
        t_clean = time.perf_counter() - t_clean_start
        stage_timings["cleaning_seconds"] = round(t_clean, 4)
        raise PermanentIngestionError(
            f"failed at cleaning: {exc}",
            stage="cleaning",
        ) from exc

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
            raise PermanentIngestionError(
                "failed at chunking: No text chunks generated from cleaned document pages",
                stage="chunking",
            )

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
            "Stage 3/5 [Chunking] completed for doc '%s' in %.4fs (%d chunks, %d total tokens)",
            document_id,
            t_chunk,
            len(chunks),
            total_tokens,
        )
    except PermanentIngestionError:
        raise
    except Exception as exc:
        t_chunk = time.perf_counter() - t_chunk_start
        stage_timings["chunking_seconds"] = round(t_chunk, 4)
        raise PermanentIngestionError(
            f"failed at chunking: {exc}",
            stage="chunking",
        ) from exc

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
        t_embed = time.perf_counter() - t_embed_start
        stage_timings["embedding_seconds"] = round(t_embed, 4)

        warn_threshold = getattr(settings, "EMBEDDING_TIMEOUT_WARN_SECONDS", 300.0)
        if t_embed > warn_threshold:
            logger.warning(
                "Embedding generation for doc '%s' took %.2fs, exceeding threshold of %.2fs",
                document_id,
                t_embed,
                warn_threshold,
            )

        if len(embeddings) != len(chunks):
            raise PermanentIngestionError(
                f"failed at embedding: Vector count mismatch ({len(embeddings)} vs {len(chunks)})",
                stage="embedding",
            )

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        reg.save_chunks(document_id, chunks)
        logger.info(
            "Stage 4/5 [Embedding] completed for doc '%s' in %.4fs (%d vectors generated)",
            document_id,
            t_embed,
            len(embeddings),
        )
    except PermanentIngestionError:
        raise
    except (TimeoutError, ConnectionError, OSError) as exc:
        t_embed = time.perf_counter() - t_embed_start
        stage_timings["embedding_seconds"] = round(t_embed, 4)
        raise TransientIngestionError(
            f"failed at embedding: {exc}",
            stage="embedding",
        ) from exc
    except Exception as exc:
        t_embed = time.perf_counter() - t_embed_start
        stage_timings["embedding_seconds"] = round(t_embed, 4)
        raise PermanentIngestionError(
            f"failed at embedding: {exc}",
            stage="embedding",
        ) from exc

    # -------------------------------------------------------------------------
    # Stage 5: Pinecone Vector Upsert & Post-Upsert Verification
    # -------------------------------------------------------------------------
    reg.update_document_metadata(
        document_id,
        status=DocumentStatus.PROCESSING,
        current_stage="upserting",
    )
    t_upsert_start = time.perf_counter()
    pc_client = pinecone_client or get_pinecone_client()
    upserted_count = 0

    try:
        logger.info(
            "Enforcing idempotency: Purging existing vectors for doc '%s' from Pinecone...",
            document_id,
        )
        try:
            pc_client.delete_vectors(filter={"document_id": document_id})
        except Exception as del_exc:
            logger.warning(
                "Pre-upsert cleanup encountered exception for doc '%s': %s",
                document_id,
                del_exc,
            )

        vector_payloads: list[dict[str, Any]] = []
        for c in chunks:
            if c.embedding is None:
                raise PermanentIngestionError(
                    f"failed at upserting: Chunk '{c.chunk_id}' has no embedding attached",
                    stage="upserting",
                )
            vec_id = create_vector_id(document_id, c.chunk_id)
            meta = c.to_pinecone_metadata(
                filename=doc.filename,
                document_title=doc.filename,
            )
            vector_payloads.append({
                "id": vec_id,
                "values": c.embedding,
                "metadata": meta,
            })

        upsert_res = pc_client.upsert_vectors(vector_payloads, batch_size=100)
        upserted_count = upsert_res.get("upserted_count", len(vector_payloads))

        if verify_upsert and chunks:
            sample_count = min(3, len(chunks))
            if sample_count == 1:
                sampled_indices = [0]
            elif sample_count == 2:
                sampled_indices = [0, len(chunks) - 1]
            else:
                sampled_indices = [0, len(chunks) // 2, len(chunks) - 1]

            sampled_chunks = [chunks[i] for i in sampled_indices[:sample_count]]
            sample_ids = [create_vector_id(document_id, sc.chunk_id) for sc in sampled_chunks]

            fetched_map = pc_client.fetch_vectors(ids=sample_ids)
            for sc, vid in zip(sampled_chunks, sample_ids):
                fetched = fetched_map.get(vid)
                if not fetched:
                    logger.warning("Post-upsert verification: Vector '%s' not yet visible via fetch.", vid)
                    continue
                fetched_text = fetched.get("metadata", {}).get("source_text", "")
                if fetched_text != sc.text:
                    logger.warning(
                        "Post-upsert verification: Content mismatch for vector '%s'! "
                        "(Expected %d chars, fetched %d chars)",
                        vid,
                        len(sc.text),
                        len(fetched_text),
                    )

        t_upsert = time.perf_counter() - t_upsert_start
        stage_timings["upserting_seconds"] = round(t_upsert, 4)
        logger.info(
            "Stage 5/5 [Upserting] completed for doc '%s' in %.4fs (%d vectors upserted to Pinecone)",
            document_id,
            t_upsert,
            upserted_count,
        )
    except PineconeBatchUpsertError as exc:
        t_upsert = time.perf_counter() - t_upsert_start
        stage_timings["upserting_seconds"] = round(t_upsert, 4)
        raise TransientIngestionError(
            f"failed at upserting: {exc.message}",
            stage="upserting",
        ) from exc
    except (TimeoutError, ConnectionError, OSError) as exc:
        t_upsert = time.perf_counter() - t_upsert_start
        stage_timings["upserting_seconds"] = round(t_upsert, 4)
        raise TransientIngestionError(
            f"failed at upserting: {exc}",
            stage="upserting",
        ) from exc
    except Exception as exc:
        t_upsert = time.perf_counter() - t_upsert_start
        stage_timings["upserting_seconds"] = round(t_upsert, 4)
        if isinstance(exc, (TransientIngestionError, PermanentIngestionError)):
            raise
        exc_name = exc.__class__.__name__.lower()
        if "pinecone" in exc_name or "timeout" in exc_name or "connection" in exc_name or "http" in exc_name:
            raise TransientIngestionError(f"failed at upserting: {exc}", stage="upserting") from exc
        else:
            raise PermanentIngestionError(f"failed at upserting: {exc}", stage="upserting") from exc

    # -------------------------------------------------------------------------
    # Stage 6: Completion & Metric Finalization
    # -------------------------------------------------------------------------
    total_time = round(time.perf_counter() - overall_start, 4)
    stage_timings["total_seconds"] = total_time
    logger.info(
        "Ingestion pipeline completed for document '%s': %d chunks, %d vectors upserted, %d total tokens in %.4fs. "
        "Stage timing breakdown: extraction=%.4fs, cleaning=%.4fs, chunking=%.4fs, embedding=%.4fs, upserting=%.4fs",
        document_id,
        len(chunks),
        upserted_count,
        total_tokens,
        total_time,
        stage_timings.get("extraction_seconds", 0.0),
        stage_timings.get("cleaning_seconds", 0.0),
        stage_timings.get("chunking_seconds", 0.0),
        stage_timings.get("embedding_seconds", 0.0),
        stage_timings.get("upserting_seconds", 0.0),
    )

    reg.update_document_metadata(
        document_id,
        status=DocumentStatus.READY,
        current_stage="ready",
        failure_reason=None,
        retryable=None,
        chunk_count=len(chunks),
        upserted_count=upserted_count,
        total_tokens=total_tokens,
        processing_time_seconds=total_time,
        stage_timings=stage_timings,
    )

    # Rebuild in-memory BM25 index to immediately incorporate new chunks
    try:
        from app.retrieval.bm25_index import rebuild_bm25_index
        rebuild_bm25_index(reg)
    except Exception as bm25_exc:
        logger.warning(
            "BM25 index rebuild failed post-ingestion for doc '%s': %s",
            document_id,
            bm25_exc,
        )

    return IngestionResult(
        document_id=document_id,
        status=DocumentStatus.READY,
        current_stage="ready",
        chunk_count=len(chunks),
        upserted_count=upserted_count,
        total_tokens=total_tokens,
        processing_time_seconds=total_time,
        failure_reason=None,
        retryable=None,
        stage_timings=stage_timings,
    )


def run_ingestion_pipeline(
    document_id: str,
    registry: DocumentRegistry | None = None,
    pinecone_client: PineconeClient | None = None,
    verify_upsert: bool = True,
) -> IngestionResult:
    """Execute the complete document ingestion pipeline with resilience and retries.

    Args:
        document_id: UUID of the uploaded document in the registry.
        registry: Optional custom DocumentRegistry instance (defaults to global singleton).
        pinecone_client: Optional custom PineconeClient instance (defaults to global singleton).
        verify_upsert: If True, fetches sampled vectors from Pinecone post-upsert to verify text integrity.

    Returns:
        IngestionResult detailing final status, chunk counts, token totals,
        per-stage timing breakdown, retry eligibility, and failure reason.
    """
    reg = registry or get_document_registry()

    try:
        return _execute_pipeline_stages(
            document_id=document_id,
            reg=reg,
            pinecone_client=pinecone_client,
            verify_upsert=verify_upsert,
        )
    except PermanentIngestionError as exc:
        stage = exc.stage or "unknown"
        failure_reason = str(exc.message)
        log_payload = {
            "event": "ingestion_permanent_failure",
            "document_id": document_id,
            "stage": stage,
            "error_type": exc.__class__.__name__,
            "error_message": failure_reason,
            "attempt_number": 1,
            "retryable": False,
        }
        logger.error("Permanent ingestion error halted pipeline: %s", json.dumps(log_payload))

        reg.update_document_metadata(
            document_id,
            status=DocumentStatus.FAILED,
            current_stage="failed",
            failure_reason=failure_reason,
            retryable=False,
        )
        return IngestionResult(
            document_id=document_id,
            status=DocumentStatus.FAILED,
            current_stage="failed",
            chunk_count=0,
            upserted_count=0,
            total_tokens=0,
            processing_time_seconds=0.0,
            failure_reason=failure_reason,
            retryable=False,
            stage_timings={},
        )
    except TransientIngestionError as exc:
        stage = exc.stage or "unknown"
        failure_reason = str(exc.message)
        log_payload = {
            "event": "ingestion_transient_failure_exhausted",
            "document_id": document_id,
            "stage": stage,
            "error_type": exc.__class__.__name__,
            "error_message": failure_reason,
            "attempt_number": 3,
            "retryable": True,
        }
        logger.error("Transient ingestion error exhausted all retries: %s", json.dumps(log_payload))

        reg.update_document_metadata(
            document_id,
            status=DocumentStatus.FAILED,
            current_stage="failed",
            failure_reason=failure_reason,
            retryable=True,
        )
        return IngestionResult(
            document_id=document_id,
            status=DocumentStatus.FAILED,
            current_stage="failed",
            chunk_count=0,
            upserted_count=0,
            total_tokens=0,
            processing_time_seconds=0.0,
            failure_reason=failure_reason,
            retryable=True,
            stage_timings={},
        )
    except Exception as exc:
        failure_reason = f"Unexpected error during ingestion: {exc}"
        log_payload = {
            "event": "ingestion_unexpected_failure",
            "document_id": document_id,
            "stage": "unknown",
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
            "attempt_number": 1,
            "retryable": False,
        }
        logger.error("Unexpected error halted pipeline: %s", json.dumps(log_payload))

        reg.update_document_metadata(
            document_id,
            status=DocumentStatus.FAILED,
            current_stage="failed",
            failure_reason=failure_reason,
            retryable=False,
        )
        return IngestionResult(
            document_id=document_id,
            status=DocumentStatus.FAILED,
            current_stage="failed",
            chunk_count=0,
            upserted_count=0,
            total_tokens=0,
            processing_time_seconds=0.0,
            failure_reason=failure_reason,
            retryable=False,
            stage_timings={},
        )


def verify_document_upsert(
    document_id: str,
    registry: DocumentRegistry | None = None,
    pinecone_client: PineconeClient | None = None,
    sample_size: int = 3,
) -> DocumentVerificationResponse:
    """Verify that stored chunks for a document exist in Pinecone and match their source text.

    Args:
        document_id: UUID of document to verify.
        registry: Document registry instance.
        pinecone_client: PineconeClient instance.
        sample_size: Number of chunks to sample and verify (default: 3).

    Returns:
        DocumentVerificationResponse with match details and verified boolean.
    """
    reg = registry or get_document_registry()
    doc = reg.get_document(document_id)
    if not doc:
        return DocumentVerificationResponse(
            document_id=document_id,
            status=DocumentStatus.FAILED,
            verified=False,
            chunk_count=0,
            sampled_count=0,
            matched_count=0,
            mismatches=[f"Document with ID '{document_id}' not found in registry."],
            verified_chunks=[],
            message="Document not found.",
        )

    chunks = reg.get_chunks(document_id)
    if not chunks:
        return DocumentVerificationResponse(
            document_id=document_id,
            status=doc.status,
            verified=False,
            chunk_count=0,
            sampled_count=0,
            matched_count=0,
            mismatches=["No chunks found in local storage."],
            verified_chunks=[],
            message="No chunks found on disk for document.",
        )

    pc_client = pinecone_client or get_pinecone_client()
    sample_count = min(sample_size, len(chunks))

    if sample_count == 1:
        sampled_indices = [0]
    elif sample_count == 2:
        sampled_indices = [0, len(chunks) - 1]
    else:
        sampled_indices = [0, len(chunks) // 2, len(chunks) - 1]

    sampled_chunks = [chunks[i] for i in sampled_indices[:sample_count]]
    sample_ids = [create_vector_id(document_id, sc.chunk_id) for sc in sampled_chunks]

    fetched_map = pc_client.fetch_vectors(ids=sample_ids)
    matched_count = 0
    mismatches: list[str] = []
    verified_chunks: list[dict[str, Any]] = []

    for sc, vid in zip(sampled_chunks, sample_ids):
        fetched = fetched_map.get(vid)
        if not fetched:
            msg = f"Vector '{vid}' not found in Pinecone index."
            mismatches.append(msg)
            verified_chunks.append({
                "vector_id": vid,
                "chunk_id": sc.chunk_id,
                "page_number": sc.page_number,
                "matched": False,
                "error": "Vector not found",
            })
            continue

        fetched_text = fetched.get("metadata", {}).get("source_text", "")
        if fetched_text == sc.text:
            matched_count += 1
            verified_chunks.append({
                "vector_id": vid,
                "chunk_id": sc.chunk_id,
                "page_number": sc.page_number,
                "matched": True,
                "char_length": len(sc.text),
            })
        else:
            msg = (
                f"Vector '{vid}' content mismatch: expected {len(sc.text)} chars, "
                f"found {len(fetched_text)} chars in Pinecone."
            )
            mismatches.append(msg)
            verified_chunks.append({
                "vector_id": vid,
                "chunk_id": sc.chunk_id,
                "page_number": sc.page_number,
                "matched": False,
                "error": "source_text mismatch",
            })

    is_verified = (matched_count == sample_count and len(mismatches) == 0)
    summary_message = (
        f"Pinecone verification {'successful' if is_verified else 'failed'}. "
        f"{matched_count}/{sample_count} sampled chunks match on-disk source text."
    )

    return DocumentVerificationResponse(
        document_id=document_id,
        status=doc.status,
        verified=is_verified,
        chunk_count=len(chunks),
        sampled_count=sample_count,
        matched_count=matched_count,
        mismatches=mismatches,
        verified_chunks=verified_chunks,
        message=summary_message,
    )
