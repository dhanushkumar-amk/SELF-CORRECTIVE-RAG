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

Stage Tracking & Observability:
- Updates registry.json at each stage transition:
  uploaded -> extracting -> cleaning -> chunking -> embedding -> upserting -> ready (or failed)
- Records granular per-stage latency timings for performance monitoring and bottleneck profiling.
- Enforces strict failure isolation: an exception at any stage immediately halts execution,
  attributes the failure to that specific stage in failure_reason, and prevents wasted compute.
- Enforces idempotency by purging pre-existing document vectors before upserting new chunks.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from app.core.logging import get_logger
from app.ingestion.chunk_metadata import create_vector_id
from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import embed_texts
from app.ingestion.pdf_extractor import PDFExtractionError, extract_raw_pages
from app.ingestion.storage import DocumentRegistry, get_document_registry
from app.ingestion.text_cleaner import clean_document_pages
from app.models.schemas import (
    Chunk,
    DocumentStatus,
    DocumentVerificationResponse,
    IngestionResult,
)
from app.retrieval import PineconeBatchUpsertError, PineconeClient, get_pinecone_client

logger = get_logger(__name__)

__all__ = ["run_ingestion_pipeline", "verify_document_upsert"]


def run_ingestion_pipeline(
    document_id: str,
    registry: DocumentRegistry | None = None,
    pinecone_client: PineconeClient | None = None,
    verify_upsert: bool = True,
) -> IngestionResult:
    """Execute the complete document ingestion pipeline for a given document_id.

    Args:
        document_id: UUID of the uploaded document in the registry.
        registry: Optional custom DocumentRegistry instance (defaults to global singleton).
        pinecone_client: Optional custom PineconeClient instance (defaults to global singleton).
        verify_upsert: If True, fetches sampled vectors from Pinecone post-upsert to verify text integrity.

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
            upserted_count=0,
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
            upserted_count=0,
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
            "Stage 1/5 [Extracting] completed for doc '%s' in %.4fs (%d pages extracted)",
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
            upserted_count=0,
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
            upserted_count=0,
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
            "Stage 2/5 [Cleaning] completed for doc '%s' in %.4fs (%d chars retained across %d pages)",
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
            upserted_count=0,
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
            "Stage 3/5 [Chunking] completed for doc '%s' in %.4fs (%d chunks, %d total tokens)",
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
            upserted_count=0,
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
            "Stage 4/5 [Embedding] completed for doc '%s' in %.4fs (%d vectors generated, dim: %d)",
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
            upserted_count=0,
            total_tokens=total_tokens,
            processing_time_seconds=total_time,
            failure_reason=failure_reason,
            stage_timings=stage_timings,
        )

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
        # 1. Idempotency Check: Purge any pre-existing vectors for this document_id
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

        # 2. Build Pinecone vector payloads with metadata
        vector_payloads: list[dict[str, Any]] = []
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"Chunk '{c.chunk_id}' has no embedding attached.")
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

        # 3. Batched Upsert with Exponential Backoff (100 vectors/batch)
        upsert_res = pc_client.upsert_vectors(vector_payloads, batch_size=100)
        upserted_count = upsert_res.get("upserted_count", len(vector_payloads))

        # 4. Post-Upsert Verification: Sample 2-3 chunks and confirm source_text matches
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
                else:
                    logger.debug("Post-upsert verification: Vector '%s' content matches source text.", vid)

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
        total_time = round(time.perf_counter() - overall_start, 4)
        stage_timings["total_seconds"] = total_time
        failure_reason = f"failed at upserting: {exc.message}"
        logger.error("Ingestion halted: %s for doc '%s'", failure_reason, document_id)
        reg.update_document_metadata(
            document_id,
            status=DocumentStatus.FAILED,
            current_stage="failed",
            failure_reason=failure_reason,
            stage_timings=stage_timings,
            processing_time_seconds=total_time,
            upserted_count=len(exc.successful_ids),
        )
        return IngestionResult(
            document_id=document_id,
            status=DocumentStatus.FAILED,
            current_stage="failed",
            chunk_count=len(chunks),
            upserted_count=len(exc.successful_ids),
            total_tokens=total_tokens,
            processing_time_seconds=total_time,
            failure_reason=failure_reason,
            stage_timings=stage_timings,
        )
    except Exception as exc:
        t_upsert = time.perf_counter() - t_upsert_start
        stage_timings["upserting_seconds"] = round(t_upsert, 4)
        total_time = round(time.perf_counter() - overall_start, 4)
        stage_timings["total_seconds"] = total_time
        failure_reason = f"failed at upserting: {exc}"
        logger.error("Ingestion halted: %s for doc '%s'", failure_reason, document_id)
        reg.update_document_metadata(
            document_id,
            status=DocumentStatus.FAILED,
            current_stage="failed",
            failure_reason=failure_reason,
            stage_timings=stage_timings,
            processing_time_seconds=total_time,
            upserted_count=upserted_count,
        )
        return IngestionResult(
            document_id=document_id,
            status=DocumentStatus.FAILED,
            current_stage="failed",
            chunk_count=len(chunks),
            upserted_count=upserted_count,
            total_tokens=total_tokens,
            processing_time_seconds=total_time,
            failure_reason=failure_reason,
            stage_timings=stage_timings,
        )

    # -------------------------------------------------------------------------
    # Stage 6: Completion & Metric Finalization
    # -------------------------------------------------------------------------
    total_time = round(time.perf_counter() - overall_start, 4)
    stage_timings["total_seconds"] = total_time

    reg.update_document_metadata(
        document_id,
        status=DocumentStatus.READY,
        current_stage="ready",
        failure_reason=None,
        chunk_count=len(chunks),
        upserted_count=upserted_count,
        total_tokens=total_tokens,
        processing_time_seconds=total_time,
        stage_timings=stage_timings,
    )

    logger.info(
        "Ingestion pipeline completed for document '%s': %d chunks, %d vectors upserted, %d total tokens in %.4fs. "
        "Stage timing breakdown: extraction=%.4fs, cleaning=%.4fs, chunking=%.4fs, embedding=%.4fs, upserting=%.4fs",
        document_id,
        len(chunks),
        upserted_count,
        total_tokens,
        total_time,
        stage_timings["extraction_seconds"],
        stage_timings["cleaning_seconds"],
        stage_timings["chunking_seconds"],
        stage_timings["embedding_seconds"],
        stage_timings["upserting_seconds"],
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
        stage_timings=stage_timings,
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

    # Pick representative chunks (first, middle, last)
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
