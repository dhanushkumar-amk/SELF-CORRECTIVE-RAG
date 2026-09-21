"""
Partial Regeneration Module for Phase 40: Targeted Claim Re-Generation & Merging.

Architecture & Design Decisions:
1. Partial Regeneration (`regenerate_failed_claims`):
   - Rather than re-generating the entire answer from scratch (which risks invalidating
     previously verified claims), partial regeneration prompts the LLM to correct ONLY
     the specific claims that failed verification using the enriched context chunk set.

2. Positional Merging (`merge_corrected_claims`):
   - Replaces failed claims (matched by claim_id) in the original claim sequence with
     newly mapped `ClaimWithSource` objects produced from the corrected claims.
   - Claims that were already verified as correct remain untouched byte-for-byte.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.generation.llm_client import generate_answer
from app.generation.output_parser import parse_llm_output
from app.generation.prompts import (
    PARTIAL_REGENERATE_SYSTEM_PROMPT,
    build_partial_regenerate_user_prompt,
)
from app.models.schemas import (
    Claim,
    ClaimWithSource,
    GeneratedAnswer,
    RetrievalResult,
)

logger = get_logger(__name__)

__all__ = [
    "merge_corrected_claims",
    "regenerate_failed_claims",
]


def regenerate_failed_claims(
    original_query: str,
    failed_claims: list[ClaimWithSource],
    enriched_chunks: list[RetrievalResult],
) -> list[Claim]:
    """Execute targeted LLM re-generation for failed claims using enriched context.

    Args:
        original_query: Original user natural language question.
        failed_claims: List of ClaimWithSource objects that failed verification.
        enriched_chunks: Combined candidate chunks (original + targeted search results).

    Returns:
        List of newly generated Claim objects tagged with source_chunk_id citations.
    """
    if not failed_claims or not enriched_chunks:
        logger.info("regenerate_failed_claims called with empty failed_claims or enriched_chunks.")
        return []

    failed_statements = [c.claim_text for c in failed_claims]
    logger.info("Executing partial regeneration across %d failed statements...", len(failed_statements))

    user_prompt = build_partial_regenerate_user_prompt(
        query=original_query,
        failed_statements=failed_statements,
        chunks=enriched_chunks,
    )

    llm_res = generate_answer(
        prompt=user_prompt,
        system_prompt=PARTIAL_REGENERATE_SYSTEM_PROMPT,
    )

    valid_ids = {c.chunk_id for c in enriched_chunks if hasattr(c, "chunk_id")}
    parsed = parse_llm_output(
        llm_res.content,
        valid_chunk_ids=valid_ids,
        provider=llm_res.provider,
        model_name=llm_res.model_name,
        latency_ms=llm_res.latency_ms,
    )

    if isinstance(parsed, GeneratedAnswer):
        logger.info(
            "Partial regeneration complete: %d corrected claims produced.",
            len(parsed.claims),
        )
        return parsed.claims

    logger.warning("Partial regeneration output parsing failed: %s", parsed)
    return []


def merge_corrected_claims(
    original_claims: list[ClaimWithSource],
    failed_claim_ids: list[str],
    corrected_claims: list[ClaimWithSource],
) -> list[ClaimWithSource]:
    """Merge newly resolved corrected claims back into the original claim list.

    Preserves verified claims in their original positions, replaces failed claims
    (matched by claim_id) with corrected ones, and appends any overflow corrected claims.

    Args:
        original_claims: Full list of ClaimWithSource objects before re-generation.
        failed_claim_ids: List of UUID claim_ids that failed verification and were sent for correction.
        corrected_claims: Newly resolved ClaimWithSource objects produced after partial re-generation.

    Returns:
        Unified list of ClaimWithSource objects with failed claims updated.
    """
    failed_set = set(failed_claim_ids)
    merged: list[ClaimWithSource] = []

    corrected_iter = iter(corrected_claims)

    for claim in original_claims:
        if claim.claim_id in failed_set:
            try:
                new_claim = next(corrected_iter)
                merged.append(new_claim)
            except StopIteration:
                # If LLM produced fewer corrected claims than failed claims, omit the failed claim
                logger.info("Omitted uncorrectable failed claim '%s' [ID: %s]", claim.claim_text[:40], claim.claim_id)
        else:
            # Preserve already-verified claims byte-for-byte
            merged.append(claim)

    # Append any remaining corrected claims produced by the LLM
    for extra_claim in corrected_iter:
        merged.append(extra_claim)

    return merged
