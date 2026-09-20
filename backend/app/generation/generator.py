"""
Structured RAG Answer Generator with Defensive JSON Parsing, Corrective Retries, and Fallback Handling.

Architecture & Design Decisions:
1. Citation-Forced Generation:
   Combines query and reranked context chunks into a grounded citation prompt, invoking the LLM provider layer
   (Groq primary -> Gemini fallback).

2. Formal Output Parsing & Validation:
   Delegates output sanitization, parsing, schema validation, and citation chunk ID verification
   to `app.generation.output_parser.parse_llm_output()`.

3. Corrective Prompt Retry Loop (Phase 29):
   When `parse_llm_output()` detects a malformed response (ParseError), retries up to 2 times on the primary provider
   with a targeted corrective prompt appending the specific error details.

4. Secondary Provider Fallback:
   If the primary provider exhausts all attempts without valid JSON output, attempts 1 generation on the fallback provider.

5. Terminal Error Guarantee:
   If all attempts across primary retries and fallback fail, raises `GenerationError` with a clean user-facing message.
"""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.generation.llm_client import generate_answer
from app.generation.output_parser import parse_llm_output, sanitize_json_string
from app.generation.prompts import CITATION_SYSTEM_PROMPT, build_citation_user_prompt
from app.models.schemas import Claim, GeneratedAnswer, GenerationError, ParseError, RetrievalResult

logger = get_logger(__name__)

__all__ = [
    "clean_json_output",
    "generate_answer_with_citations",
]


def clean_json_output(raw_text: str) -> str:
    """Backward-compatible alias for sanitize_json_string.

    Args:
        raw_text: Raw string returned by LLM provider.

    Returns:
        Clean JSON substring string.
    """
    return sanitize_json_string(raw_text)


def generate_answer_with_citations(
    query: str,
    chunks: list[RetrievalResult],
    system_prompt: str | None = None,
    force_provider: str | None = None,
    temperature: float = 0.0,
    max_retries: int = 2,
) -> GeneratedAnswer:
    """Generate a structured, citation-anchored answer from query and context chunks.

    Retries with corrective feedback up to `max_retries` on primary provider, falls back to
    secondary provider if primary fails, and raises `GenerationError` if all attempts fail.

    Args:
        query: User input query.
        chunks: List of relevant candidate RetrievalResult chunks.
        system_prompt: Optional system prompt override.
        force_provider: Optional provider override ('groq' or 'gemini').
        temperature: Sampling temperature (0.0 for deterministic output).
        max_retries: Maximum number of corrective retries on primary provider (default 2).

    Returns:
        GeneratedAnswer containing validated claims, citation metadata, and insufficiency flags.

    Raises:
        GenerationError: If all primary retries and secondary fallback attempts fail.
    """
    if not query or not query.strip() or not chunks:
        logger.warning("Empty query or empty chunk list passed to generate_answer_with_citations.")
        return GeneratedAnswer(
            claims=[],
            insufficient_information=True,
            raw_response="No context chunks available.",
            provider=None,
            model_name=None,
            latency_ms=0.0,
        )

    sys_prompt = system_prompt or CITATION_SYSTEM_PROMPT
    base_user_prompt = build_citation_user_prompt(query, chunks)
    valid_chunk_ids = {c.chunk_id for c in chunks}

    primary_provider = force_provider or settings.LLM_PRIMARY_PROVIDER
    fallback_provider = (
        settings.LLM_FALLBACK_PROVIDER
        if (force_provider is None and settings.LLM_FALLBACK_PROVIDER != primary_provider)
        else None
    )

    curr_prompt = base_user_prompt
    last_error: ParseError | None = None

    # 1. Primary Provider Corrective Retry Loop (1 initial + up to max_retries)
    for attempt in range(1, max_retries + 2):
        llm_res = generate_answer(
            prompt=curr_prompt,
            system_prompt=sys_prompt,
            force_provider=primary_provider,
            temperature=temperature,
        )

        parsed = parse_llm_output(
            raw_text=llm_res.content,
            valid_chunk_ids=valid_chunk_ids,
            provider=llm_res.provider,
            model_name=llm_res.model_name,
            latency_ms=llm_res.latency_ms,
        )

        if isinstance(parsed, GeneratedAnswer):
            if attempt > 1:
                logger.info(
                    "Generation succeeded on primary provider '%s' after corrective retry (attempt %d).",
                    llm_res.provider,
                    attempt,
                )
            else:
                logger.info(
                    "Generation succeeded on primary provider '%s' (attempt 1).",
                    llm_res.provider,
                )
            return parsed

        # Handle ParseError
        last_error = parsed
        logger.warning(
            "Generation attempt %d on primary provider '%s' failed parsing [%s]: %s",
            attempt,
            llm_res.provider,
            parsed.error_type,
            parsed.error_message,
        )

        # Append targeted corrective instruction for subsequent retries
        if attempt < max_retries + 1:
            corrective_msg = (
                f"\n\n[CORRECTIVE INSTRUCTION]: Your previous response failed with error '{parsed.error_type}': "
                f"{parsed.error_message}. Respond with ONLY the required JSON object, with no preamble, "
                f"markdown fences, or extra text."
            )
            curr_prompt = base_user_prompt + corrective_msg

    # 2. Secondary Provider Fallback Attempt
    if fallback_provider and fallback_provider != primary_provider:
        logger.warning(
            "Primary provider '%s' exhausted all %d attempts. Attempting fallback on secondary provider '%s'...",
            primary_provider,
            max_retries + 1,
            fallback_provider,
        )
        llm_res = generate_answer(
            prompt=base_user_prompt,
            system_prompt=sys_prompt,
            force_provider=fallback_provider,
            temperature=temperature,
        )

        parsed = parse_llm_output(
            raw_text=llm_res.content,
            valid_chunk_ids=valid_chunk_ids,
            provider=llm_res.provider,
            model_name=llm_res.model_name,
            latency_ms=llm_res.latency_ms,
        )

        if isinstance(parsed, GeneratedAnswer):
            logger.info(
                "Generation succeeded after falling back to secondary provider '%s'.",
                llm_res.provider,
            )
            return parsed

        last_error = parsed
        logger.warning(
            "Fallback provider '%s' attempt failed parsing [%s]: %s",
            llm_res.provider,
            parsed.error_type,
            parsed.error_message,
        )

    # 3. Terminal Failure Execution
    logger.error(
        "All generation attempts (primary retries + fallback) failed to produce valid output. Raising GenerationError."
    )
    raise GenerationError(
        message="Unable to generate a reliable answer, please try rephrasing your question.",
        details={
            "primary_provider": primary_provider,
            "fallback_provider": fallback_provider,
            "last_error_type": last_error.error_type if last_error else "unknown",
            "last_error_message": last_error.error_message if last_error else "unknown",
            "last_raw_response": last_error.raw_text if last_error else None,
        },
    )

