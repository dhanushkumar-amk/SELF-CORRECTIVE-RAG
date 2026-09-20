"""
Formal LLM Output Parser & Defensive Semantic Validator for Phase 28.

Architecture & Design Decisions:
1. Multi-Stage Sanitization:
   - Strips markdown code fences (```json ... ```).
   - Extracts payload between outermost braces ({ ... }).
   - Removes trailing commas before closing braces/brackets (r",\\s*([\\}\\]])" -> r"\\1").

2. Failure Classification:
   - Classifies malformed responses into structured `ParseError` categories:
     - `empty_response`: Response is null or whitespace-only (UNRECOVERABLE).
     - `json_decode_error`: Syntax error in JSON structure (UNRECOVERABLE).
     - `truncated_json`: Token limit reached mid-object (UNRECOVERABLE).
     - `contradictory_response`: `insufficient_information=False` but 0 claims provided (UNRECOVERABLE).
     - `invalid_schema`: Payload is not a JSON dictionary (UNRECOVERABLE).

3. Advanced Semantic Validation & Deduplication:
   - Filters out empty claim_text strings.
   - Deduplicates identical (claim_text, source_chunk_id) pairs.
   - Quarantines hallucinated chunk IDs into `unverified_claims`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.core.logging import get_logger
from app.models.schemas import Claim, GeneratedAnswer, ParseError

logger = get_logger(__name__)

__all__ = [
    "parse_llm_output",
    "sanitize_json_string",
]


def sanitize_json_string(text: str) -> str:
    """Sanitize raw LLM response text into clean JSON substring format.

    Args:
        text: Raw response string from LLM provider.

    Returns:
        Cleaned JSON string candidate.
    """
    if not text or not text.strip():
        return ""

    cleaned = text.strip()

    # 1. Strip markdown code fences (e.g. ```json ... ``` or ``` ... ```)
    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()

    # 2. Extract JSON payload between outermost '{' and '}' if JSON object is primary
    first_brace = cleaned.find("{")
    first_bracket = cleaned.find("[")

    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        last_brace = cleaned.rfind("}")
        if last_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace : last_brace + 1]

    # 3. Strip trailing commas before closing braces/brackets (e.g. {"a": 1,})
    cleaned = re.sub(r",\s*([\}\]])", r"\1", cleaned)

    return cleaned


def parse_llm_output(
    raw_text: str,
    valid_chunk_ids: set[str],
    provider: str | None = None,
    model_name: str | None = None,
    latency_ms: float = 0.0,
) -> GeneratedAnswer | ParseError:
    """Parse, sanitize, and semantically validate raw LLM text into a GeneratedAnswer or ParseError.

    Args:
        raw_text: Raw string returned by LLM provider.
        valid_chunk_ids: Set of valid input chunk_ids provided to the LLM context.
        provider: Provider name ('groq' or 'gemini').
        model_name: Model name executed.
        latency_ms: Generation latency in ms.

    Returns:
        GeneratedAnswer on successful parse, or ParseError on failure.
    """
    # 1. Empty response check
    if not raw_text or not raw_text.strip():
        logger.warning("Parse failure: LLM returned empty or whitespace response.")
        return ParseError(
            raw_text=raw_text or "",
            error_type="empty_response",
            error_message="LLM output is empty or whitespace-only.",
            is_recoverable=False,
        )

    # 2. Sanitize raw text
    sanitized = sanitize_json_string(raw_text)

    # 3. JSON Decode Attempt
    try:
        data = json.loads(sanitized)
    except json.JSONDecodeError as decode_err:
        # Classify truncated vs syntax decode error
        if sanitized.startswith("{") and not sanitized.endswith("}"):
            err_type = "truncated_json"
            err_msg = f"JSON payload appears truncated mid-object: {decode_err}"
        else:
            err_type = "json_decode_error"
            err_msg = f"Failed to decode JSON: {decode_err}"

        logger.warning("Parse failure (%s): %s.", err_type, err_msg)
        return ParseError(
            raw_text=raw_text,
            error_type=err_type,
            error_message=err_msg,
            is_recoverable=False,
        )

    # 4. Schema dictionary validation
    if not isinstance(data, dict):
        logger.warning("Parse failure (invalid_schema): Payload is not a JSON object dictionary.")
        return ParseError(
            raw_text=raw_text,
            error_type="invalid_schema",
            error_message="Expected JSON object dictionary root, got different data structure.",
            is_recoverable=False,
        )

    insufficient_info = bool(data.get("insufficient_information", False))
    raw_claims = data.get("claims", [])
    if not isinstance(raw_claims, list):
        raw_claims = []

    # 5. Semantic Constraint Validation: contradictory response check
    if not insufficient_info and not raw_claims:
        logger.warning("Parse failure (contradictory_response): insufficient_information=False but 0 claims provided.")
        return ParseError(
            raw_text=raw_text,
            error_type="contradictory_response",
            error_message="Model reported sufficient information (insufficient_information=False) but provided 0 claims.",
            is_recoverable=False,
        )

    # 6. Process claims, filter empty text, deduplicate, and validate citation IDs
    valid_claims: list[Claim] = []
    unverified_claims: list[Claim] = []
    seen_claims: set[tuple[str, str]] = set()

    for item in raw_claims:
        if not isinstance(item, dict):
            continue
        c_text = str(item.get("claim_text", "")).strip()
        c_src = str(item.get("source_chunk_id", "")).strip()

        if not c_text:
            continue

        # Deduplicate identical (claim_text.lower(), source_chunk_id)
        dedup_key = (c_text.lower(), c_src)
        if dedup_key in seen_claims:
            logger.debug("Deduplicated repeated claim: '%s' [%s].", c_text[:40], c_src)
            continue
        seen_claims.add(dedup_key)

        claim_obj = Claim(claim_text=c_text, source_chunk_id=c_src)

        if c_src in valid_chunk_ids:
            valid_claims.append(claim_obj)
        else:
            logger.warning(
                "Claim cited non-existent source_chunk_id '%s' (not in context chunks). Quarantined to unverified_claims.",
                c_src,
            )
            unverified_claims.append(claim_obj)

    parsed_answer = GeneratedAnswer(
        claims=valid_claims,
        insufficient_information=insufficient_info,
        unverified_claims=unverified_claims,
        raw_response=raw_text,
        provider=provider,
        model_name=model_name,
        latency_ms=latency_ms,
    )

    logger.debug("Successfully parsed LLM output: %d valid claims, %d unverified.", len(valid_claims), len(unverified_claims))
    return parsed_answer
