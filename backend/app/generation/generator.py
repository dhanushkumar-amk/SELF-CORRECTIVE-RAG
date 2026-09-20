"""
Structured RAG Answer Generator with Defensive JSON Parsing and Citation Integrity Validation.

Architecture & Design Decisions:
1. Citation-Forced Generation:
   Combines query and reranked context chunks into a grounded citation prompt, invoking the LLM provider layer
   (Groq primary -> Gemini fallback).

2. Defensive JSON Output Parsing:
   LLM providers occasionally include markdown code fences (```json ... ```) or conversational preambles.
   `clean_json_output()` uses regex and string boundary extraction to guarantee clean JSON payload parsing.

3. Claim-to-Chunk Citation Validation:
   Every parsed claim's `source_chunk_id` is cross-checked against the exact set of input `chunk_id`s provided.
   If the LLM invents or hallucinates a non-existent `source_chunk_id`, the claim is quarantined into `unverified_claims`
   and logged as an untrusted claim.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.core.logging import get_logger
from app.generation.llm_client import generate_answer
from app.generation.prompts import CITATION_SYSTEM_PROMPT, build_citation_user_prompt
from app.models.schemas import Claim, GeneratedAnswer, RetrievalResult

logger = get_logger(__name__)

__all__ = [
    "clean_json_output",
    "generate_answer_with_citations",
]


def clean_json_output(raw_text: str) -> str:
    """Strip markdown code fences, preambles, and stray formatting from raw LLM text.

    Args:
        raw_text: Raw string returned by LLM provider.

    Returns:
        Clean JSON substring string.
    """
    if not raw_text or not raw_text.strip():
        return "{}"

    text = raw_text.strip()

    # 1. Remove markdown code fences like ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        # Match ```json or ``` at start and ``` at end
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()

    # 2. Extract JSON object string between outermost '{' and '}'
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace : last_brace + 1]

    return text


def generate_answer_with_citations(
    query: str,
    chunks: list[RetrievalResult],
    system_prompt: str | None = None,
    force_provider: str | None = None,
    temperature: float = 0.0,
) -> GeneratedAnswer:
    """Generate a structured, citation-anchored answer from query and context chunks.

    Args:
        query: User input query.
        chunks: List of relevant candidate RetrievalResult chunks.
        system_prompt: Optional system prompt override.
        force_provider: Optional provider override ('groq' or 'gemini').
        temperature: Sampling temperature (0.0 for deterministic output).

    Returns:
        GeneratedAnswer containing validated claims, citation metadata, and insufficiency flags.
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

    # 1. Build prompt string and invoke LLM provider layer
    sys_prompt = system_prompt or CITATION_SYSTEM_PROMPT
    user_prompt = build_citation_user_prompt(query, chunks)

    llm_res = generate_answer(
        prompt=user_prompt,
        system_prompt=sys_prompt,
        force_provider=force_provider,
        temperature=temperature,
    )

    # 2. Defensive JSON cleaning
    cleaned_json = clean_json_output(llm_res.content)

    # 3. Parse JSON response into structured dictionaries
    try:
        data = json.loads(cleaned_json)
        if not isinstance(data, dict):
            raise ValueError("Parsed JSON payload is not a dictionary.")
    except Exception as parse_err:
        logger.error(
            "Failed to parse LLM JSON response (%s). Raw text: '%s'.",
            parse_err,
            llm_res.content[:200],
        )
        return GeneratedAnswer(
            claims=[],
            insufficient_information=True,
            raw_response=llm_res.content,
            provider=llm_res.provider,
            model_name=llm_res.model_name,
            latency_ms=llm_res.latency_ms,
        )

    # 4. Extract insufficiency flag and claims
    insufficient_info = bool(data.get("insufficient_information", False))
    raw_claims = data.get("claims", [])
    if not isinstance(raw_claims, list):
        raw_claims = []

    # 5. Validate claim-to-chunk citation consistency
    valid_chunk_ids = {c.chunk_id for c in chunks}
    valid_claims: list[Claim] = []
    unverified_claims: list[Claim] = []

    for item in raw_claims:
        if not isinstance(item, dict):
            continue
        c_text = str(item.get("claim_text", "")).strip()
        c_src = str(item.get("source_chunk_id", "")).strip()
        if not c_text:
            continue

        claim_obj = Claim(claim_text=c_text, source_chunk_id=c_src)

        if c_src in valid_chunk_ids:
            valid_claims.append(claim_obj)
        else:
            logger.warning(
                "Claim cited non-existent source_chunk_id '%s' (not in input chunks). Quarantining claim.",
                c_src,
            )
            unverified_claims.append(claim_obj)

    logger.info(
        "generate_answer_with_citations completed via %s ('%s') in %.2f ms (claims: %d valid, %d unverified, insufficient: %s).",
        llm_res.provider,
        llm_res.model_name,
        llm_res.latency_ms,
        len(valid_claims),
        len(unverified_claims),
        insufficient_info,
    )

    return GeneratedAnswer(
        claims=valid_claims,
        insufficient_information=insufficient_info,
        unverified_claims=unverified_claims,
        raw_response=llm_res.content,
        provider=llm_res.provider,
        model_name=llm_res.model_name,
        latency_ms=llm_res.latency_ms,
    )
