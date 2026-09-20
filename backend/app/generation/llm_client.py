"""
LLM Provider Integration & Fallback Interface (Groq Primary -> Gemini Fallback).

Architecture & Design Decisions:
1. Unified Abstraction:
   `generate_answer()` provides a single, decoupled interface for downstream RAG nodes.
   Downstream components invoke `generate_answer()` without coupling to provider SDKs.

2. Primary + Fallback Resiliency Chain:
   - Primary: Groq (Llama 3.3 70B Versatile / Llama 3.1 8B Instant) - ultra-fast inference (< 300ms).
   - Fallback: Google Gemini (Gemini 1.5 Flash) - triggered automatically if Groq encounters API rate limits,
     network timeouts, or provider 5xx errors.

3. Transient vs. Non-Transient Error Handling:
   - Non-transient input errors (e.g. empty or whitespace-only prompt) fail fast immediately with a ValueError
     and are NEVER retried into infinite fallback loops.
   - Transient provider failures (network errors, rate limits 429, service unavailable 503) log a warning and
     seamlessly fail over to Gemini, marking `fallback_triggered=True` on the response.
"""

from __future__ import annotations

import time
from typing import Any

from app.core.config import _is_placeholder, settings
from app.core.logging import get_logger
from app.models.schemas import LLMResponse

logger = get_logger(__name__)

__all__ = [
    "LLMResponse",
    "generate_answer",
    "generate_gemini",
    "generate_groq",
]


def generate_groq(
    prompt: str,
    system_prompt: str | None = None,
    model_name: str | None = None,
    temperature: float = 0.0,
) -> LLMResponse:
    """Generate response text directly via Groq API.

    Args:
        prompt: User input prompt string.
        system_prompt: Optional system instruction prompt string.
        model_name: Optional explicit Groq model name; defaults to settings.DEFAULT_GROQ_MODEL.
        temperature: Sampling temperature (0.0 for deterministic output).

    Returns:
        LLMResponse containing generated content, provider metadata, and latency.
    """
    if not settings.GROQ_API_KEY or _is_placeholder(settings.GROQ_API_KEY):
        raise ValueError("GROQ_API_KEY is not configured or set to an example placeholder.")

    from groq import Groq

    model = model_name or settings.DEFAULT_GROQ_MODEL
    client = Groq(api_key=settings.GROQ_API_KEY)

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    start_t = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    duration_ms = (time.perf_counter() - start_t) * 1000.0

    content = response.choices[0].message.content or ""
    logger.info("Served LLM request via Groq ('%s') in %.2f ms.", model, duration_ms)
    return LLMResponse(
        content=content,
        provider="groq",
        model_name=model,
        latency_ms=round(duration_ms, 2),
        fallback_triggered=False,
    )


def generate_gemini(
    prompt: str,
    system_prompt: str | None = None,
    model_name: str | None = None,
    temperature: float = 0.0,
) -> LLMResponse:
    """Generate response text directly via Google Gemini API.

    Args:
        prompt: User input prompt string.
        system_prompt: Optional system instruction prompt string.
        model_name: Optional explicit Gemini model name; defaults to settings.DEFAULT_GEMINI_MODEL.
        temperature: Sampling temperature (0.0 for deterministic output).

    Returns:
        LLMResponse containing generated content, provider metadata, and latency.
    """
    if not settings.GEMINI_API_KEY or _is_placeholder(settings.GEMINI_API_KEY):
        raise ValueError("GEMINI_API_KEY is not configured or set to an example placeholder.")

    from google import genai
    from google.genai import types

    model = model_name or settings.DEFAULT_GEMINI_MODEL
    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    config_kwargs: dict[str, Any] = {"temperature": temperature}
    if system_prompt:
        config_kwargs["system_instruction"] = system_prompt

    config = types.GenerateContentConfig(**config_kwargs)

    start_t = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    duration_ms = (time.perf_counter() - start_t) * 1000.0

    content = response.text or ""
    logger.info("Served LLM request via Gemini ('%s') in %.2f ms.", model, duration_ms)
    return LLMResponse(
        content=content,
        provider="gemini",
        model_name=model,
        latency_ms=round(duration_ms, 2),
        fallback_triggered=False,
    )


def generate_answer(
    prompt: str,
    system_prompt: str | None = None,
    model_name: str | None = None,
    force_provider: str | None = None,
    temperature: float = 0.0,
) -> LLMResponse:
    """Generate an answer using Groq as primary provider with automatic Gemini failover.

    Args:
        prompt: User input prompt string.
        system_prompt: Optional system instruction prompt string.
        model_name: Optional specific model name override.
        force_provider: Optional explicit provider override ('groq' or 'gemini').
        temperature: Sampling temperature (0.0 for deterministic RAG answers).

    Returns:
        LLMResponse containing generated content, serving provider, latency, and fallback status.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt string cannot be empty or whitespace-only.")

    # Force explicit provider if requested
    if force_provider and force_provider.lower() == "gemini":
        return generate_gemini(prompt, system_prompt, model_name, temperature)
    if force_provider and force_provider.lower() == "groq":
        return generate_groq(prompt, system_prompt, model_name, temperature)

    # Primary (Groq) -> Fallback (Gemini) execution chain
    try:
        return generate_groq(prompt, system_prompt, model_name, temperature)
    except ValueError as ve:
        # Non-transient input or config validation errors: re-raise or attempt fallback if key missing
        if "Prompt string" in str(ve):
            raise ve
        logger.warning(
            "Groq provider configuration issue (%s). Failing over to Gemini fallback...",
            ve,
        )
    except Exception as groq_exc:
        logger.warning(
            "Groq primary generation failed (%s: %s). Failing over to Gemini fallback...",
            type(groq_exc).__name__,
            groq_exc,
        )

    # Fallback to Gemini
    fallback_res = generate_gemini(prompt, system_prompt, model_name=None, temperature=temperature)
    return LLMResponse(
        content=fallback_res.content,
        provider=fallback_res.provider,
        model_name=fallback_res.model_name,
        latency_ms=fallback_res.latency_ms,
        fallback_triggered=True,
    )
