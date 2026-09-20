"""
Unit & integration tests for Phase 25: LLM Provider Integration (Groq + Gemini fallback).
"""

from unittest.mock import patch
import pytest

from app.core.config import _is_placeholder, settings
from app.generation import (
    LLMResponse,
    generate_answer,
    generate_gemini,
    generate_groq,
)


def test_generate_groq_live_or_skip():
    """Test direct Groq LLM generation (skips gracefully if GROQ_API_KEY is missing or placeholder)."""
    if not settings.GROQ_API_KEY or _is_placeholder(settings.GROQ_API_KEY):
        pytest.skip("GROQ_API_KEY is not configured in backend/.env; skipping live Groq API test.")

    res = generate_groq("Reply with exactly the word 'PONG' and nothing else.", temperature=0.0)
    assert isinstance(res, LLMResponse)
    assert res.provider == "groq"
    assert "PONG" in res.content.upper()
    assert res.latency_ms > 0.0
    assert res.fallback_triggered is False


def test_generate_gemini_live_or_skip():
    """Test direct Gemini LLM generation (skips gracefully if GEMINI_API_KEY is missing or placeholder)."""
    if not settings.GEMINI_API_KEY or _is_placeholder(settings.GEMINI_API_KEY):
        pytest.skip("GEMINI_API_KEY is not configured in backend/.env; skipping live Gemini API test.")

    res = generate_gemini("Reply with exactly the word 'PONG' and nothing else.", temperature=0.0)
    assert isinstance(res, LLMResponse)
    assert res.provider == "gemini"
    assert "PONG" in res.content.upper()
    assert res.latency_ms > 0.0
    assert res.fallback_triggered is False


def test_fallback_triggers_when_groq_fails():
    """Verify that when Groq raises a transient API exception, generate_answer automatically fails over to Gemini."""
    mock_gemini_res = LLMResponse(
        content="Gemini fallback response text.",
        provider="gemini",
        model_name="gemini-1.5-flash",
        latency_ms=145.0,
        fallback_triggered=False,
    )

    with patch("app.generation.llm_client.generate_groq", side_effect=Exception("Groq API rate limit exceeded (429)")), \
         patch("app.generation.llm_client.generate_gemini", return_value=mock_gemini_res) as mock_gemini:

        res = generate_answer("What is Reciprocal Rank Fusion?")

        # Assert fallback occurred
        assert mock_gemini.called
        assert res.provider == "gemini"
        assert res.content == "Gemini fallback response text."
        assert res.fallback_triggered is True


def test_invalid_prompt_rejects_without_fallback():
    """Verify non-transient input errors (empty query) raise ValueError immediately without triggering fallback loops."""
    with patch("app.generation.llm_client.generate_gemini") as mock_gemini:
        with pytest.raises(ValueError, match="Prompt string cannot be empty"):
            generate_answer("")

        with pytest.raises(ValueError, match="Prompt string cannot be empty"):
            generate_answer("   \n\t  ")

        # Confirm Gemini fallback was NEVER called
        assert not mock_gemini.called
