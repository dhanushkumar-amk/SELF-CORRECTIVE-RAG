"""Tests for environment variable validation and fail-fast behavior."""

import pytest

from app.core.config import get_settings


class TestConfigValidation:
    """Verify that the app fails fast when required env vars are missing."""

    def test_missing_pinecone_key_raises(self) -> None:
        """App should refuse to start if PINECONE_API_KEY is missing and validation is on."""
        with pytest.raises(ValueError, match="PINECONE_API_KEY"):
            get_settings(
                REQUIRE_API_KEYS=True,
                PINECONE_API_KEY="",
                GEMINI_API_KEY="test-gemini-key",
            )

    def test_missing_all_llm_keys_raises(self) -> None:
        """App should refuse to start if NEITHER Gemini nor Groq key is set."""
        with pytest.raises(ValueError, match="GEMINI_API_KEY or GROQ_API_KEY"):
            get_settings(
                REQUIRE_API_KEYS=True,
                PINECONE_API_KEY="test-pinecone-key",
                GEMINI_API_KEY="",
                GROQ_API_KEY="",
            )

    def test_groq_alone_is_sufficient(self) -> None:
        """Having only GROQ_API_KEY (without Gemini) should pass validation."""
        s = get_settings(
            REQUIRE_API_KEYS=True,
            PINECONE_API_KEY="test-pinecone-key",
            GEMINI_API_KEY="",
            GROQ_API_KEY="test-groq-key",
        )
        assert s.GROQ_API_KEY == "test-groq-key"

    def test_gemini_alone_is_sufficient(self) -> None:
        """Having only GEMINI_API_KEY (without Groq) should pass validation."""
        s = get_settings(
            REQUIRE_API_KEYS=True,
            PINECONE_API_KEY="test-pinecone-key",
            GEMINI_API_KEY="test-gemini-key",
            GROQ_API_KEY="",
        )
        assert s.GEMINI_API_KEY == "test-gemini-key"

    def test_validation_skipped_when_disabled(self) -> None:
        """Setting REQUIRE_API_KEYS=false should skip all validation."""
        s = get_settings(
            REQUIRE_API_KEYS=False,
            PINECONE_API_KEY="",
            GEMINI_API_KEY="",
            GROQ_API_KEY="",
        )
        assert s.PINECONE_API_KEY == ""

    def test_langsmith_enabled_property(self) -> None:
        """langsmith_enabled should be True only when both key and flag are set."""
        s = get_settings(
            REQUIRE_API_KEYS=False,
            LANGCHAIN_API_KEY="test-key",
            LANGCHAIN_TRACING_V2=True,
        )
        assert s.langsmith_enabled is True

    def test_langsmith_disabled_without_key(self) -> None:
        """langsmith_enabled should be False if API key is missing."""
        s = get_settings(
            REQUIRE_API_KEYS=False,
            LANGCHAIN_API_KEY="",
            LANGCHAIN_TRACING_V2=True,
        )
        assert s.langsmith_enabled is False
