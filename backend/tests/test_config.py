"""Tests for centralized configuration, fail-fast validation, and secret safeguards."""

import pytest

from app.core.config import get_settings


class TestConfigValidation:
    """Verify that the app fails fast when required configuration is missing or malformed."""

    def test_missing_pinecone_key_raises_actionable_error(self) -> None:
        """App should refuse to start if PINECONE_API_KEY is missing, providing URL to get one."""
        with pytest.raises(ValueError) as exc_info:
            get_settings(
                REQUIRE_API_KEYS=True,
                PINECONE_API_KEY="",
                PINECONE_INDEX_NAME="self-correcting-rag",
                PINECONE_ENVIRONMENT="us-east-1",
            )
        err_msg = str(exc_info.value)
        assert "PINECONE_API_KEY is missing" in err_msg
        assert "https://app.pinecone.io" in err_msg

    @pytest.mark.parametrize(
        "placeholder",
        [
            "your-pinecone-api-key-here",
            "your_pinecone_api_key_here",
            "pcsk_your_actual_key_here",
            "placeholder",
            "change-me",
            "replace-me-now",
        ],
    )
    def test_placeholder_pinecone_key_raises(self, placeholder: str) -> None:
        """App should detect example placeholder values and refuse to start."""
        with pytest.raises(ValueError) as exc_info:
            get_settings(
                REQUIRE_API_KEYS=True,
                PINECONE_API_KEY=placeholder,
                PINECONE_INDEX_NAME="self-correcting-rag",
                PINECONE_ENVIRONMENT="us-east-1",
            )
        err_msg = str(exc_info.value)
        assert "set to an example placeholder" in err_msg
        assert placeholder in err_msg
        assert "https://app.pinecone.io" in err_msg

    def test_missing_pinecone_index_name_raises(self) -> None:
        """App should refuse to start if PINECONE_INDEX_NAME is empty."""
        with pytest.raises(ValueError) as exc_info:
            get_settings(
                REQUIRE_API_KEYS=True,
                PINECONE_API_KEY="valid-test-key",
                PINECONE_INDEX_NAME="",
                PINECONE_ENVIRONMENT="us-east-1",
            )
        err_msg = str(exc_info.value)
        assert "PINECONE_INDEX_NAME is missing" in err_msg
        assert "self-correcting-rag" in err_msg

    def test_missing_pinecone_environment_raises(self) -> None:
        """App should refuse to start if PINECONE_ENVIRONMENT is empty."""
        with pytest.raises(ValueError) as exc_info:
            get_settings(
                REQUIRE_API_KEYS=True,
                PINECONE_API_KEY="valid-test-key",
                PINECONE_INDEX_NAME="self-correcting-rag",
                PINECONE_ENVIRONMENT="",
            )
        err_msg = str(exc_info.value)
        assert "PINECONE_ENVIRONMENT is missing" in err_msg

    def test_valid_config_loads_successfully(self) -> None:
        """Valid configuration should load cleanly without any exceptions."""
        s = get_settings(
            REQUIRE_API_KEYS=True,
            PINECONE_API_KEY="valid-pinecone-key",
            PINECONE_INDEX_NAME="self-correcting-rag",
            PINECONE_ENVIRONMENT="us-east-1",
            ENVIRONMENT="development",
            LOG_LEVEL="INFO",
        )
        assert s.PINECONE_API_KEY == "valid-pinecone-key"
        assert s.PINECONE_INDEX_NAME == "self-correcting-rag"
        assert s.PINECONE_ENVIRONMENT == "us-east-1"
        assert s.ENVIRONMENT == "development"
        assert s.LOG_LEVEL == "INFO"

    def test_validation_skipped_when_disabled(self) -> None:
        """Setting REQUIRE_API_KEYS=false should allow empty or placeholder keys for offline dev/CI."""
        s = get_settings(
            REQUIRE_API_KEYS=False,
            PINECONE_API_KEY="your-pinecone-api-key-here",
            PINECONE_INDEX_NAME="",
            PINECONE_ENVIRONMENT="",
        )
        assert s.REQUIRE_API_KEYS is False
        assert s.PINECONE_API_KEY == "your-pinecone-api-key-here"

    def test_production_requires_llm_key(self) -> None:
        """In production environment, at least one LLM key should be verified."""
        with pytest.raises(ValueError) as exc_info:
            get_settings(
                REQUIRE_API_KEYS=True,
                ENVIRONMENT="production",
                PINECONE_API_KEY="valid-key",
                PINECONE_INDEX_NAME="self-correcting-rag",
                PINECONE_ENVIRONMENT="us-east-1",
                GEMINI_API_KEY="",
                GROQ_API_KEY="",
            )
        assert "at least one LLM key" in str(exc_info.value)

    def test_log_level_normalization(self) -> None:
        """Log level string should be normalized to uppercase."""
        s = get_settings(
            REQUIRE_API_KEYS=False,
            LOG_LEVEL="debug",
        )
        assert s.LOG_LEVEL == "DEBUG"

    def test_langsmith_enabled_property(self) -> None:
        """langsmith_enabled should be True only when both key and flag are set."""
        s = get_settings(
            REQUIRE_API_KEYS=False,
            LANGCHAIN_API_KEY="valid-langsmith-key",
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
