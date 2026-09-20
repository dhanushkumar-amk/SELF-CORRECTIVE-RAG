"""
Unit tests for malformed output retry, fallback handling, and terminal GenerationError (Phase 29).
"""

from unittest.mock import MagicMock, patch
import pytest

from app.generation.generator import generate_answer_with_citations
from app.models.schemas import GeneratedAnswer, GenerationError, LLMResponse, RetrievalResult


@pytest.fixture
def sample_chunks() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="chunk_101",
            score=0.92,
            metadata={"source_text": "Solar energy is renewable energy from the sun."},
        )
    ]


def test_retry_succeeds_on_second_attempt(sample_chunks):
    """Test that a malformed first response triggers a corrective prompt retry and succeeds on 2nd attempt."""
    malformed_res = LLMResponse(
        content="Here is the answer: Solar energy is great but this is non-JSON text.",
        provider="groq",
        model_name="mock-groq-model",
        latency_ms=120.0,
    )
    valid_res = LLMResponse(
        content='{"claims": [{"claim_text": "Solar energy is renewable energy from the sun.", "source_chunk_id": "chunk_101"}], "insufficient_information": false}',
        provider="groq",
        model_name="mock-groq-model",
        latency_ms=110.0,
    )

    with patch("app.generation.generator.generate_answer", side_effect=[malformed_res, valid_res]) as mock_gen:
        answer = generate_answer_with_citations(
            query="What is solar energy?",
            chunks=sample_chunks,
            force_provider="groq",
            max_retries=2,
        )

        assert isinstance(answer, GeneratedAnswer)
        assert len(answer.claims) == 1
        assert answer.claims[0].source_chunk_id == "chunk_101"
        assert mock_gen.call_count == 2

        # Verify second call received corrective instruction in prompt
        second_call_prompt = mock_gen.call_args_list[1].kwargs["prompt"]
        assert "[CORRECTIVE INSTRUCTION]" in second_call_prompt
        assert "json_decode_error" in second_call_prompt


def test_retry_cap_respected(sample_chunks):
    """Test that max_retries cap is respected (1 initial + 2 retries = 3 attempts total)."""
    malformed_res = LLMResponse(
        content="Persistent non-JSON string response.",
        provider="groq",
        model_name="mock-groq-model",
        latency_ms=100.0,
    )

    with patch("app.generation.generator.generate_answer", return_value=malformed_res) as mock_gen:
        with pytest.raises(GenerationError) as exc_info:
            generate_answer_with_citations(
                query="What is solar energy?",
                chunks=sample_chunks,
                force_provider="groq",  # Disable fallback to isolate primary retries
                max_retries=2,
            )

        assert "Unable to generate a reliable answer, please try rephrasing your question." in str(exc_info.value)
        # Attempt 1 + 2 retries = 3 total calls on groq
        assert mock_gen.call_count == 3


def test_fallback_after_retry_exhaustion(sample_chunks):
    """Test fallback to secondary provider (gemini) after primary provider (groq) exhausts all retries."""
    groq_malformed = LLMResponse(
        content="Groq consistently outputting non-JSON.",
        provider="groq",
        model_name="mock-groq-model",
        latency_ms=100.0,
    )
    gemini_valid = LLMResponse(
        content='{"claims": [{"claim_text": "Solar energy is renewable energy.", "source_chunk_id": "chunk_101"}], "insufficient_information": false}',
        provider="gemini",
        model_name="mock-gemini-model",
        latency_ms=150.0,
    )

    def side_effect_func(prompt, system_prompt, force_provider, temperature):
        if force_provider == "groq":
            return groq_malformed
        return gemini_valid

    with patch("app.generation.generator.settings") as mock_settings, \
         patch("app.generation.generator.generate_answer", side_effect=side_effect_func) as mock_gen:
        mock_settings.LLM_PRIMARY_PROVIDER = "groq"
        mock_settings.LLM_FALLBACK_PROVIDER = "gemini"

        answer = generate_answer_with_citations(
            query="What is solar energy?",
            chunks=sample_chunks,
            force_provider=None,  # Use settings primary/fallback
            max_retries=2,
        )

        assert isinstance(answer, GeneratedAnswer)
        assert answer.provider == "gemini"
        assert len(answer.claims) == 1
        # 3 calls on groq + 1 fallback call on gemini = 4 total calls
        assert mock_gen.call_count == 4


def test_total_failure_raises_generation_error(sample_chunks):
    """Test total failure across primary retries and secondary fallback raises GenerationError cleanly."""
    malformed = LLMResponse(
        content="Completely broken output across all providers.",
        provider="groq",
        model_name="mock-model",
        latency_ms=50.0,
    )

    with patch("app.generation.generator.settings") as mock_settings, \
         patch("app.generation.generator.generate_answer", return_value=malformed) as mock_gen:
        mock_settings.LLM_PRIMARY_PROVIDER = "groq"
        mock_settings.LLM_FALLBACK_PROVIDER = "gemini"

        with pytest.raises(GenerationError) as exc_info:
            generate_answer_with_citations(
                query="What is solar energy?",
                chunks=sample_chunks,
                force_provider=None,
                max_retries=2,
            )

        assert exc_info.value.message == "Unable to generate a reliable answer, please try rephrasing your question."
        assert exc_info.value.details["last_error_type"] == "json_decode_error"
