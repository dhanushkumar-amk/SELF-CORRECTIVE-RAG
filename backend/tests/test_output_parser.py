"""
Unit tests for output parsing, sanitization, and defensive schema validation (Phase 28).
"""

import pytest

from app.generation.output_parser import parse_llm_output, sanitize_json_string
from app.models.schemas import GeneratedAnswer, ParseError


def test_sanitize_json_string_code_fences():
    raw = "```json\n{\n  \"claims\": [],\n  \"insufficient_information\": true\n}\n```"
    cleaned = sanitize_json_string(raw)
    assert cleaned == "{\n  \"claims\": [],\n  \"insufficient_information\": true\n}"


def test_sanitize_json_string_trailing_commas():
    raw = '{"claims": [{"claim_text": "Test claim", "source_chunk_id": "chunk_1",}], "insufficient_information": false,}'
    cleaned = sanitize_json_string(raw)
    assert 'chunk_1"}]' in cleaned
    assert "false}" in cleaned


def test_sanitize_json_string_embedded_in_sentence():
    raw = 'Here is the requested answer: {"claims": [{"claim_text": "Sample", "source_chunk_id": "chunk_1"}], "insufficient_information": false}. Hope this helps!'
    cleaned = sanitize_json_string(raw)
    assert cleaned.startswith("{")
    assert cleaned.endswith("}")
    assert '"claim_text": "Sample"' in cleaned


def test_parse_llm_output_empty_response():
    res = parse_llm_output("", valid_chunk_ids={"chunk_1"})
    assert isinstance(res, ParseError)
    assert res.error_type == "empty_response"

    res_spaces = parse_llm_output("   \n\t  ", valid_chunk_ids={"chunk_1"})
    assert isinstance(res_spaces, ParseError)
    assert res_spaces.error_type == "empty_response"


def test_parse_llm_output_non_json_text():
    raw = "I am sorry, but I cannot fulfill this request in JSON format."
    res = parse_llm_output(raw, valid_chunk_ids={"chunk_1"})
    assert isinstance(res, ParseError)
    assert res.error_type == "json_decode_error"


def test_parse_llm_output_truncated_json():
    raw = '{"claims": [{"claim_text": "Truncated claim", "source_chunk_id": "chunk_1"'
    res = parse_llm_output(raw, valid_chunk_ids={"chunk_1"})
    assert isinstance(res, ParseError)
    assert res.error_type == "truncated_json"


def test_parse_llm_output_invalid_schema():
    raw = '[{"claim_text": "Array root instead of object"}]'
    res = parse_llm_output(raw, valid_chunk_ids={"chunk_1"})
    assert isinstance(res, ParseError)
    assert res.error_type == "invalid_schema"


def test_parse_llm_output_contradictory_response():
    raw = '{"claims": [], "insufficient_information": false}'
    res = parse_llm_output(raw, valid_chunk_ids={"chunk_1"})
    assert isinstance(res, ParseError)
    assert res.error_type == "contradictory_response"


def test_parse_llm_output_valid_response_deduplication_and_citation():
    valid_ids = {"chunk_1", "chunk_2"}
    raw = """{
      "claims": [
        {"claim_text": "First factual statement.", "source_chunk_id": "chunk_1"},
        {"claim_text": "first factual statement.", "source_chunk_id": "chunk_1"},
        {"claim_text": "Second statement with invalid chunk.", "source_chunk_id": "fake_chunk_99"},
        {"claim_text": "", "source_chunk_id": "chunk_2"}
      ],
      "insufficient_information": false
    }"""
    res = parse_llm_output(raw, valid_chunk_ids=valid_ids, provider="groq", model_name="test-model")
    assert isinstance(res, GeneratedAnswer)
    assert len(res.claims) == 1
    assert res.claims[0].claim_text == "First factual statement."
    assert res.claims[0].source_chunk_id == "chunk_1"
    assert len(res.unverified_claims) == 1
    assert res.unverified_claims[0].source_chunk_id == "fake_chunk_99"
    assert res.insufficient_information is False
