"""Tests for chunk metadata schema design, Pinecone constraints, and vector ID management."""

import uuid
import pytest

from app.ingestion.chunk_metadata import (
    PINECONE_MAX_METADATA_BYTES,
    TRUNCATION_MARKER,
    PineconeMetadataValidationError,
    create_vector_id,
    estimate_metadata_size_bytes,
    parse_vector_id,
    to_pinecone_metadata,
    validate_pinecone_metadata,
)
from app.models.schemas import Chunk, ChunkMetadata


class TestChunkMetadataSchema:
    """Unit tests verifying Pinecone metadata constraints, types, and schema conversions."""

    def test_to_pinecone_metadata_allowed_types(self) -> None:
        """Pinecone metadata values must strictly be str, int, float, bool, or list[str]."""
        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id="doc-test-123",
            chunk_index=0,
            text="This is a test chunk with standard text content.",
            token_count=10,
            page_number=1,
            page_number_end=2,
            char_start=0,
            char_end=48,
        )

        metadata = to_pinecone_metadata(
            chunk,
            filename="annual_report_2024.pdf",
            document_title="Annual Report 2024",
        )

        # 1. Assert all keys are strings
        for key in metadata.keys():
            assert isinstance(key, str)

        # 2. Assert all values are valid Pinecone primitive types
        for key, val in metadata.items():
            if isinstance(val, bool):
                continue
            elif isinstance(val, (int, float, str)):
                continue
            elif isinstance(val, list):
                assert all(isinstance(x, str) for x in val)
            else:
                pytest.fail(f"Field '{key}' has illegal Pinecone type: {type(val)}")

        # 3. Assert internal char coordinates are explicitly excluded
        assert "char_start" not in metadata
        assert "char_end" not in metadata

        # 4. Assert required fields exist
        assert metadata["document_id"] == "doc-test-123"
        assert metadata["chunk_id"] == chunk.chunk_id
        assert metadata["chunk_index"] == 0
        assert metadata["page_number"] == 1
        assert metadata["page_number_end"] == 2
        assert metadata["source_text"] == chunk.text
        assert metadata["token_count"] == 10
        assert metadata["filename"] == "annual_report_2024.pdf"
        assert metadata["document_title"] == "Annual Report 2024"

    def test_realistic_chunk_passes_size_validator(self) -> None:
        """A realistic ~500 token chunk must comfortably pass the 40KB size limit."""
        # ~500 tokens of realistic academic text (~2500 characters)
        realistic_text = (
            "Retrieval-Augmented Generation (RAG) models combine parametric memory (a pre-trained language model) "
            "with non-parametric memory (a dense vector index of authoritative documents). When presented with a "
            "user query, the retrieval mechanism fetches the most relevant text chunks from the vector database. "
            "The retrieved passages are then prepended or concatenated to the user's prompt as grounded context. "
            "However, standard RAG systems frequently suffer from hallucinations when the retrieved context is "
            "irrelevant, contradictory, or insufficient to answer the query. Self-correcting RAG architectures "
            "solve this challenge through autonomous retrieval evaluation, query rewriting, and downstream Natural "
            "Language Inference (NLI) verification. The NLI verifier checks whether each atomic statement generated "
            "by the language model is strictly entailed by the cited source text. If a claim is contradicted or "
            "unsupported, the system triggers self-correction loops to refine or retract the claim before returning "
            "the final answer to the user. This guarantees verified citation attribution and prevents subtle "
            "hallucinations from poisoning mission-critical workflows."
        )

        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=str(uuid.uuid4()),
            chunk_index=4,
            text=realistic_text,
            token_count=485,
            page_number=3,
            page_number_end=4,
            char_start=5120,
            char_end=7620,
        )

        meta = to_pinecone_metadata(
            chunk,
            filename="self_rag_paper.pdf",
            document_title="Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        )

        errors = validate_pinecone_metadata(meta)
        assert errors == [], f"Validation failed: {errors}"

        size_bytes = estimate_metadata_size_bytes(meta)
        # Should be roughly 1.5KB to 3KB, well below 40,960 bytes
        assert 1000 <= size_bytes <= 5000
        assert size_bytes < PINECONE_MAX_METADATA_BYTES

    def test_truncation_fallback_on_artificially_oversized_chunk(self) -> None:
        """If source_text exceeds Pinecone's 40KB limit, it must be safely truncated with a marker."""
        # Create a massive 50,000 character string (> 40,960 bytes)
        massive_text = "Important evidence paragraph. " * 1700
        assert len(massive_text.encode("utf-8")) > PINECONE_MAX_METADATA_BYTES

        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id="doc-huge",
            chunk_index=1,
            text=massive_text,
            token_count=12000,
            page_number=5,
            page_number_end=5,
            char_start=0,
            char_end=len(massive_text),
        )

        # With auto_truncate=True (default), metadata should be trimmed to fit
        meta = to_pinecone_metadata(chunk, filename="huge_doc.pdf", auto_truncate=True)

        size_bytes = estimate_metadata_size_bytes(meta)
        assert size_bytes <= PINECONE_MAX_METADATA_BYTES
        assert meta["source_text"].endswith(TRUNCATION_MARKER.strip())
        assert validate_pinecone_metadata(meta) == []

        # With auto_truncate=False, validation error must be raised
        with pytest.raises(PineconeMetadataValidationError, match="exceeds Pinecone limit"):
            to_pinecone_metadata(chunk, filename="huge_doc.pdf", auto_truncate=False)

    def test_none_values_are_omitted_not_null(self) -> None:
        """Pinecone rejects null values; all None fields must be stripped from the output dict."""
        raw_meta = {
            "document_id": "doc-999",
            "chunk_id": "chunk-888",
            "chunk_index": 0,
            "page_number": 1,
            "page_number_end": 1,
            "source_text": "Sample text",
            "token_count": 5,
            "filename": "doc.pdf",
            "document_title": "Doc",
            "optional_tag": None,
            "author": None,
            "extra_notes": None,
        }

        # Conversion must remove all keys whose value is None
        cleaned = to_pinecone_metadata(raw_meta)
        assert "optional_tag" not in cleaned
        assert "author" not in cleaned
        assert "extra_notes" not in cleaned
        assert None not in cleaned.values()

        # Direct validation of a dict containing None must report error
        errors = validate_pinecone_metadata({"doc_id": "123", "bad_field": None})
        assert any("None/null" in err for err in errors)

    def test_vector_id_generation_uniqueness_and_parsing(self) -> None:
        """Vector IDs must follow '{document_id}::{chunk_id}', be unique, and parse cleanly."""
        doc_id = str(uuid.uuid4())
        generated_ids: set[str] = set()

        for i in range(100):
            chunk_id = str(uuid.uuid4())
            vec_id = create_vector_id(doc_id, chunk_id)

            assert vec_id.startswith(f"{doc_id}::")
            assert vec_id.endswith(chunk_id)

            # Test reversibility
            parsed_doc, parsed_chunk = parse_vector_id(vec_id)
            assert parsed_doc == doc_id
            assert parsed_chunk == chunk_id

            generated_ids.add(vec_id)

        # Confirm all 100 generated vector IDs are completely unique
        assert len(generated_ids) == 100

        # Invalid vector ID format handling
        with pytest.raises(ValueError, match="missing separator"):
            parse_vector_id("invalid-vector-id-without-separator")

        with pytest.raises(ValueError, match="non-empty strings"):
            create_vector_id("", "chunk-1")

    def test_type_coercion_and_rejected_types(self) -> None:
        """Numeric types and tuples should coerce properly, while nested dicts must be rejected."""
        # Test tuple converted to list of strings
        raw = {
            "document_id": "doc-coerce",
            "chunk_id": "chunk-coerce",
            "chunk_index": "0",  # string 0 coerced to int
            "page_number": 1,
            "page_number_end": 1,
            "source_text": "Content",
            "token_count": 10,
            "tags": ("tag1", "tag2"),  # tuple -> list[str]
        }
        meta = to_pinecone_metadata(raw)
        assert meta["tags"] == ["tag1", "tag2"]
        assert isinstance(meta["chunk_index"], int)

        # Test rejected nested dict
        nested = {
            "document_id": "doc-bad",
            "chunk_id": "chunk-bad",
            "page_number": 1,
            "source_text": "Content",
            "nested_obj": {"inner_key": "inner_val"},
        }
        errors = validate_pinecone_metadata(nested)
        assert any("nested dictionary" in err for err in errors)

        # Test rejected list with non-strings
        bad_list = {
            "document_id": "doc-bad",
            "chunk_id": "chunk-bad",
            "page_number": 1,
            "source_text": "Content",
            "scores": [1, 2, 3],  # list[int] is forbidden in Pinecone
        }
        errors = validate_pinecone_metadata(bad_list)
        assert any("non-string items" in err for err in errors)

    def test_chunk_metadata_model_helpers(self) -> None:
        """ChunkMetadata model methods (from_chunk, to_pinecone_metadata) work seamlessly."""
        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id="doc-model-test",
            chunk_index=2,
            text="Evidence statement for citation verification.",
            token_count=12,
            page_number=4,
            page_number_end=5,
            char_start=120,
            char_end=166,
        )

        # Create model via from_chunk
        model = ChunkMetadata.from_chunk(chunk, filename="paper.pdf")
        assert model.document_id == chunk.document_id
        assert model.page_number == 4
        assert model.page_number_end == 5
        assert model.document_title == "paper.pdf"

        # Serialize to pinecone metadata
        pinecone_dict = model.to_pinecone_metadata()
        assert pinecone_dict["document_id"] == "doc-model-test"
        assert pinecone_dict["page_number"] == 4
        assert pinecone_dict["page_number_end"] == 5
        assert "char_start" not in pinecone_dict

        # Reconstruct from pinecone metadata
        reconstructed = ChunkMetadata.from_pinecone_metadata(pinecone_dict)
        assert reconstructed.document_id == model.document_id
        assert reconstructed.chunk_id == model.chunk_id
        assert reconstructed.source_text == model.source_text
