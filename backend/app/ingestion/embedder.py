"""
Local embedding generation module using sentence-transformers.

Architecture & Design Decisions:
1. Model Choice: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions).
   - Matched exactly to the Pinecone Starter index created in Phase 3 (dimension=384, metric=cosine).
   - Fully local execution: no API keys, no network calls after initial download, deterministic inference.
   - Fast CPU inference (typically 50-150 chunks/sec on commodity hardware).
2. Singleton Lifecycle:
   - The model is loaded once on first access via `get_embedding_model()` and cached in-memory.
   - Avoids expensive model reloading overhead across API requests.
3. Batch Processing:
   - `embed_texts()` uses native vectorized batching (`batch_size=32` default) rather than looping one-by-one.
4. Normalization:
   - Embeddings are L2-normalized (`normalize_embeddings=True`) for optimal cosine similarity evaluation in Pinecone.
5. Max Sequence Length & Truncation (Critical Finding):
   - all-MiniLM-L6-v2 has a default `max_seq_length = 256` tokens, while the underlying BERT architecture
     has `max_position_embeddings = 512`.
   - By default, texts longer than `max_seq_length` are truncated by the tokenizer at 256 WordPiece tokens.
   - We expose `max_seq_length` configuration and document the truncation tradeoff explicitly.
"""

from __future__ import annotations

import threading
from typing import Any

from sentence_transformers import SentenceTransformer

from app.core.logging import get_logger

logger = get_logger(__name__)

# Canonical model settings
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384
DEFAULT_MAX_SEQ_LENGTH: int = 256
DEFAULT_BATCH_SIZE: int = 32

_model_lock = threading.Lock()
_model_instance: SentenceTransformer | None = None

__all__ = [
    "EMBEDDING_MODEL_NAME",
    "EMBEDDING_DIMENSION",
    "DEFAULT_MAX_SEQ_LENGTH",
    "DEFAULT_BATCH_SIZE",
    "get_embedding_model",
    "embed_texts",
    "embed_query",
    "get_model_info",
]


def get_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
    max_seq_length: int | None = None,
) -> SentenceTransformer:
    """Retrieve the cached SentenceTransformer model singleton, loading it on first call.

    Thread-safe initialization ensures only one instance is loaded in memory.
    """
    global _model_instance
    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:
                logger.info(
                    "Loading local embedding model '%s' (dimension: %d)...",
                    model_name,
                    EMBEDDING_DIMENSION,
                )
                model = SentenceTransformer(model_name)
                if max_seq_length is not None:
                    model.max_seq_length = max_seq_length
                if hasattr(model, "get_embedding_dimension"):
                    actual_dim = model.get_embedding_dimension()
                else:
                    actual_dim = model.get_sentence_embedding_dimension()
                if actual_dim != EMBEDDING_DIMENSION:
                    logger.critical(
                        "Embedding dimension mismatch! Model '%s' produces %d dims, but Pinecone index requires %d.",
                        model_name,
                        actual_dim,
                        EMBEDDING_DIMENSION,
                    )
                logger.info(
                    "Embedding model '%s' loaded successfully (dimension: %d, max_seq_length: %d).",
                    model_name,
                    actual_dim,
                    model.max_seq_length,
                )
                _model_instance = model
    return _model_instance


def get_model_info() -> dict[str, Any]:
    """Return runtime metadata and configuration of the active embedding model."""
    model = get_embedding_model()
    if hasattr(model, "get_embedding_dimension"):
        dim = model.get_embedding_dimension()
    else:
        dim = model.get_sentence_embedding_dimension()
    return {
        "model_name": EMBEDDING_MODEL_NAME,
        "dimension": dim,
        "max_seq_length": model.max_seq_length,
        "max_position_embeddings": getattr(model[0].auto_model.config, "max_position_embeddings", None),
        "tokenizer_type": type(model.tokenizer).__name__,
    }


def embed_texts(
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    normalize: bool = True,
) -> list[list[float]]:
    """Generate dense vector embeddings for a list of texts using native batch inference.

    Args:
        texts: List of strings to encode.
        batch_size: Number of texts to process in parallel per mini-batch.
        normalize: If True, vectors are unit-normalized (L2 norm = 1.0) for cosine similarity.

    Returns:
        list[list[float]]: A list of 384-dimensional float vectors.
    """
    if not texts:
        return []

    model = get_embedding_model()

    # Pre-process edge cases: check for empty or whitespace-only texts
    cleaned_texts: list[str] = []
    empty_indices: list[int] = []

    for i, t in enumerate(texts):
        if not t or not t.strip():
            logger.warning(
                "Empty text passed to embed_texts at index %d. Chunker output should never be empty.",
                i,
            )
            cleaned_texts.append("")
            empty_indices.append(i)
        else:
            cleaned_texts.append(t)

    # Encode in batched mode using sentence-transformers native vectorization
    raw_embeddings = model.encode(
        cleaned_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )

    # Convert to pure Python list[list[float]]
    result: list[list[float]] = [
        [float(val) for val in vec]
        for vec in raw_embeddings
    ]

    # For empty strings, ensure zero vector or log warning
    for idx in empty_indices:
        result[idx] = [0.0] * EMBEDDING_DIMENSION

    return result


def embed_query(
    text: str,
    normalize: bool = True,
) -> list[float]:
    """Generate a dense vector embedding for a single user retrieval query.

    Used at query time in Phase 16 (Hybrid Retrieval).

    Args:
        text: Query string.
        normalize: If True, vector is unit-normalized for cosine similarity.

    Returns:
        list[float]: A single 384-dimensional float vector.
    """
    if not text or not text.strip():
        logger.warning("Empty query string passed to embed_query. Returning zero vector.")
        return [0.0] * EMBEDDING_DIMENSION

    model = get_embedding_model()
    raw_embedding = model.encode(
        text,
        show_progress_bar=False,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return [float(val) for val in raw_embedding]
