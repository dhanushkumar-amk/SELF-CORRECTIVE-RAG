r"""
Independent Sentence & Claim Granularity Splitter for Phase 30.

Architecture & Design Decisions:
1. Sentence Tokenizer Selection:
   Uses NLTK's `sent_tokenize` (Punkt sentence boundary detector).
   Justification: NLTK is lightweight (~1.8MB footprint), fast, handles abbreviations
   (e.g., "Dr. Smith", "U.S.A.") accurately, and avoids heavy native model
   dependencies (e.g., spaCy's 15MB+ binary model downloads).

2. Defensive Fallback Strategy:
   If NLTK resource download or tokenization encounters an issue (e.g. offline environment),
   falls back gracefully to regex sentence boundary splitting (`r'(?<=[.!?])\s+'`).

3. Claim Atomicity Verification & Decomposition:
   Checks whether LLM-generated `Claim` instances contain compound/multiple sentences.
   If compound, decomposes the claim into atomic single-sentence claims where each sub-claim
   inherits the parent's `source_chunk_id`.

   Architectural Note / Heuristic Limitation:
   When an LLM merges multiple factual sentences into a single claim under one chunk_id,
   we cannot determine which specific sentence corresponds to which sub-part of the chunk.
   Therefore, inheriting the original `source_chunk_id` for all decomposed atomic sentences
   is the best available fallback for downstream NLI premise verification.
"""

from __future__ import annotations

import re
from typing import Any

from app.core.logging import get_logger
from app.models.schemas import Claim

logger = get_logger(__name__)

__all__ = [
    "ensure_atomic_claims",
    "is_atomic_claim",
    "split_into_sentences",
]

_NLTK_INITIALIZED = False


def _ensure_nltk_punkt() -> bool:
    """Ensure NLTK punkt and punkt_tab tokenizer resources are downloaded and available."""
    global _NLTK_INITIALIZED
    if _NLTK_INITIALIZED:
        return True

    try:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        _NLTK_INITIALIZED = True
        return True
    except Exception as exc:
        logger.warning("Failed to initialize NLTK punkt tokenizer (%s). Falling back to regex splitter.", exc)
        return False


def split_into_sentences(text: str) -> list[str]:
    """Split input text string into a list of individual atomic sentences.

    Args:
        text: Raw text or claim string.

    Returns:
        List of non-empty sentence strings with leading/trailing whitespace removed.
    """
    if not text or not text.strip():
        return []

    cleaned_text = text.strip()

    if _ensure_nltk_punkt():
        try:
            import nltk

            raw_sentences = nltk.sent_tokenize(cleaned_text)
            sentences = [s.strip() for s in raw_sentences if s and s.strip()]
            if sentences:
                return sentences
        except Exception as exc:
            logger.debug("NLTK sentence tokenize error (%s). Falling back to regex.", exc)

    # Fallback regex sentence tokenizer: splits on punctuation (. ! ?) followed by whitespace
    raw_sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)
    return [s.strip() for s in raw_sentences if s and s.strip()]


def is_atomic_claim(claim_text: str) -> bool:
    """Check if a claim string consists of a single atomic sentence.

    Args:
        claim_text: Factual claim statement string.

    Returns:
        True if claim contains <= 1 sentence, False if it contains multiple sentences.
    """
    sentences = split_into_sentences(claim_text)
    return len(sentences) <= 1


def ensure_atomic_claims(claims: list[Claim]) -> list[Claim]:
    """Ensure all claims in the input list are atomic single-sentence claims.

    If a Claim contains multiple sentences, decomposes it into separate atomic Claim objects.
    Each decomposed sentence inherits the parent Claim's source_chunk_id.

    Args:
        claims: List of input Claim objects.

    Returns:
        List of atomic Claim objects.
    """
    atomic_claims: list[Claim] = []

    for claim in claims:
        if not claim.claim_text or not claim.claim_text.strip():
            continue

        sentences = split_into_sentences(claim.claim_text)

        if len(sentences) <= 1:
            atomic_claims.append(claim)
        else:
            logger.info(
                "Decomposed compound non-atomic claim into %d atomic sentences (chunk_id: '%s'): '%s'",
                len(sentences),
                claim.source_chunk_id,
                claim.claim_text[:60],
            )

            # Heuristic Fallback: Each decomposed sentence inherits parent claim's source_chunk_id
            for sentence in sentences:
                atomic_claims.append(
                    Claim(
                        claim_text=sentence,
                        source_chunk_id=claim.source_chunk_id,
                    )
                )

    return atomic_claims
