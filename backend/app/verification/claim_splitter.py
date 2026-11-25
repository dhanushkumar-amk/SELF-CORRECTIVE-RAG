r"""
Independent Sentence & Claim Granularity Splitter for Phase 30.

Architecture & Design Decisions:
1. Sentence Tokenizer Selection:
   Uses NLTK's `sent_tokenize` (Punkt sentence boundary detector).
   Justification: NLTK is lightweight (~1.8MB footprint), fast, handles abbreviations
   (e.g., "Dr. Smith", "U.S.A.") accurately, and avoids heavy native model
   dependencies (e.g., spaCy's 15MB+ binary model downloads).

2. Defensive Fallback Strategy:
   If the installed Punkt data is unavailable, uses punctuation and abbreviation rules.

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

from app.core.logging import get_logger
from app.models.schemas import Claim

logger = get_logger(__name__)

__all__ = [
    "ensure_atomic_claims",
    "is_atomic_claim",
    "split_into_sentences",
]

_TITLE_END = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr)\.$", re.IGNORECASE)
_INITIALS_END = re.compile(r"\b(?:[A-Z]\.)+$")
_SENTENCE_GAP = re.compile(r"(?<=[.!?])\s+")


def _split_without_punkt(text: str) -> list[str]:
    sentences = []
    start = 0
    for gap in _SENTENCE_GAP.finditer(text):
        segment = text[start:gap.start()]
        next_char = text[gap.end():gap.end() + 1]
        if _TITLE_END.search(segment) or (_INITIALS_END.search(segment) and next_char.islower()):
            continue
        sentences.append(segment)
        start = gap.end()
    sentences.append(text[start:])
    return sentences


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

    try:
        from nltk import sent_tokenize

        return sent_tokenize(cleaned_text)
    except (ImportError, LookupError):
        return _split_without_punkt(cleaned_text)


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
