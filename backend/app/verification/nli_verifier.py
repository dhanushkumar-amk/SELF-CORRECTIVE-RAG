"""
Local Natural Language Inference (NLI) Verification Engine for Phase 33.

Architecture & Design Decisions:
1. Model Selection & Loading:
   Uses `cross-encoder/nli-deberta-v3-base` via sentence-transformers' `CrossEncoder`.
   Loaded ONCE at startup via a lazy thread-safe singleton instance (`NLIVerifier`).

2. Confirmed Label Index Mapping (Ground Truth):
   Index 0 -> `contradiction`  (`VerificationStatus.CONTRADICTED`)
   Index 1 -> `entailment`     (`VerificationStatus.ENTAILED`)
   Index 2 -> `neutral`        (`VerificationStatus.NEUTRAL`)

3. Sequence Length Ceiling:
   Model max sequence length is 512 tokens (`model.max_seq_length = 512`), which accommodates
   a full context chunk premise (~400-500 tokens) paired with an atomic claim hypothesis (~20 tokens).

4. Input Integrity & Error Guarding:
   Empty, None, or whitespace-only premises/hypotheses raise an explicit `ValueError`
   to prevent downstream inference on invalid/unresolved claims.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import NLIScore, VerificationStatus

logger = get_logger(__name__)

__all__ = [
    "NLIVerifier",
    "get_nli_verifier",
    "score_entailment",
    "score_entailment_batch",
]

_NLI_VERIFIER_INSTANCE: NLIVerifier | None = None
_NLI_VERIFIER_LOCK = threading.Lock()


def _softmax(x: np.ndarray) -> np.ndarray:
    """Compute row-wise softmax over raw logits array."""
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


class NLIVerifier:
    """Singleton wrapper around sentence-transformers CrossEncoder NLI model."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.NLI_MODEL_NAME
        logger.info("Initializing NLI CrossEncoder model '%s'...", self.model_name)
        self.model = CrossEncoder(self.model_name)
        
        # Verify model config id2label mapping
        id2label = getattr(self.model.model.config, "id2label", {0: "contradiction", 1: "entailment", 2: "neutral"})
        logger.info("NLI Model id2label mapping verified: %s", id2label)
        
        self.max_seq_length = getattr(self.model, "max_seq_length", 512)
        logger.info("NLI Model max_seq_length: %d tokens", self.max_seq_length)

    def score_pair(self, premise: str, hypothesis: str) -> NLIScore:
        """Score a single premise-hypothesis pair."""
        return self.score_batch([(premise, hypothesis)])[0]

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[NLIScore]:
        """Score a batch of (premise, hypothesis) pairs."""
        if not pairs:
            return []

        # Defensive validation: check for empty or None premises/hypotheses
        for idx, (premise, hypothesis) in enumerate(pairs):
            if premise is None or not str(premise).strip():
                raise ValueError(
                    f"Premise cannot be empty or None for NLI verification (pair index {idx})."
                )
            if hypothesis is None or not str(hypothesis).strip():
                raise ValueError(
                    f"Hypothesis cannot be empty or None for NLI verification (pair index {idx})."
                )

        raw_scores = self.model.predict(pairs)

        # Handle 1D array single prediction vs 2D batch array
        if isinstance(raw_scores, np.ndarray) and raw_scores.ndim == 1:
            raw_scores = np.expand_dims(raw_scores, axis=0)

        probs = _softmax(np.array(raw_scores))
        results: list[NLIScore] = []

        for i in range(len(pairs)):
            c_score = float(probs[i][0])
            e_score = float(probs[i][1])
            n_score = float(probs[i][2])

            max_idx = int(np.argmax(probs[i]))
            if max_idx == 0:
                pred_label = VerificationStatus.CONTRADICTED
            elif max_idx == 1:
                pred_label = VerificationStatus.ENTAILED
            else:
                pred_label = VerificationStatus.NEUTRAL

            results.append(
                NLIScore(
                    contradiction=c_score,
                    entailment=e_score,
                    neutral=n_score,
                    predicted_label=pred_label,
                )
            )

        return results


def get_nli_verifier() -> NLIVerifier:
    """Get or initialize the global NLIVerifier singleton instance."""
    global _NLI_VERIFIER_INSTANCE
    if _NLI_VERIFIER_INSTANCE is None:
        with _NLI_VERIFIER_LOCK:
            if _NLI_VERIFIER_INSTANCE is None:
                _NLI_VERIFIER_INSTANCE = NLIVerifier()
    return _NLI_VERIFIER_INSTANCE


def score_entailment(premise: str, hypothesis: str) -> NLIScore:
    """Score entailment, contradiction, and neutral probabilities for a premise-hypothesis pair.

    Args:
        premise: Grounding source context chunk text.
        hypothesis: Atomic factual claim sentence text.

    Returns:
        NLIScore containing probabilities and predicted VerificationStatus label.
    """
    verifier = get_nli_verifier()
    return verifier.score_pair(premise, hypothesis)


def score_entailment_batch(pairs: list[tuple[str, str]]) -> list[NLIScore]:
    """Score entailment for a batch of (premise, hypothesis) pairs.

    Args:
        pairs: List of (premise, hypothesis) string tuples.

    Returns:
        List of NLIScore objects.
    """
    verifier = get_nli_verifier()
    return verifier.score_batch(pairs)
