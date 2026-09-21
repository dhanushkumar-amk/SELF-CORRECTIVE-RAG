"""
Unit tests for local NLI DeBERTa CrossEncoder verification engine (Phase 33).
Includes CRITICAL ground-truth sanity check for label order verification.
"""

import pytest

from app.models.schemas import NLIScore, VerificationStatus
from app.verification.nli_verifier import (
    get_nli_verifier,
    score_entailment,
    score_entailment_batch,
)


def test_nli_critical_label_order_ground_truth_sanity_check():
    """CRITICAL GROUND TRUTH SANITY CHECK:

    Verifies that the NLI DeBERTa model index mapping correctly classifies:
    1. Obvious Entailment ("The sky is blue" -> "The sky is blue") -> ENTAILED
    2. Obvious Contradiction ("The sky is blue" -> "The sky is red") -> CONTRADICTED
    """
    # 1. Obvious Entailment test
    entailment_score = score_entailment("The sky is blue.", "The sky is blue.")
    print("\n--- ENTAILMENT SANITY CHECK OUTPUT ---")
    print(f"Raw Probabilities -> contradiction: {entailment_score.contradiction:.6f}, "
          f"entailment: {entailment_score.entailment:.6f}, neutral: {entailment_score.neutral:.6f}")
    print(f"Predicted Label   -> {entailment_score.predicted_label}")

    assert entailment_score.predicted_label == VerificationStatus.ENTAILED
    assert entailment_score.entailment > 0.90

    # 2. Obvious Contradiction test
    contradiction_score = score_entailment("The sky is blue.", "The sky is red.")
    print("--- CONTRADICTION SANITY CHECK OUTPUT ---")
    print(f"Raw Probabilities -> contradiction: {contradiction_score.contradiction:.6f}, "
          f"entailment: {contradiction_score.entailment:.6f}, neutral: {contradiction_score.neutral:.6f}")
    print(f"Predicted Label   -> {contradiction_score.predicted_label}")

    assert contradiction_score.predicted_label == VerificationStatus.CONTRADICTED
    assert contradiction_score.contradiction > 0.90


def test_nli_batched_vs_single_equivalence():
    """Test that batched scoring produces identical results to individual pair scoring."""
    pairs = [
        ("Retrieval-Augmented Generation reduces LLM hallucinations.", "RAG helps decrease hallucinations."),
        ("Pinecone is a cloud vector database.", "Pinecone is a relational SQL database."),
    ]

    single_1 = score_entailment(pairs[0][0], pairs[0][1])
    single_2 = score_entailment(pairs[1][0], pairs[1][1])

    batch_scores = score_entailment_batch(pairs)

    assert len(batch_scores) == 2
    assert batch_scores[0].predicted_label == single_1.predicted_label
    assert pytest.approx(batch_scores[0].entailment, abs=1e-5) == single_1.entailment

    assert batch_scores[1].predicted_label == single_2.predicted_label
    assert pytest.approx(batch_scores[1].contradiction, abs=1e-5) == single_2.contradiction


def test_nli_max_sequence_length_finding():
    """Test that NLI verifier max sequence length is confirmed at 512 tokens."""
    verifier = get_nli_verifier()
    assert verifier.max_seq_length == 512


def test_nli_defensive_empty_premise_raises_error():
    """Test defensive checks on empty or None premise/hypothesis raise ValueError."""
    with pytest.raises(ValueError) as exc_info:
        score_entailment("", "The sky is blue.")
    assert "Premise cannot be empty or None" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info_none:
        score_entailment(None, "The sky is blue.")
    assert "Premise cannot be empty or None" in str(exc_info_none.value)

    with pytest.raises(ValueError) as exc_info_hyp:
        score_entailment("The sky is blue.", "   ")
    assert "Hypothesis cannot be empty or None" in str(exc_info_hyp.value)
