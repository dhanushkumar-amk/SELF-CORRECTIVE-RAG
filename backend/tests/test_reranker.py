"""
Unit tests for Phase 22: Local Cross-Encoder Reranker Integration.
"""

from unittest.mock import MagicMock, patch
import pytest

from app.models.schemas import Chunk, DocumentStatus, PageText, RetrievalResult
from app.reranking import (
    get_cross_encoder_model,
    get_reranker_info,
    rerank,
)
from app.retrieval import build_bm25_index, hybrid_search


def test_cross_encoder_max_sequence_length_finding():
    """Document and verify max sequence length behavior of cross-encoder/ms-marco-MiniLM-L-6-v2."""
    info = get_reranker_info()
    assert info["model_name"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # MiniLM L6 cross-encoder has a 512 token max position sequence length
    assert info["max_length"] == 512
    assert "Tokenizer" in info["tokenizer_type"] or "Bert" in info["tokenizer_type"]


def test_rerank_empty_and_small_candidates():
    """Verify rerank handles empty candidate lists and candidates smaller than top_n."""
    # 1. Empty candidate list returns empty immediately
    assert rerank("quantum computing", [], top_n=5) == []
    assert rerank("", [RetrievalResult(chunk_id="c1", score=0.5)], top_n=5) == []

    # 2. Candidates count < top_n returns all candidates sorted without error
    cands = [
        RetrievalResult(
            chunk_id="c1",
            score=0.1,
            metadata={"source_text": "Chocolate baking recipes require cocoa powder and flour."},
        ),
        RetrievalResult(
            chunk_id="c2",
            score=0.1,
            metadata={"source_text": "Quantum superposition allows qubits to exist in multiple states."},
        ),
    ]

    res = rerank("quantum qubit superposition", cands, top_n=5)
    assert len(res) == 2
    # Quantum chunk (c2) should rank #1 with a positive logit score, while chocolate chunk (c1) gets negative score
    assert res[0].chunk_id == "c2"
    assert res[0].score > res[1].score
    assert res[0].score > 0.0
    assert res[1].score < 0.0


def test_rerank_reorders_rrf_candidates(tmp_path):
    """Verify cross-encoder reranking changes candidate rank order when joint attention detects better semantic fit."""
    # Setup candidate list where candidate #2 is far more relevant to the query than candidate #1
    c1 = RetrievalResult(
        chunk_id="c_general_ml",
        score=0.032,  # Higher initial RRF score
        metadata={"source_text": "Machine learning algorithms learn statistical patterns from data."},
    )
    c2 = RetrievalResult(
        chunk_id="c_exact_alphafold",
        score=0.016,  # Lower initial RRF score
        metadata={"source_text": "DeepMind AlphaFold 3 predicts 3D protein-ligand biomolecular complex interactions."},
    )

    query = "How does AlphaFold predict 3D protein structures?"
    initial_candidates = [c1, c2]

    # RRF initial order: c_general_ml at rank 1, c_exact_alphafold at rank 2
    assert initial_candidates[0].chunk_id == "c_general_ml"

    reranked = rerank(query, initial_candidates, top_n=2)
    assert len(reranked) == 2

    # Cross-encoder reranked order: c_exact_alphafold MUST move to rank 1!
    assert reranked[0].chunk_id == "c_exact_alphafold"
    assert reranked[1].chunk_id == "c_general_ml"
    assert reranked[0].score > reranked[1].score
