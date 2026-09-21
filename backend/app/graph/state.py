"""
LangGraph State Schema Definition for Phase 37.

Architecture & Design Decisions:
1. Shared Execution State:
   `RAGState` is a TypedDict passed sequentially across nodes in the LangGraph state graph.
   It encapsulates query input, retrieved candidate chunks, LLM-generated answer,
   atomic claim verification statuses, retry loop counters, and final system status.
"""

from __future__ import annotations

from typing import TypedDict

from app.models.schemas import ClaimWithSource, GeneratedAnswer, RetrievalResult

__all__ = [
    "RAGState",
]


class RAGState(TypedDict, total=False):
    """Shared state container passed between nodes in the self-correcting RAG state machine."""

    query: str
    retrieved_chunks: list[RetrievalResult]
    generated_answer: GeneratedAnswer | None
    claims: list[ClaimWithSource]
    retry_count: int
    max_retries: int
    final_status: str
    error_message: str | None
