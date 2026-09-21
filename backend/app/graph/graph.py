"""
LangGraph State Machine Topology & Execution Pipeline for Phases 37-41.

Architecture & Design Decisions:
1. Complete Node Topology:
   - `retrieve`: Executes initial hybrid retrieval + CrossEncoder reranking to fetch top context chunks.
   - `generate`: Invokes LLM answer generator to produce citation-forced structured answer.
   - `verify`: Decomposes answer into atomic claims, maps to source chunks, and runs NLI verifier.
   - `targeted_retrieve`: Re-retrieves specifically for failed claims using claim text as query and merges chunks.
   - `regenerate`: Executes targeted partial re-generation for failed claims using enriched context.
   - `finalize`: Calculates overall aggregate response status ("fully_verified", "partially_verified", "unverifiable") and synthesizes answer text.

2. Loop Topology:
   - `retrieve` -> `generate` -> `verify`
   - `verify` -> `correction_router`:
       - If all claims verified or retries exhausted -> `finalize`
       - If failed claims & retries remain -> `targeted_retrieve`
   - `targeted_retrieve` -> `regenerate` -> `verify` (loops back to verify updated claims)
   - `finalize` -> `END`
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from app.core.logging import get_logger
from app.generation.generator import generate_answer_with_citations
from app.graph.finalize import finalize_node
from app.graph.regenerate import regenerate_node
from app.graph.routing import correction_router, get_failed_claims
from app.graph.state import RAGState
from app.graph.targeted_retrieve import targeted_retrieve_node
from app.models.schemas import GeneratedAnswer
from app.reranking.reranker import select_relevant_chunks
from app.retrieval.hybrid_retriever import hybrid_search
from app.verification.claim_mapper import map_claims_to_chunks
from app.verification.claim_verifier import verify_claims

logger = get_logger(__name__)

__all__ = [
    "app_graph",
    "build_rag_graph",
    "correction_router",
    "finalize_node",
    "generate_node",
    "get_failed_claims",
    "regenerate_node",
    "retrieve_node",
    "targeted_retrieve_node",
    "verify_node",
]


def retrieve_node(state: RAGState) -> dict[str, Any]:
    """LangGraph Node: Execute hybrid retrieval & threshold-based reranking."""
    query = state.get("query", "")
    logger.info("--- LANGGRAPH NODE: RETRIEVE --- (Query: '%s')", query[:50])

    doc_id = state.get("document_id")
    filter_dict = {"document_id": doc_id} if doc_id else None

    # 1. Hybrid search (Dense Pinecone + Sparse BM25 + RRF)
    raw_results = hybrid_search(query=query, filter=filter_dict)

    # 2. Threshold-based CrossEncoder reranking
    rerank_res = select_relevant_chunks(query=query, candidates=raw_results)

    logger.info("Retrieve node complete: %d relevant chunks selected.", len(rerank_res.chunks))
    return {"retrieved_chunks": rerank_res.chunks}


def generate_node(state: RAGState) -> dict[str, Any]:
    """LangGraph Node: Generate citation-forced answer from query and context chunks."""
    query = state.get("query", "")
    chunks = state.get("retrieved_chunks", [])
    logger.info("--- LANGGRAPH NODE: GENERATE --- (Chunks: %d)", len(chunks))

    if not chunks:
        logger.warning("Generate node called with 0 retrieved chunks. Returning empty answer.")
        return {
            "generated_answer": GeneratedAnswer(
                claims=[],
                insufficient_information=True,
                raw_response="No relevant context chunks found.",
            )
        }

    answer = generate_answer_with_citations(query=query, chunks=chunks)
    logger.info("Generate node complete: %d raw claims produced.", len(answer.claims) + len(answer.unverified_claims))
    return {"generated_answer": answer}


def verify_node(state: RAGState) -> dict[str, Any]:
    """LangGraph Node: Decompose claims, map to source chunks, and run NLI verification."""
    answer = state.get("generated_answer")
    existing_claims = state.get("claims", [])
    chunks = state.get("retrieved_chunks", [])
    logger.info("--- LANGGRAPH NODE: VERIFY ---")

    # If state already has claims from a partial regeneration pass, use those
    if existing_claims:
        claims_to_verify = existing_claims
    elif answer:
        claims_to_verify = map_claims_to_chunks(answer, chunks)
    else:
        return {"claims": []}

    # Run batched NLI verification engine across claims
    verified_claims = verify_claims(claims_to_verify)

    logger.info("Verify node complete: %d claims verified.", len(verified_claims))
    return {"claims": verified_claims}


def build_rag_graph() -> Any:
    """Construct and compile the LangGraph self-correcting RAG StateGraph topology."""
    workflow = StateGraph(RAGState)

    # Add Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("targeted_retrieve", targeted_retrieve_node)
    workflow.add_node("regenerate", regenerate_node)
    workflow.add_node("finalize", finalize_node)

    # Wire Entry & Base Edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "verify")

    # Conditional Routing Edge
    workflow.add_conditional_edges(
        "verify",
        correction_router,
        {
            "finalize": "finalize",
            "targeted_retrieve": "targeted_retrieve",
        },
    )

    # Correction Loop Back Edges
    workflow.add_edge("targeted_retrieve", "regenerate")
    workflow.add_edge("regenerate", "verify")

    # Terminal Edge
    workflow.add_edge("finalize", END)

    return workflow.compile()


# Module-level compiled graph instance
app_graph = build_rag_graph()
