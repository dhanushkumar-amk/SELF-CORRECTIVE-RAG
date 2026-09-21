"""
LangGraph State Machine Topology & Execution Pipeline for Phases 37-39.

Architecture & Design Decisions:
1. Node Topology:
   - `retrieve`: Executes initial hybrid retrieval + reranking to fetch top relevant context chunks.
   - `generate`: Invokes LLM answer generator to produce citation-forced structured answer.
   - `verify`: Decomposes answer into atomic claims, maps to source chunks, and runs NLI verifier.
   - `targeted_retrieve`: Re-retrieves specifically for failed claims using claim text as query and merges chunks.
   - `finalize`: Calculates overall aggregate response status ("verified", "partially_verified", "failed").

2. Conditional Routing Edge (`correction_router`):
   - Evaluates verification status of claims after `verify_node`.
   - If all claims are verified (or max retries reached), routes to `finalize`.
   - If any claim is ungrounded / contradicted / unverifiable and retries remain, routes to `targeted_retrieve`.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from app.core.logging import get_logger
from app.generation.generator import generate_answer_with_citations
from app.graph.routing import correction_router, get_failed_claims
from app.graph.state import RAGState
from app.graph.targeted_retrieve import targeted_retrieve_node
from app.models.schemas import ClaimWithSource, GeneratedAnswer, RetrievalResult
from app.reranking.reranker import select_relevant_chunks
from app.retrieval.hybrid_retriever import hybrid_search
from app.verification.claim_mapper import map_claims_to_chunks
from app.verification.claim_verifier import get_claim_final_status, verify_claims

logger = get_logger(__name__)

__all__ = [
    "app_graph",
    "build_rag_graph",
    "correction_router",
    "finalize_node",
    "generate_node",
    "get_failed_claims",
    "retrieve_node",
    "targeted_retrieve_node",
    "verify_node",
]


def retrieve_node(state: RAGState) -> dict[str, Any]:
    """LangGraph Node: Execute hybrid retrieval & threshold-based reranking."""
    query = state.get("query", "")
    logger.info("--- LANGGRAPH NODE: RETRIEVE --- (Query: '%s')", query[:50])

    if not query:
        return {"retrieved_chunks": []}

    # 1. Hybrid search (Dense Pinecone + Sparse BM25 + RRF)
    raw_results = hybrid_search(query=query)

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
    chunks = state.get("retrieved_chunks", [])
    logger.info("--- LANGGRAPH NODE: VERIFY ---")

    if not answer:
        return {"claims": []}

    # 1. Map claims to source chunk context
    mapped_claims = map_claims_to_chunks(answer, chunks)

    # 2. Run batched NLI verification engine
    verified_claims = verify_claims(mapped_claims)

    logger.info("Verify node complete: %d claims processed.", len(verified_claims))
    return {"claims": verified_claims}


def finalize_node(state: RAGState) -> dict[str, Any]:
    """LangGraph Node: Calculate aggregate system status and prepare final output."""
    claims = state.get("claims", [])
    logger.info("--- LANGGRAPH NODE: FINALIZE ---")

    if not claims:
        final_status = "unverifiable"
    else:
        statuses = [get_claim_final_status(c) for c in claims]
        if all(s == "verified" for s in statuses):
            final_status = "verified"
        elif any(s in ("contradicted", "needs_review") for s in statuses):
            final_status = "partially_verified"
        else:
            final_status = "unverifiable"

    logger.info("Finalize node complete: System Final Status = '%s'.", final_status)
    return {"final_status": final_status}


def build_rag_graph() -> Any:
    """Construct and compile the LangGraph self-correcting RAG StateGraph topology."""
    workflow = StateGraph(RAGState)

    # Add Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("targeted_retrieve", targeted_retrieve_node)
    workflow.add_node("finalize", finalize_node)

    # Wire Edges
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

    # Loop back from targeted_retrieve -> generate
    workflow.add_edge("targeted_retrieve", "generate")

    # Finalize -> END
    workflow.add_edge("finalize", END)

    return workflow.compile()


# Module-level compiled graph instance
app_graph = build_rag_graph()
