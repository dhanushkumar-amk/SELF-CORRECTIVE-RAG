"""
Scratch script demonstrating traced end-to-end execution of the LangGraph state machine (Phase 37).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.graph.graph import app_graph
from app.graph.state import RAGState
from app.verification.claim_verifier import get_claim_final_status


def run_traced_graph_execution():
    print("==========================================================================")
    print("           LANGGRAPH STATE MACHINE END-TO-END TRACED EXECUTION           ")
    print("==========================================================================")

    initial_state: RAGState = {
        "query": "What dimension vectors does Sentence-Transformers generate and what is it used for?",
        "max_retries": 2,
    }

    print(f"\nInitial Input Query: '{initial_state['query']}'")
    print("\nExecuting StateGraph topology...")
    print("Nodes: [retrieve] -> [generate] -> [verify] -> [correction_router] -> [finalize] -> [END]")
    print("-" * 75)

    final_state = app_graph.invoke(initial_state)

    print("\n--------------------------------------------------------------------------")
    print("                     FINAL TRACED RAGSTATE OUTPUT                         ")
    print("--------------------------------------------------------------------------")
    print(f"1. Query:               '{final_state.get('query')}'")
    print(f"2. Retrieved Chunks:     {len(final_state.get('retrieved_chunks', []))} chunks selected by reranker")
    gen_ans = final_state.get('generated_answer')
    if gen_ans:
        print(f"3. Generated Answer:     {len(gen_ans.claims)} valid claims, insufficient={gen_ans.insufficient_information}")
        print(f"   Provider/Model:      {gen_ans.provider} ('{gen_ans.model_name}')")
    
    claims = final_state.get('claims', [])
    print(f"4. Processed Claims:     {len(claims)} atomic claims mapped & verified:")
    for idx, c in enumerate(claims, 1):
        print(f"   Claim #{idx}:          '{c.claim_text}'")
        print(f"     source_chunk_id:   '{c.source_chunk_id}'")
        print(f"     NLI Status:        {c.verification_status.value.upper() if c.verification_status else 'NONE'}")
        print(f"     Confidence:        {c.confidence:.4f}" if c.confidence is not None else "     Confidence: N/A")
        print(f"     UI Final Status:   '{get_claim_final_status(c)}'")

    print(f"\n5. Retry Count:          {final_state.get('retry_count', 0)}")
    print(f"6. System Final Status:  '{final_state.get('final_status')}'")
    print("==========================================================================\n")


if __name__ == "__main__":
    run_traced_graph_execution()
