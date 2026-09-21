"""
Scratch script to measure non-atomic claim frequency across real LLM generation runs (Phase 30-31 analysis).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.generation.generator import generate_answer_with_citations
from app.models.schemas import RetrievalResult
from app.verification.claim_mapper import map_claims_to_chunks
from app.verification.claim_splitter import is_atomic_claim

sample_chunks = [
    RetrievalResult(
        chunk_id="chunk_pdf_p3_1",
        score=0.92,
        metadata={
            "source_text": "Sentence-Transformers all-MiniLM-L6-v2 maps sentences & paragraphs to a 384 dimensional dense vector space. It is designed for semantic search and feature extraction.",
            "page_number": 3,
            "page_number_end": 3,
            "document_id": "doc_test_pdf_1",
            "filename": "rag_architecture.pdf",
        },
    ),
    RetrievalResult(
        chunk_id="chunk_pdf_p5_2",
        score=0.89,
        metadata={
            "source_text": "Reciprocal Rank Fusion (RRF) combines search results from dense vector search and BM25 sparse search using the formula score = sum(1 / (60 + rank)).",
            "page_number": 5,
            "page_number_end": 5,
            "document_id": "doc_test_pdf_1",
            "filename": "rag_architecture.pdf",
        },
    ),
    RetrievalResult(
        chunk_id="chunk_pdf_p8_3",
        score=0.85,
        metadata={
            "source_text": "Pinecone is a cloud-native vector database optimized for ultra-low latency similarity queries across millions of vectors.",
            "page_number": 8,
            "page_number_end": 8,
            "document_id": "doc_test_pdf_1",
            "filename": "rag_architecture.pdf",
        },
    ),
]

test_queries = [
    "What dimension vectors does Sentence-Transformers generate and what is it used for?",
    "How does Reciprocal Rank Fusion combine search results?",
    "What is Pinecone and what is it optimized for?",
]

def run_analysis():
    total_claims = 0
    atomic_count = 0
    non_atomic_count = 0
    sample_mapped = []

    print("--- Running Real LLM Generation & Claim Atomicity Analysis ---")
    for i, q in enumerate(test_queries, 1):
        try:
            answer = generate_answer_with_citations(q, sample_chunks, temperature=0.0)
            mapped = map_claims_to_chunks(answer, sample_chunks)
            if not sample_mapped and mapped:
                sample_mapped = mapped

            raw_claims = answer.claims + answer.unverified_claims
            print(f"\nQuery {i}: '{q}'")
            print(f"Generated {len(raw_claims)} raw claims, {len(mapped)} mapped atomic claims:")

            for c in raw_claims:
                total_claims += 1
                is_atom = is_atomic_claim(c.claim_text)
                if is_atom:
                    atomic_count += 1
                    print(f"  [ATOMIC] '{c.claim_text}' [{c.source_chunk_id}]")
                else:
                    non_atomic_count += 1
                    print(f"  [NON-ATOMIC] '{c.claim_text}' [{c.source_chunk_id}]")

        except Exception as exc:
            print(f"Query {i} failed: {exc}")

    print("\n================== FREQUENCY SUMMARY ==================")
    print(f"Total Raw Claims Evaluated: {total_claims}")
    print(f"Atomic (Single-Sentence) Claims: {atomic_count}")
    print(f"Non-Atomic (Compound Multi-Sentence) Claims: {non_atomic_count}")
    freq_pct = (non_atomic_count / total_claims * 100.0) if total_claims > 0 else 0.0
    print(f"Non-Atomic Frequency Rate: {freq_pct:.1f}%")
    print("=======================================================\n")

    if sample_mapped:
        print("--- Real Full Mapping Output Example (ClaimWithSource) ---")
        for idx, item in enumerate(sample_mapped, 1):
            print(f"\n[ClaimWithSource #{idx}]")
            print(f"  claim_text:       '{item.claim_text}'")
            print(f"  source_chunk_id:  '{item.source_chunk_id}'")
            print(f"  is_valid_source:  {item.is_valid_source}")
            print(f"  page_number:      {item.page_number}")
            print(f"  filename:         '{item.filename}'")
            print(f"  source_text:      '{item.source_text[:100]}...'")

if __name__ == "__main__":
    run_analysis()
