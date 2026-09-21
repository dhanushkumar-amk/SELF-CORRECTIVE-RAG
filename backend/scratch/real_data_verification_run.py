"""
Real-data validation script executing verify_claims() across a spectrum of claims (Phase 35 Task 6).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.models.schemas import ClaimWithSource, RetrievalResult, VerificationStatus
from app.verification.claim_verifier import verify_claims


def run_real_data_validation():
    print("==========================================================================")
    print("         REAL-DATA NLI CLAIM VERIFICATION VALIDATION RUN (PHASE 35)       ")
    print("==========================================================================")

    # Context chunk premises retrieved from test PDF
    chunk_1_text = (
        "SentenceTransformers all-MiniLM-L6-v2 maps sentences & paragraphs to a 384 dimensional "
        "dense vector space. It is designed for high-performance semantic search and feature extraction."
    )
    chunk_2_text = (
        "Reciprocal Rank Fusion (RRF) combines search results from dense vector search and BM25 sparse search "
        "using the formula score = sum(1 / (60 + rank))."
    )

    test_claims = [
        # Claim 1: Well-grounded, exact factual entailment
        ClaimWithSource(
            claim_text="SentenceTransformers models map sentences to a 384-dimensional dense vector space.",
            source_chunk_id="chunk_1",
            source_text=chunk_1_text,
            is_valid_source=True,
            page_number=3,
            filename="rag_architecture.pdf",
        ),
        # Claim 2: Well-grounded, semantic entailment
        ClaimWithSource(
            claim_text="Reciprocal Rank Fusion combines dense vector search and BM25 search result lists.",
            source_chunk_id="chunk_2",
            source_text=chunk_2_text,
            is_valid_source=True,
            page_number=5,
            filename="rag_architecture.pdf",
        ),
        # Claim 3: Deliberately contradicted claim (states incorrect formula and constant)
        ClaimWithSource(
            claim_text="RRF multiplies rank scores together using a constant of 100 instead of summing reciprocals.",
            source_chunk_id="chunk_2",
            source_text=chunk_2_text,
            is_valid_source=True,
            page_number=5,
            filename="rag_architecture.pdf",
        ),
        # Claim 4: Deliberately ungrounded / neutral claim (introduces unmentioned facts)
        ClaimWithSource(
            claim_text="SentenceTransformers models require GPUs manufactured exclusively by NVIDIA.",
            source_chunk_id="chunk_1",
            source_text=chunk_1_text,
            is_valid_source=True,
            page_number=3,
            filename="rag_architecture.pdf",
        ),
        # Claim 5: Invalid/hallucinated chunk citation ID (unverifiable)
        ClaimWithSource(
            claim_text="Pinecone is a cloud-native vector database.",
            source_chunk_id="fake_chunk_999",
            source_text=None,
            is_valid_source=False,
            page_number=None,
            filename=None,
        ),
    ]

    verified = verify_claims(test_claims)

    print("\n--------------------------------------------------------------------------")
    print("                      DETAILED VERIFICATION RESULTS                      ")
    print("--------------------------------------------------------------------------")

    for i, c in enumerate(verified, 1):
        print(f"\n[Claim #{i}]")
        print(f"  Claim Text:          '{c.claim_text}'")
        print(f"  Source Chunk ID:     '{c.source_chunk_id}' (is_valid_source={c.is_valid_source})")
        print(f"  Verification Status: {c.verification_status.value.upper() if c.verification_status else 'NONE'}")
        print(f"  Confidence Score:    {c.confidence:.4f}" if c.confidence is not None else "  Confidence Score: N/A")
        if c.source_text:
            print(f"  Source Excerpt:      '{c.source_text[:90]}...'")
        else:
            print(f"  Source Excerpt:      [None - Invalid Chunk ID]")

    print("\n--------------------------------------------------------------------------")
    print("                            SUMMARY TABLE                                ")
    print("--------------------------------------------------------------------------")
    print(f"{'#':<3} | {'Status':<14} | {'Conf':<6} | {'Claim Excerpt':<45}")
    print("-" * 75)
    for i, c in enumerate(verified, 1):
        status_str = c.verification_status.value.upper() if c.verification_status else "NONE"
        conf_str = f"{c.confidence:.4f}" if c.confidence is not None else "N/A"
        claim_short = (c.claim_text[:42] + "...") if len(c.claim_text) > 42 else c.claim_text
        print(f"{i:<3} | {status_str:<14} | {conf_str:<6} | {claim_short:<45}")
    print("==========================================================================\n")


if __name__ == "__main__":
    run_real_data_validation()
