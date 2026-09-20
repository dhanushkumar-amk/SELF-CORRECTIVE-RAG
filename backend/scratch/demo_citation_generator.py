"""
Demonstration script for Phase 26 & Phase 27: Citation-Forced Prompt & Structured Answer Generator.
"""

import sys, json
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.storage import DocumentRegistry
from app.ingestion.pipeline import run_ingestion_pipeline
from app.models.schemas import PageText, RetrievalResult
from app.retrieval import build_bm25_index, hybrid_search
from app.reranking import select_relevant_chunks
from app.generation import generate_answer_with_citations

def run_demo():
    print("=== Phase 26/27 Citation-Forced Answer Generator Demonstration ===")
    
    tmp_path = Path(mkdtemp())
    registry = DocumentRegistry(upload_dir=tmp_path)
    doc_id = "doc_rag_demo"
    registry.save_document(doc_id, "rag_guide.pdf", b"%PDF-demo")
    
    pages = [
        PageText(page_number=1, text="The embedding generator uses sentence-transformers all-MiniLM-L6-v2 producing 384 dimensional vectors for semantic indexing. " * 5, char_count=700),
        PageText(page_number=2, text="BM25 search uses rank_bm25 BM25Okapi for keyword matching over tokenized document terms. " * 5, char_count=600),
        PageText(page_number=3, text="Reciprocal Rank Fusion calculates chunk relevance using k=60 rank decay constant across retrieval lists. " * 5, char_count=650),
        PageText(page_number=4, text="Pinecone manages cloud vector storage and cosine similarity vector queries. " * 5, char_count=500),
        PageText(page_number=5, text="LangGraph state machine executes self-correction when NLI verification detects hallucinated claims. " * 5, char_count=700),
    ]
    
    mock_pc = MagicMock()
    mock_pc.delete_vectors.return_value = None
    mock_pc.upsert_vectors.return_value = {"upserted_count": 5, "successful_ids": [f"v{i}" for i in range(5)], "failed_ids": []}
    
    with patch("app.ingestion.pipeline.extract_raw_pages", return_value=pages), \
         patch("app.ingestion.pipeline.get_pinecone_client", return_value=mock_pc), \
         patch("app.retrieval.dense_search.embed_query", return_value=[0.1] * 384):
        
        run_ingestion_pipeline(doc_id, registry=registry, verify_upsert=False)
        bm25_idx = build_bm25_index(registry=registry)
        chunks = registry.get_chunks(doc_id)
        
        def mock_dense(vector, top_k=5, filter=None, include_metadata=True, namespace=""):
            return [
                {"id": f"{doc_id}::{c.chunk_id}", "score": 0.95 - (idx*0.05), "metadata": c.to_pinecone_metadata(filename="rag_guide.pdf")}
                for idx, c in enumerate(chunks[:top_k])
            ]
        mock_pc.query_vectors.side_effect = mock_dense

        demo_queries = [
            ("What model does sentence-transformers use for dense embeddings?", True),
            ("How does BM25 calculate keyword relevance scores?", True),
            ("How does LangGraph state machine handle hallucinated answers?", True),
            ("What is the capital city of Japan and quantum physics string theory?", False),
        ]
        
        for q_text, is_answerable in demo_queries:
            print(f"\n" + "="*80)
            print(f"QUERY: '{q_text}'")
            print("="*80)
            
            # Step 1: Hybrid Search + Rerank
            hyb = hybrid_search(q_text, top_k=5, pinecone_client=mock_pc, bm25_index=bm25_idx)
            rerank_res = select_relevant_chunks(q_text, hyb, min_score=-2.0, top_n=3)
            
            print(f"Reranked Chunks Found: {len(rerank_res.chunks)} (Reason: {rerank_res.reason})")
            
            # Step 2: Citation-Forced Generation
            ans = generate_answer_with_citations(q_text, rerank_res.chunks)
            
            print(f"Provider: {ans.provider} | Model: {ans.model_name} | Latency: {ans.latency_ms:.2f} ms")
            print(f"Insufficient Information Flag: {ans.insufficient_information}")
            print(f"Valid Claims ({len(ans.claims)}):")
            for idx, c in enumerate(ans.claims):
                print(f"  [{idx+1}] Claim: \"{c.claim_text}\"")
                print(f"      Source Chunk ID: {c.source_chunk_id}")
            if ans.unverified_claims:
                print(f"Quarantined Unverified Claims ({len(ans.unverified_claims)}):")
                for c in ans.unverified_claims:
                    print(f"  [UNVERIFIED] Claim: \"{c.claim_text}\" -> Cited Chunk ID: {c.source_chunk_id}")

if __name__ == "__main__":
    run_demo()
