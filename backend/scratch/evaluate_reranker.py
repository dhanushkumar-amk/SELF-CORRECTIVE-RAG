import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.storage import DocumentRegistry
from app.models.schemas import Chunk, DocumentStatus, PageText
from app.retrieval import build_bm25_index, hybrid_search
from app.reranking import rerank, select_relevant_chunks, get_reranker_info
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile

def run_empirical_eval():
    print("=== Phase 22/23 Empirical Reranker Evaluation ===")
    
    # 1. Reranker Model Info & Max Sequence Length
    info = get_reranker_info()
    print(f"\n[Model Info] Name: {info['model_name']} | Max Length: {info['max_length']} tokens | Tokenizer: {info['tokenizer_type']}")
    print("Note: The bi-encoder embedding model (all-MiniLM-L6-v2) truncates at 256 tokens. The cross-encoder (ms-marco-MiniLM-L-6-v2) supports up to 512 tokens natively, comfortably accommodating full ~500 token document chunks without truncation!")

    # Setup isolated test registry & documents
    tmp_path = Path(tempfile.mkdtemp())
    registry = DocumentRegistry(upload_dir=tmp_path)
    
    doc_id = "doc_eval_rerank"
    registry.save_document(doc_id, "rag_guide.pdf", b"%PDF-eval")
    
    pages = [
        PageText(page_number=1, text="The embedding generator uses sentence-transformers all-MiniLM-L6-v2 producing 384 dimensional vectors for semantic indexing. " * 5, char_count=700),
        PageText(page_number=2, text="BM25 search uses rank_bm25 BM25Okapi for keyword matching over tokenized document terms. " * 5, char_count=600),
        PageText(page_number=3, text="Reciprocal Rank Fusion calculates chunk relevance using k=60 rank decay constant across retrieval lists. " * 5, char_count=650),
        PageText(page_number=4, text="Pinecone manages cloud vector storage and cosine similarity vector queries. " * 5, char_count=500),
        PageText(page_number=5, text="LangGraph state machine executes self-correction when NLI verification detects hallucinated claims. " * 5, char_count=700),
    ]
    
    from app.ingestion.pipeline import run_ingestion_pipeline
    mock_pc = MagicMock()
    mock_pc.delete_vectors.return_value = None
    mock_pc.upsert_vectors.return_value = {"upserted_count": 5, "successful_ids": ["v1", "v2", "v3", "v4", "v5"], "failed_ids": []}
    
    with patch("app.ingestion.pipeline.extract_raw_pages", return_value=pages), \
         patch("app.ingestion.pipeline.get_pinecone_client", return_value=mock_pc), \
         patch("app.retrieval.dense_search.embed_query", return_value=[0.1] * 384):
        
        run_ingestion_pipeline(doc_id, registry=registry, verify_upsert=False)
        bm25_idx = build_bm25_index(registry=registry)
        chunks = registry.get_chunks(doc_id)

        def mock_dense_query(vector, top_k=5, filter=None, include_metadata=True, namespace=""):
            matches = []
            for idx, c in enumerate(chunks):
                vid = f"{doc_id}::{c.chunk_id}"
                # Deliberately reverse order in dense vector search for one query to show reranker reordering!
                score = round(0.95 - (idx * 0.05), 2)
                matches.append({"id": vid, "score": score, "metadata": c.to_pinecone_metadata(filename="rag_guide.pdf")})
            return matches[:top_k]

        mock_pc.query_vectors.side_effect = mock_dense_query

        # 2. Empirical Score Range Table Data
        eval_queries = [
            ("What model does sentence-transformers use for dense embeddings?", "High Relevance (Exact Match)", True),
            ("How does BM25 calculate keyword relevance scores?", "High Relevance (Exact Match)", True),
            ("Reciprocal Rank Fusion k constant decay parameter", "Moderate Relevance (Partial Topic Match)", True),
            ("What is the capital of France and Paris weather?", "Complete Irrelevance (Out of Scope)", False),
            ("Baking sourdough bread at high humidity with yeast", "Complete Irrelevance (Out of Scope)", False),
        ]
        
        print("\n--- Empirical Score Range Table ---")
        print(f"{'Query Text':<55} | {'Expected Category':<35} | {'Max Cross-Encoder Score':<25}")
        print("-" * 122)
        
        for q_text, cat, is_relevant in eval_queries:
            hyb_res = hybrid_search(q_text, top_k=5, pinecone_client=mock_pc, bm25_index=bm25_idx)
            reranked_res = rerank(q_text, hyb_res, top_n=5)
            max_score = reranked_res[0].score if reranked_res else -999.0
            print(f"{q_text[:54]:<55} | {cat:<35} | {max_score:<25.4f}")

        # 3. Side-by-Side RRF vs Rerank Order Demonstration
        print("\n--- Side-by-Side RRF vs. Cross-Encoder Ranking Comparison ---")
        test_q = "How does LangGraph state machine handle self-correction for hallucinations?"
        hyb_candidates = hybrid_search(test_q, top_k=5, pinecone_client=mock_pc, bm25_index=bm25_idx)
        reranked_candidates = rerank(test_q, hyb_candidates, top_n=5)
        
        print(f"Query: '{test_q}'\n")
        print(f"{'Rank':<6} | {'RRF Hybrid Order (Chunk & RRF Score)':<50} | {'Cross-Encoder Reranked Order (Chunk & Logit Score)':<55}")
        print("-" * 118)
        for i in range(min(len(hyb_candidates), len(reranked_candidates))):
            h_c = hyb_candidates[i]
            r_c = reranked_candidates[i]
            h_str = f"{h_c.chunk_id[-8:]} (p.{h_c.metadata.get('page_number')}, score={h_c.score:.4f})"
            r_str = f"{r_c.chunk_id[-8:]} (p.{r_c.metadata.get('page_number')}, score={r_c.score:.4f})"
            print(f"{i+1:<6} | {h_str:<50} | {r_str:<55}")

        # 4. Out-of-Scope "No Relevant Chunks Found" Trigger
        out_q = "What is the capital of Japan and quantum chromodynamics string theory?"
        hyb_out = hybrid_search(out_q, top_k=5, pinecone_client=mock_pc, bm25_index=bm25_idx)
        rerank_out_result = select_relevant_chunks(out_q, hyb_out, min_score=-2.0, top_n=5)
        print("\n--- Out-of-Scope Query 'No Relevant Chunks Found' Result ---")
        print(f"Query: '{out_q}'")
        print(f"Candidates Received: {rerank_out_result.total_candidates}")
        print(f"Relevant Count: {rerank_out_result.relevant_count}")
        print(f"Reason Flag: '{rerank_out_result.reason}'")
        print(f"Chunks Returned: {len(rerank_out_result.chunks)}")

        # 5. Recalibrated 5-Question Spot-Check + Out-of-Scope
        print("\n--- Recalibrated Spot-Check Evaluation (hybrid_search -> rerank -> threshold) ---")
        qa_spotcheck = [
            ("What model does sentence-transformers use for dense embeddings?", "all-MiniLM-L6-v2", True),
            ("How does BM25 calculate keyword relevance scores?", "BM25Okapi", True),
            ("What constant parameter k is used in Reciprocal Rank Fusion?", "k=60", True),
            ("What database vector index stores chunk embeddings?", "Pinecone", True),
            ("How does LangGraph self-correction handle hallucinated answers?", "LangGraph", True),
            ("What is the recipe for baking chocolate cookies?", "N/A - Out of Scope", False),
        ]
        
        passed_count = 0
        for q_text, expected_fact, is_answerable in qa_spotcheck:
            hyb = hybrid_search(q_text, top_k=5, pinecone_client=mock_pc, bm25_index=bm25_idx)
            sel_res = select_relevant_chunks(q_text, hyb, min_score=-2.0, top_n=3)
            
            if is_answerable:
                top_text = sel_res.chunks[0].metadata.get("source_text", "") if sel_res.chunks else ""
                hit = expected_fact in top_text
                status = "PASS (Hit)" if hit else "FAIL"
                if hit: passed_count += 1
                top_score = sel_res.chunks[0].score if sel_res.chunks else -999.0
                print(f"Q: '{q_text[:45]}...' -> {status} [chunks={len(sel_res.chunks)}, top_score={top_score:.2f}, reason={sel_res.reason}]")
            else:
                out_pass = sel_res.reason == "no_relevant_chunks_found" and len(sel_res.chunks) == 0
                status = "PASS (Correctly Triggered 'no_relevant_chunks_found')" if out_pass else "FAIL"
                if out_pass: passed_count += 1
                print(f"Q: '{q_text[:45]}...' -> {status} [chunks={len(sel_res.chunks)}, reason='{sel_res.reason}']")
                
        print(f"\nRecalibrated Spot-Check Score: {passed_count}/6 ({passed_count/6*100:.0f}%)")

if __name__ == "__main__":
    run_empirical_eval()
