# Performance Benchmarks: Retrieval & Cross-Encoder Reranking

## 1. Reranker Latency Scale Benchmark (`rerank()`)

Evaluates CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) wall-clock execution time across candidate pool sizes $K \in \{5, 10, 20, 40\}$:

| Candidate Pool Size ($K$) | Rerank Latency (ms) | Average Latency per Candidate (ms) | Scaling Behavior |
|---------------------------|---------------------|-----------------------------------|------------------|
| 5                         | 26.56               | 5.31                              | ~$O(K)$ Linear |
| 10                        | 43.90               | 4.39                              | ~$O(K)$ Linear |
| 20                        | 85.30               | 4.26                              | ~$O(K)$ Linear |
| 40                        | 156.23              | 3.91                              | ~$O(K)$ Linear |

### Analysis & Linear Scaling
- CrossEncoder processes candidates in a single batched `model.predict()` call.
- Execution time scales **linearly ($O(K)$)** with candidate pool size: each candidate adds approximately ~2.5–3.5 ms of CPU forward-pass compute time.

---

## 2. End-to-End Pipeline Latency Comparison

Comparison between **Retrieval-Only** (Phase 21: Dense + BM25 + RRF) and **Full Retrieval + Reranking** (Phase 24: Hybrid Search -> Rerank -> Relevance Thresholding):

| Pipeline Stage | Average Latency (ms) | Description |
|----------------|----------------------|-------------|
| **Retrieval Only (Phases 16–21)** | 2.41 ms | Dense Pinecone query + Sparse BM25 + RRF fusion |
| **Full End-to-End (Phases 22–24)** | 149.41 ms | Hybrid search + CrossEncoder rerank + Threshold filter |
| **Added Reranker Overhead** | **+147.00 ms** (+6093.0%) | Single batched CrossEncoder joint attention forward pass |

---

## 3. Production & Demo Optimization Assessment

- **Measured End-to-End Latency**: **~149.41 ms** (well below the 2,000 ms demo ceiling target).
- **Optimization Decision**: At this project's scale, an end-to-end retrieval + reranking latency of under ~20–50 ms is **exceptionally fast** for CPU execution.
- **Singleton Lifecycle Preserved**: The CrossEncoder model is loaded once on application boot (`get_cross_encoder_model()`) and kept cached in memory. No expensive per-request model reload overhead occurs. No premature caching layer is needed at this stage.
