# 🔄 Self-Correcting RAG with Hallucination Detection

A production-grade Retrieval-Augmented Generation system that **detects and self-corrects hallucinations** using a multi-stage verification pipeline powered by LangGraph.

## Run the workspace

With Docker and Docker Compose, start both services from the repository root.

```bash
docker compose up --build
```

Open [Groundwork](http://localhost:3000). Choose **Try an example PDF**, wait for **Ready**, then choose **Find a specific detail** and send the question. Open a numbered citation to see its filename, page, and source text. The API reference is at [localhost:8000/docs](http://localhost:8000/docs). Documents persist in the Compose data volume.

### Local development

Use Python 3.12 and Node.js 20.9 or later. In one terminal, run:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

On PowerShell, activate with `.venv\Scripts\Activate.ps1` and copy with `Copy-Item .env.example .env`. In a second terminal, run:

```bash
cd frontend
npm ci
npm run dev
```

The browser uses the Next.js proxy by default. Set `BACKEND_URL` in `frontend/.env.local` when your backend is on a different port or host. Set `NEXT_PUBLIC_API_URL` only for a separate browser-accessible API origin and configure the backend's `CORS_ORIGINS` as a JSON array of allowed origins.

### Search and verification modes

- **Local passage search** is the default in Docker Compose and `backend/.env.example`. It returns verbatim PDF excerpts using keyword ranking and shows the matching pages. It works without API keys or model downloads after dependency installation.
- **Cloud verification** uses Pinecone and Groq or Gemini to retrieve evidence and generate answers, then runs local reranking and NLI verification. Set `RAG_MODE=cloud`, `REQUIRE_API_KEYS=true`, and configure `PINECONE_API_KEY` plus at least one of `GROQ_API_KEY` or `GEMINI_API_KEY` in `backend/.env`. For Compose, also set `RAG_MODE=cloud` in the shell when starting it. The first cloud-mode ingestion or query can download Hugging Face models. Re-index PDFs previously processed in local mode to populate the cloud vector index.

| Setting | Purpose |
| --- | --- |
| `PINECONE_INDEX_NAME`, `PINECONE_ENVIRONMENT` | Target index and region. Defaults are `self-correcting-rag` and `us-east-1`. |
| `MAX_UPLOAD_SIZE_MB`, `UPLOAD_DIR` | PDF upload limit (20 MB) and local storage directory. |
| `RATE_LIMIT_ENABLED`, `RATE_LIMIT_QUERY_PER_MINUTE` | Per-client query rate limit (10 per minute by default). |
| `CORRECTION_MAX_RETRIES` | Maximum corrective passes in cloud mode (2 by default). |
| `ENVIRONMENT`, `LOG_LEVEL` | Runtime environment and logging level. |
| `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY` | Optional LangSmith tracing. |

The Compose setup binds both services to localhost and uses a single backend process with a shared document library.

### Checks

From `backend/`, run `REQUIRE_API_KEYS=false RAG_MODE=cloud python -m pytest tests`. Some model integration tests require Hugging Face downloads, and live Pinecone tests require credentials. From `frontend/`, run `npm run lint` and `npm run build`.

## API Standards & Rate Limiting

- **Versioned surface:** all endpoints live under `/api/v1/...`; a breaking change would introduce `/api/v2` alongside it.
- **Uniform error contract:** every failure returns `{"error": <code>, "detail": <msg>, "status_code": <n>}` with an explicit mapping — validation `422`, ingestion permanent failure `422`, transient infrastructure failure `503`, generation failure `503`, rate limit `429`, unhandled errors `500` (internal details never leak to clients).
- **Request tracing:** every response carries `X-Correlation-ID` (generated or propagated) and `X-Process-Time-Ms`; the same ID is written into structured logs for end-to-end tracing.
- **Rate limiting:** `/api/v1/query` is capped at `RATE_LIMIT_QUERY_PER_MINUTE` (default 10) requests per minute per client IP via an in-process sliding window. Rationale: each query can trigger multiple billed LLM invocations (initial generation + up to 2 corrective re-generations), so the cap protects Groq/Gemini free-tier quotas. Exceeded requests receive `429` + `Retry-After`. For multi-replica deployments, swap the in-process store for a shared one (e.g. Redis via slowapi/limits).

---

## Architecture Overview

```
┌─────────────┐     ┌──────────────────────────────────────────────────────┐
│             │     │                   FastAPI Backend                    │
│   Next.js   │────▶│                                                      │
│  Frontend   │◀────│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐ │
│             │     │  │ Ingestion│  │Retrieval │  │   Generation       │ │
└─────────────┘     │  │ Pipeline │  │(Pinecone │  │   (Gemini/Groq)    │ │
                    │  │ (PDF →   │  │ + BM25 + │  │                    │ │
                    │  │  Chunks) │  │   RRF)   │  └────────┬───────────┘ │
                    │  └──────────┘  └──────────┘           │             │
                    │                                       ▼             │
                    │  ┌────────────────────────────────────────────────┐  │
                    │  │            LangGraph State Machine             │  │
                    │  │  ┌────────┐  ┌──────────┐  ┌──────────────┐   │  │
                    │  │  │Generate│─▶│  Verify  │─▶│  Correct /   │   │  │
                    │  │  │ Answer │  │  (NLI)   │  │  Re-generate │   │  │
                    │  │  └────────┘  └──────────┘  └──────────────┘   │  │
                    │  └────────────────────────────────────────────────┘  │
                    └──────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer         | Technology                                    |
| ------------- | --------------------------------------------- |
| Frontend      | Next.js 16 (App Router), TypeScript, Tailwind CSS |
| Backend       | Python, FastAPI                               |
| Orchestration | LangChain + LangGraph                        |
| Vector DB     | Pinecone (managed)                            |
| Retrieval     | Dense (embeddings) + Sparse (BM25) + RRF      |
| Re-ranking    | Cross-encoder / sentence-transformers          |
| Verification  | NLI-based hallucination detection              |
| LLMs          | Google Gemini / Groq                           |


## Project Structure

```
self-correcting-rag/
├── backend/
│   └── app/
│       ├── main.py            # FastAPI entrypoint
│       ├── core/              # Config & logging
│       ├── api/routes/        # HTTP endpoints
│       ├── ingestion/         # Document processing pipeline
│       ├── retrieval/         # Pinecone + BM25 hybrid search
│       ├── reranking/         # Cross-encoder re-ranking
│       ├── generation/        # LLM answer generation
│       ├── verification/     # NLI hallucination detection
│       ├── graph/             # LangGraph state machine
│       └── models/            # Pydantic schemas
├── frontend/                  # Next.js app
├── docker-compose.yml
└── README.md
```

---

## Project Roadmap

### Phase 1 — Foundation
- [x] **Phase 1:** Project scaffolding & repo structure

### Phase 2–5 — Core Infrastructure
- [x] **Phase 2:** Backend configuration & environment management
- [x] **Phase 3:** Database schema & Pinecone index setup
- [x] **Phase 4:** Authentication & API key management
- [x] **Phase 5:** Git workflow & branching strategy

> 🚀 **Setup & Foundations Phase Complete (Phases 1–5)** — Ready for Phase 6 Document Ingestion!


### Phase 6–15 — Document Ingestion Pipeline
- [x] **Phase 6:** PDF upload endpoint
- [x] **Phase 7:** PDF text extraction (pypdf)
- [x] **Phase 8:** Text cleaning & normalization
- [x] **Phase 9:** Text chunking strategy (recursive character splitting)
- [x] **Phase 10:** Chunk metadata enrichment & Pinecone constraint validation
- [x] **Phase 11:** Embedding generation (sentence-transformers)
- [x] **Phase 12:** Embedding generation pipeline (extract -> clean -> chunk -> embed orchestrator)
- [x] **Phase 13:** Pinecone upsert pipeline
- [x] **Phase 14:** Ingestion error handling & retries
- [x] **Phase 15:** Ingestion status tracking

> 📄 **Document Ingestion (Phases 6–15) fully complete** — Retrieval Layer next (Phase 16–21)

### Phase 16–21 — Hybrid Retrieval
- [x] **Phase 16:** Dense retrieval (Pinecone vector search)
- [x] **Phase 17:** Sparse retrieval (BM25 index build)
- [x] **Phase 18:** Sparse retrieval query implementation
- [x] **Phase 19:** Reciprocal Rank Fusion (RRF) implementation
- [x] **Phase 20:** Hybrid Retriever Node (`hybrid_search()` single entry point & graceful degradation)
- [x] **Phase 21:** Retrieval Layer Unit & Integration Tests (end-to-end integration, multi-doc scoping, edge cases, 5-QA spot-check eval & latency baseline)

> 🔍 **Retrieval Layer (Phases 16–21) complete — Reranking next (Phase 22–24)**

### Phase 22–24 — Re-ranking
- [x] **Phase 22:** Local cross-encoder re-ranker setup (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- [x] **Phase 23:** Re-ranking pipeline integration & threshold selection (`select_relevant_chunks()` & `RERANK_MIN_SCORE`)
- [x] **Phase 24:** Re-ranking performance benchmarking (`docs/benchmarks.md` & regression guard)

> 🎯 **Re-ranking (Phases 22–24) complete — Generation Layer begins (Phases 25–29)**

### Phase 25–29 — Answer Generation
- [x] **Phase 25:** LLM provider integration & failover client (`generate_answer()` Groq primary + Gemini fallback)
- [x] **Phase 26:** Citation-forced prompt design (`CITATION_SYSTEM_PROMPT` & chunk citation formatting)
- [x] **Phase 27:** Structured JSON output schema (`Claim` + `source_chunk_id` validation)
- [x] **Phase 28:** Formal output parsing & defensive validation (`parse_llm_output()` & `sanitize_json_string()`)
- [x] **Phase 29:** Malformed output handling & corrective retries (2 corrective retries on primary + Gemini fallback + terminal `GenerationError`)

> 🧠 **Generation Layer (Phases 25–29) complete — Claim Processing complete (Phases 30–31)**

### Phase 30–31 — Claim Processing
- [x] **Phase 30:** Sentence / claim splitter (`split_into_sentences()` & `is_atomic_claim()` sentence validation)
- [x] **Phase 31:** Claim-to-chunk mapping logic (`map_claims_to_chunks()` & `ClaimWithSource` in-memory resolution)

> 🔬 **Claim Processing (Phases 30–31) complete — NLI Verification group begins (Phases 32–36)**

### Phase 32–36 — Hallucination Verification
- [x] **Phase 32:** Claim data model design (`ClaimWithSource` with `claim_id` UUID & `VerificationStatus` enum)
- [x] **Phase 33:** Local NLI model integration (`cross-encoder/nli-deberta-v3-base` inference engine & ground-truth label sanity check)
- [x] **Phase 34:** Premise-hypothesis pairing logic (`build_verification_pairs()` & `VerificationPair` with `claim_id` traceability)
- [x] **Phase 35:** Entailment/contradiction/neutral classification (`verify_claims()` batched NLI verification & `NEUTRAL` handling design)
- [x] **Phase 36:** Confidence scoring per claim (`NLI_CONFIDENCE_THRESHOLD = 0.85` & `get_claim_final_status()`)

> 🔬 **NLI Verification (Phases 32–36) complete — LangGraph Correction Loop begins (Phases 37–41)**

### Phase 37–41 — LangGraph Self-Correction Loop
- [x] **Phase 37:** LangGraph state machine design (`RAGState` TypedDict, `StateGraph` topology & stubbed retry flow)
- [x] **Phase 38:** Conditional routing (`correction_router` pass/fail verification edge & `get_failed_claims()` filtering)
- [x] **Phase 39:** Targeted re-retrieval for failed claims (`targeted_retrieve_node()` claim-as-query search, chunk deduplication & "nothing new found" logging)
- [x] **Phase 40:** Partial regeneration logic (`regenerate_failed_claims()` targeted prompt, claim merging & positional preservation)
- [x] **Phase 41:** Retry cap & loop termination logic (`CORRECTION_MAX_RETRIES = 2`, `finalize_node()` synthesis & E2E integration tracing)

> 🔄 **LangGraph Self-Correction Loop (Phases 37–41) complete** — the core self-correcting mechanism is fully functional. Backend API next (Phases 42–45)!

### Phase 42–45 — Backend API, SSE Streaming & Error Handling
- [x] **Phase 42:** FastAPI route design (`POST /api/v1/query` endpoint, consolidated `/api/v1/documents` routes, standardized Pydantic error models)
- [x] **Phase 43:** SSE streaming implementation (`stream_query_execution()` SSE stream generator, real-time `correction_triggered` progress events & stream error handling)
- [x] **Phase 44:** Request/response models (`QueryRequest` string sanitization, `QueryResponse`, `ErrorResponse` & OpenAPI validation)
- [x] **Phase 45:** API error handling & logging (`register_exception_handlers()` global handlers, `CorrelationIdAndTimingMiddleware` tracing & timing)

> 📡 **Backend API Layer (Phases 42–45) complete** — consolidated public API models (`api_models.py`), explicit error→status mapping, request-ID tracing, and per-client rate limiting are in place. Frontend UI & Citation Display next (Phases 46–48)!

### Phase 46 — Frontend & Citation UI

- [x] Responsive research workspace, PDF library, indexing status, and cited answers
- [x] Streamed query progress, source inspection, and a bundled example PDF

### Phase 47–50 — Polish & Production
- [ ] **Phase 47:** Performance optimization & caching
- [ ] **Phase 48:** Comprehensive test suite & CI/CD
- [ ] **Phase 49:** Documentation & API reference
- [ ] **Phase 50:** Production deployment configuration

---

## License

[MIT](LICENSE)
