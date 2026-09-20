# 🔄 Self-Correcting RAG with Hallucination Detection

A production-grade Retrieval-Augmented Generation system that **detects and self-corrects hallucinations** using a multi-stage verification pipeline powered by LangGraph.

## Quickstart (Local Dev)

### Prerequisites

- **Python 3.10+** (developed with Python 3.14.7)
- **Node.js 18+** (developed with Node 24.20.0)
- **npm 9+**
- **Git**

### 1. One-Command Setup & Run (Windows PowerShell)

```powershell
# Clone the repository
git clone <repo-url>
cd self-correcting-rag

# Set up both backend (.venv + deps) and frontend (npm install)
.\dev.ps1 setup

# Configure environment variables
Copy-Item backend\.env.example backend\.env
# Note: In early dev phases, REQUIRE_API_KEYS=false is preset so you can run without API keys

# Start backend dev server (runs at http://localhost:8000)
.\dev.ps1 backend

# In a separate terminal, start frontend dev server (runs at http://localhost:3000)
.\dev.ps1 frontend

# Run backend test suite
.\dev.ps1 test
```

### 2. Manual Step-by-Step Setup

#### Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows cmd:
.venv\Scripts\activate.bat
# macOS/Linux:
source .venv/bin/activate

# Upgrade pip & install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env

# Run tests
pytest tests/ -v

# Start development server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Verify backend: `curl http://127.0.0.1:8000/health` → `{"status":"ok"}`.

#### Frontend Setup

```bash
cd frontend

# Install dependencies cleanly
npm install

# Set up environment variables (optional, defaults to http://localhost:8000)
cp .env.local.example .env.local

# Run lint and production build verification
npm run lint
npm run build

# Start development server
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

> **macOS/Linux users:** A `Makefile` is also provided — run `make setup`, `make backend`, `make frontend`, and `make test`.

---

## Quickstart (Docker)

To run the entire system (FastAPI backend + Next.js frontend) in isolated Docker containers:

### Prerequisites
- Docker Engine & Docker Compose (Docker Desktop on Windows/macOS)

### Launch Services

```bash
# Build and start all services in the background
docker-compose up --build -d

# Check running status
docker-compose ps

# View live container logs
docker-compose logs -f

# Shut down services and clean up networks
docker-compose down
```

Services will be accessible at:
- **Backend API:** [http://localhost:8000](http://localhost:8000) (Docs: [http://localhost:8000/docs](http://localhost:8000/docs))
- **Frontend App:** [http://localhost:3000](http://localhost:3000)

---

## Environment Variables Reference

### Backend Variables (`backend/.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `REQUIRE_API_KEYS` | No | `true` | When `false`, skips fail-fast API key checks (used for local scaffolding / CI) |
| `PINECONE_API_KEY` | **Yes** (Phase 3+) | `None` | Pinecone vector database API key (`pcsk_...`) |
| `PINECONE_INDEX_NAME` | **Yes** (Phase 3+) | `self-correcting-rag` | Target Pinecone vector index name |
| `PINECONE_ENVIRONMENT` | **Yes** (Phase 3+) | `us-east-1` | Pinecone cloud region/environment |
| `ENVIRONMENT` | No | `development` | Deployment environment: `development`, `staging`, `production`, `test` |
| `LOG_LEVEL` | No | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `GEMINI_API_KEY` | Phase 25+ | `None` | Google Gemini API key (for answer generation) |
| `GROQ_API_KEY` | Phase 25+ | `None` | Groq API key for low-latency Llama-3 generation |
| `API_V1_PREFIX` | No | `/api/v1` | URL prefix for REST API version 1 |
| `CORS_ORIGINS` | No | `http://localhost:3000` | Comma-separated list of allowed CORS origins |
| `LANGCHAIN_TRACING_V2` | No | `false` | Set to `true` to enable LangSmith telemetry and state-graph tracing |
| `LANGCHAIN_API_KEY` | Optional | `None` | LangSmith API key (required if tracing is enabled) |
| `LANGCHAIN_PROJECT` | No | `self-correcting-rag` | LangSmith project name to log traces under |
| `LANGCHAIN_ENDPOINT` | No | `https://api.smith.langchain.com` | LangSmith API endpoint |
| `UPLOAD_DIR` | No | `uploads` | Local directory for raw PDF uploads and document registry |
| `MAX_UPLOAD_SIZE_MB` | No | `20` | Maximum allowed file upload size in megabytes |

*\*At least one LLM API key (`GEMINI_API_KEY` or `GROQ_API_KEY`) is required in production when `REQUIRE_API_KEYS=true`.*

### Frontend Variables (`frontend/.env.local`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `NEXT_PUBLIC_API_URL` | No | `http://localhost:8000` | Backend API base URL accessible from browser client |

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
| Frontend      | Next.js 14+ (App Router), TypeScript, Tailwind CSS, shadcn/ui |
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
- [ ] **Phase 14:** Ingestion error handling & retries
- [ ] **Phase 15:** Ingestion status tracking

> 📄 **Document Ingestion core pipeline complete (upload -> extract -> clean -> chunk -> embed -> upsert)** — error handling and status polish next (Phase 14-15)

### Phase 16–21 — Hybrid Retrieval
- [ ] **Phase 16:** Dense retrieval (Pinecone vector search)
- [ ] **Phase 17:** Sparse retrieval (BM25)
- [ ] **Phase 18:** Reciprocal Rank Fusion (RRF) implementation
- [ ] **Phase 19:** Query preprocessing & expansion
- [ ] **Phase 20:** Retrieval configuration & tuning
- [ ] **Phase 21:** Retrieval integration tests

### Phase 22–24 — Re-ranking
- [ ] **Phase 22:** Cross-encoder re-ranker setup
- [ ] **Phase 23:** Re-ranking pipeline integration
- [ ] **Phase 24:** Re-ranking evaluation & tests

### Phase 25–29 — Answer Generation
- [ ] **Phase 25:** LLM client abstraction (Gemini / Groq)
- [ ] **Phase 26:** Prompt template design
- [ ] **Phase 27:** Context-aware answer generation
- [ ] **Phase 28:** Streaming response support
- [ ] **Phase 29:** Generation integration tests

### Phase 30–36 — Hallucination Verification
- [ ] **Phase 30:** NLI model integration
- [ ] **Phase 31:** Claim extraction from generated answers
- [ ] **Phase 32:** Claim-to-source verification
- [ ] **Phase 33:** Hallucination scoring & classification
- [ ] **Phase 34:** Verification result schemas
- [ ] **Phase 35:** Confidence thresholds & configuration
- [ ] **Phase 36:** Verification integration tests

### Phase 37–41 — LangGraph Self-Correction Loop
- [ ] **Phase 37:** LangGraph state definition
- [ ] **Phase 38:** Graph nodes (retrieve → generate → verify)
- [ ] **Phase 39:** Conditional edges & correction routing
- [ ] **Phase 40:** Max retry & fallback logic
- [ ] **Phase 41:** End-to-end graph integration tests

### Phase 42–46 — Frontend
- [ ] **Phase 42:** Chat UI layout & components
- [ ] **Phase 43:** Query submission & streaming display
- [ ] **Phase 44:** Source citation display
- [ ] **Phase 45:** Hallucination confidence indicators
- [ ] **Phase 46:** Document upload interface

### Phase 47–50 — Polish & Production
- [ ] **Phase 47:** Performance optimization & caching
- [ ] **Phase 48:** Comprehensive test suite & CI/CD
- [ ] **Phase 49:** Documentation & API reference
- [ ] **Phase 50:** Production deployment configuration

---

## License

[MIT](LICENSE)
