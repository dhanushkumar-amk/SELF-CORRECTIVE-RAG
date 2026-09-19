# 🔄 Self-Correcting RAG with Hallucination Detection

A production-grade Retrieval-Augmented Generation system that **detects and self-corrects hallucinations** using a multi-stage verification pipeline powered by LangGraph.

## Quickstart

```powershell
# Clone and enter the project
git clone <repo-url>
cd self-correcting-rag

# One-command setup (installs backend + frontend dependencies)
.\dev.ps1 setup

# Start both services (in separate terminals)
.\dev.ps1 backend     # → http://localhost:8000
.\dev.ps1 frontend    # → http://localhost:3000

# Run tests
.\dev.ps1 test
```

> **macOS/Linux users:** A `Makefile` is also provided — use `make setup`, `make backend`, etc.

See [backend/README.md](backend/README.md) and [frontend/README.md](frontend/README.md) for detailed setup instructions.

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

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- npm
- Git

### Backend

```bash
cd backend

# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Start the development server
uvicorn app.main:app --reload
```

Server runs at **http://localhost:8000** — verify with `curl http://localhost:8000/health`.

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

App runs at **http://localhost:3000**.

### Docker (both services)

```bash
docker-compose up --build
```

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
- [ ] **Phase 3:** Database schema & Pinecone index setup
- [ ] **Phase 4:** Authentication & API key management
- [ ] **Phase 5:** Logging, monitoring & error handling framework

### Phase 6–15 — Document Ingestion Pipeline
- [ ] **Phase 6:** PDF upload endpoint
- [ ] **Phase 7:** PDF text extraction (pypdf)
- [ ] **Phase 8:** Text chunking strategy (recursive character splitting)
- [ ] **Phase 9:** Chunk metadata enrichment
- [ ] **Phase 10:** Embedding generation (sentence-transformers)
- [ ] **Phase 11:** Pinecone upsert pipeline
- [ ] **Phase 12:** Batch ingestion with progress tracking
- [ ] **Phase 13:** Duplicate detection & deduplication
- [ ] **Phase 14:** Ingestion error handling & retry logic
- [ ] **Phase 15:** Ingestion integration tests

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
