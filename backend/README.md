# Self-Correcting RAG — Backend

FastAPI backend for the Self-Correcting RAG with Hallucination Detection system.

## Prerequisites

- **Python 3.10+** (developed with Python 3.14.7)
- `pip` (included with Python)

> **pyenv users:** A `.python-version` file is included in this directory.

## Quick Start

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys (or leave REQUIRE_API_KEYS=false for early dev)

# 6. Run the development server
uvicorn app.main:app --reload
```

The server starts at **http://localhost:8000**.

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Required | Description |
| --- | --- | --- |
| `PINECONE_API_KEY` | Yes* | Pinecone vector DB API key |
| `GEMINI_API_KEY` | Yes* (or Groq) | Google Gemini LLM key |
| `GROQ_API_KEY` | Yes* (or Gemini) | Groq inference key |
| `PINECONE_ENVIRONMENT` | Yes* | Pinecone environment region |
| `PINECONE_INDEX_NAME` | Yes* | Pinecone index name |
| `LANGCHAIN_API_KEY` | No | LangSmith API key (for tracing) |
| `LANGCHAIN_TRACING_V2` | No | Set `true` to enable LangSmith traces |
| `REQUIRE_API_KEYS` | No | Set `false` to skip key validation (default: `true`) |

> *Required when `REQUIRE_API_KEYS=true` (production). Set `REQUIRE_API_KEYS=false` during early development phases.

### Fail-Fast Validation

The app validates required environment variables on startup. If critical keys are
missing, it will **refuse to start** with a clear error message rather than failing
later with a cryptic error deep in a retrieval or LLM call.

### LangSmith Tracing

LangSmith provides observability into the LangGraph self-correction loop:
step-by-step traces, latency, and token usage. To enable:

1. Sign up at [smith.langchain.com](https://smith.langchain.com/)
2. Set `LANGCHAIN_API_KEY=your-key` and `LANGCHAIN_TRACING_V2=true` in `.env`

Tracing is optional — the app works identically without it.

## Pinecone Setup & Configuration

This project uses [Pinecone](https://www.pinecone.io/) for high-performance serverless vector search.

### 1. Sign Up & Retrieve API Key
1. Sign up for a free account at [app.pinecone.io](https://app.pinecone.io/) (Free Starter tier — no credit card required).
2. Go to **API Keys** in the left sidebar and copy your default API key (starts with `pcsk_...`).
3. Add it to `backend/.env`:
   ```bash
   PINECONE_API_KEY=pcsk_your_actual_key_here
   PINECONE_ENVIRONMENT=us-east-1
   PINECONE_INDEX_NAME=self-correcting-rag
   ```

### 2. Create the Vector Index
Create an index with the following exact specifications in the Pinecone Console:

| Setting | Value | Rationale |
|---|---|---|
| **Index Name** | `self-correcting-rag` | Standardized project index identifier |
| **Dimensions** | `384` | Matches `sentence-transformers/all-MiniLM-L6-v2` |
| **Metric** | `cosine` | Optimal metric for normalized dense text embeddings |
| **Cloud Provider** | `AWS` | Standard serverless cloud provider |
| **Region** | `us-east-1` | Free Starter tier default region |

#### Embedding Model Decision: `all-MiniLM-L6-v2` (384d) vs `all-mpnet-base-v2` (768d)
- **Decision:** Locked in **`all-MiniLM-L6-v2` (384 dimensions)**.
- **Trade-off Analysis:**
  - `all-MiniLM-L6-v2`: Model size ~80MB, ~5x faster inference, runs seamlessly on local CPUs without GPU requirements. Pinecone vector operations are twice as memory- and latency-efficient at 384 dimensions.
  - `all-mpnet-base-v2`: Model size ~420MB, 768 dimensions. While offering a ~3–4% higher baseline MTEB score, it introduces notable latency on local execution.
  - **Why MiniLM is optimal for this architecture:** In our pipeline, dense retrieval is followed by **Cross-Encoder re-ranking (Phase 22)** and **NLI hallucination verification (Phase 30)**. High retrieval recall (top-k=15–25) from MiniLM is more than sufficient; downstream re-ranking provides high precision without sacrificing system throughput.

### 3. Verify Pinecone Connectivity
Run the test suite to execute the Pinecone smoke test:
```bash
pytest tests/test_pinecone_connection.py -v
```
*(Tests skip gracefully if `PINECONE_API_KEY` is not yet configured or is a placeholder).*

You can also check index stats via the temporary debug endpoint:
```bash
curl http://localhost:8000/debug/pinecone-stats
```

## API Endpoints

| Method | Path | Description | Status |
| --- | --- | --- | --- |
| GET | `/health` | Health check | ✅ Active |
| GET | `/debug/pinecone-stats` | Pinecone index statistics (vector count, dimensions) | ✅ Phase 3 Debug |
| * | `/ingest` | Document ingestion | 🔲 Phase 6 |
| * | `/query` | RAG query | 🔲 Phase 16 |

## Testing

```bash
# Activate venv first, then:
pytest tests/ -v
```

## Dependencies

### Installed (requirements.txt)

Core: FastAPI, uvicorn, pydantic, pydantic-settings  
LLM: langchain, langchain-core, langchain-community, langgraph  
Retrieval: pinecone-client, rank-bm25  
Document: pypdf  
Testing: pytest, pytest-asyncio, httpx

### Deferred (install in later phases)

These packages have heavy native dependencies (PyTorch, numpy build issues on Python 3.14):

- `sentence-transformers` — Phase 10+ (embedding generation)
- `langchain-pinecone` — Phase 16+ (Pinecone LangChain integration)

## Project Structure

```
app/
├── main.py            # FastAPI entrypoint with lifespan handler
├── core/              # Config (fail-fast validation), logging
├── api/routes/        # HTTP endpoints
├── ingestion/         # Document processing pipeline (Phase 6-15)
├── retrieval/         # Vector search (Phase 16-21)
├── reranking/         # Result re-ranking (Phase 22-24)
├── generation/        # LLM answer generation (Phase 25-29)
├── verification/      # Hallucination detection (Phase 30-36)
├── graph/             # LangGraph orchestration (Phase 37-41)
└── models/            # Pydantic schemas
```
