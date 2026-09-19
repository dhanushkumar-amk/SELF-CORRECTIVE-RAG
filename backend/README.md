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

## API Endpoints

| Method | Path | Description | Status |
| --- | --- | --- | --- |
| GET | `/health` | Health check | ✅ Active |
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
