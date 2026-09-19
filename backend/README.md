# Self-Correcting RAG — Backend

FastAPI backend for the Self-Correcting RAG with Hallucination Detection system.

## Quick Start

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys

# 5. Run the development server
uvicorn app.main:app --reload
```

The server starts at **http://localhost:8000**.

## API Endpoints

| Method | Path       | Description           | Status      |
| ------ | ---------- | --------------------- | ----------- |
| GET    | `/health`  | Health check          | ✅ Active   |
| *      | `/ingest`  | Document ingestion    | 🔲 Phase 6  |
| *      | `/query`   | RAG query             | 🔲 Phase 16 |

## Testing

```bash
pytest
```

## Project Structure

```
app/
├── main.py            # FastAPI entrypoint
├── core/              # Config, logging
├── api/routes/        # HTTP endpoints
├── ingestion/         # Document processing pipeline
├── retrieval/         # Vector search
├── reranking/         # Result re-ranking
├── generation/        # LLM answer generation
├── verification/      # Hallucination detection
├── graph/             # LangGraph orchestration
└── models/            # Pydantic schemas
```
