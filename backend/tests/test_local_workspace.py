from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import settings
from app.ingestion.storage import reset_document_registry
from app.main import app
from app.retrieval.bm25_index import rebuild_bm25_index


def test_local_pdf_workspace_without_cloud_credentials(monkeypatch, tmp_path):
    with monkeypatch.context() as config:
        config.setattr(settings, "RAG_MODE", "local")
        config.setattr(settings, "UPLOAD_DIR", str(tmp_path))
        reset_document_registry()
        sample = Path(__file__).resolve().parents[2] / "frontend/public/project-brief.pdf"
        with TestClient(app) as client:
            assert client.get("/api/v1/query/options").json()["mode"] == "local"
            uploaded = client.post(
                "/api/v1/documents/upload",
                files={"file": ("Project brief.pdf", sample.read_bytes(), "application/pdf")},
            )
            assert uploaded.status_code == 200
            document_id = uploaded.json()["document_id"]
            document_url = f"/api/v1/documents/{document_id}"
            assert client.post(f"{document_url}/process").status_code == 200
            assert client.get(f"{document_url}/status").json()["status"] == "ready"

            query = {"query": "What is the pilot program budget?", "document_id": document_id}
            result = client.post("/api/v1/query", json={**query, "stream": False}).json()
            assert result["final_status"] == "source_passages"
            source = result["claims"][0]
            assert "48,000" in source["claim_text"]
            assert source["claim_text"] in source["source_text"]
            assert source["filename"] == "Project brief.pdf"
            assert source["page_number"] == 1
            assert source["verification_status"] is None

            stream = client.post("/api/v1/query", json=query)
            assert "event: final_answer" in stream.text and "48,000" in stream.text
            empty = client.post("/api/v1/query", json={**query, "query": "telescope", "stream": False})
            assert empty.json()["claims"] == []
            assert client.delete(document_url).status_code == 204
            assert client.get(document_url).status_code == 404
            assert client.post("/api/v1/query", json={**query, "stream": False}).json()["claims"] == []

    reset_document_registry()
    rebuild_bm25_index()
