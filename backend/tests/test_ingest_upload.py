"""Tests for the PDF document ingestion upload and registry endpoints."""

from pathlib import Path
from unittest.mock import patch
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.ingestion.storage import reset_document_registry
from app.main import app

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF_PATH = FIXTURES_DIR / "sample.pdf"


@pytest.fixture(autouse=True)
def isolated_upload_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate uploads and registry file to a temporary directory for each test."""
    upload_dir = tmp_path / "test_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "UPLOAD_DIR", str(upload_dir))
    reset_document_registry()
    yield upload_dir
    reset_document_registry()


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Load or generate valid sample PDF fixture bytes."""
    if SAMPLE_PDF_PATH.exists():
        return SAMPLE_PDF_PATH.read_bytes()
    # Fallback minimal valid PDF
    return (
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 300 144]>>endobj\nxref\n"
        b"0 4\n0000000000 65535 f \ntrailer<</Size 4/Root 1 0 R>>\n"
        b"startxref\n120\n%%EOF\n"
    )


@pytest.mark.asyncio
async def test_upload_valid_pdf_success(sample_pdf_bytes: bytes, isolated_upload_dir: Path) -> None:
    """Uploading a valid PDF should succeed and return document metadata."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/ingest/upload",
            files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    # Verify document_id is a valid UUID
    doc_uuid = uuid.UUID(data["document_id"])
    assert str(doc_uuid) == data["document_id"]
    assert data["filename"] == "sample.pdf"
    assert data["status"] == "uploaded"
    assert data["size_bytes"] == len(sample_pdf_bytes)

    # Verify physical file storage on disk
    stored_file = isolated_upload_dir / data["document_id"] / "sample.pdf"
    assert stored_file.exists()
    assert stored_file.read_bytes() == sample_pdf_bytes

    # Verify registry.json entry
    registry_file = isolated_upload_dir / "registry.json"
    assert registry_file.exists()
    assert data["document_id"] in registry_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_upload_non_pdf_rejected() -> None:
    """Uploading a non-PDF file (e.g. text file disguised as .pdf) should return HTTP 400."""
    fake_pdf_content = b"This is plain text pretending to be a PDF file."
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/ingest/upload",
            files={"file": ("fake.pdf", fake_pdf_content, "application/pdf")},
        )

    assert response.status_code == 400
    data = response.json()
    assert "not a valid PDF" in data["detail"]
    assert "%PDF-" in data["detail"]


@pytest.mark.asyncio
async def test_upload_empty_file_rejected() -> None:
    """Uploading an empty 0-byte file should return HTTP 400."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/ingest/upload",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )

    assert response.status_code == 400
    data = response.json()
    assert "empty (0 bytes)" in data["detail"]


@pytest.mark.asyncio
async def test_upload_oversized_file_rejected(
    monkeypatch: pytest.MonkeyPatch, sample_pdf_bytes: bytes
) -> None:
    """Uploading a file exceeding MAX_UPLOAD_SIZE_MB should return HTTP 400."""
    # Set limit to 1MB and construct an oversized payload starting with %PDF-
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 1)

    # 1MB + 1024 bytes
    oversized_content = b"%PDF-1.4" + (b"0" * (1024 * 1024 + 1024))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/ingest/upload",
            files={"file": ("large.pdf", oversized_content, "application/pdf")},
        )

    assert response.status_code == 400
    data = response.json()
    assert "exceeds maximum allowed limit" in data["detail"]
    assert "1MB" in data["detail"]


@pytest.mark.asyncio
async def test_get_documents_list(sample_pdf_bytes: bytes) -> None:
    """GET /api/ingest/documents should return all registered documents."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Initially empty
        initial_res = await client.get("/api/ingest/documents")
        assert initial_res.status_code == 200
        assert initial_res.json() == {"documents": [], "total": 0}

        # Upload two documents
        up1 = await client.post(
            "/api/ingest/upload",
            files={"file": ("doc1.pdf", sample_pdf_bytes, "application/pdf")},
        )
        assert up1.status_code == 200
        doc1_id = up1.json()["document_id"]

        up2 = await client.post(
            "/api/ingest/upload",
            files={"file": ("doc2.pdf", sample_pdf_bytes, "application/pdf")},
        )
        assert up2.status_code == 200
        doc2_id = up2.json()["document_id"]

        # List documents
        list_res = await client.get("/api/ingest/documents")
        assert list_res.status_code == 200
        list_data = list_res.json()
        assert list_data["total"] == 2
        returned_ids = [d["document_id"] for d in list_data["documents"]]
        assert doc1_id in returned_ids
        assert doc2_id in returned_ids


@pytest.mark.asyncio
async def test_get_document_by_id(sample_pdf_bytes: bytes) -> None:
    """GET /api/ingest/documents/{id} should return document metadata."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Upload a document
        upload_res = await client.post(
            "/api/ingest/upload",
            files={"file": ("test_doc.pdf", sample_pdf_bytes, "application/pdf")},
        )
        assert upload_res.status_code == 200
        doc_id = upload_res.json()["document_id"]

        # Query by document_id
        get_res = await client.get(f"/api/ingest/documents/{doc_id}")
        assert get_res.status_code == 200
        doc_data = get_res.json()
        assert doc_data["document_id"] == doc_id
        assert doc_data["filename"] == "test_doc.pdf"
        assert doc_data["status"] == "uploaded"
        assert doc_data["size_bytes"] == len(sample_pdf_bytes)


@pytest.mark.asyncio
async def test_get_document_by_id_not_found() -> None:
    """GET /api/ingest/documents/{id} with nonexistent ID should return HTTP 404."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/ingest/documents/nonexistent-id-999")

    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"]


@pytest.mark.asyncio
async def test_upload_disk_error_handling(sample_pdf_bytes: bytes) -> None:
    """Simulate disk save failure: endpoint should catch error and return HTTP 500."""
    with patch("app.api.routes.ingest.get_document_registry") as mock_get_reg:
        mock_registry = mock_get_reg.return_value
        mock_registry.save_document.side_effect = IOError("Disk full or permission denied")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/ingest/upload",
                files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")},
            )

        assert response.status_code == 500
        data = response.json()
        assert "Failed to store uploaded document to disk storage." in data["detail"]
