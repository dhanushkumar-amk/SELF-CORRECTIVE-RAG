"""Tests for debug endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_debug_pinecone_stats_mocked() -> None:
    """GET /debug/pinecone-stats should return JSON statistics from PineconeClient."""
    mock_stats = {
        "dimension": 384,
        "total_vector_count": 0,
        "namespaces": {},
        "index_fullness": 0.0,
    }

    with patch("app.api.routes.debug.get_pinecone_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.get_index_stats.return_value = mock_stats
        mock_get_client.return_value = mock_client

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/debug/pinecone-stats")

        assert response.status_code == 200
        data = response.json()
        assert data["dimension"] == 384
        assert data["total_vector_count"] == 0


@pytest.mark.asyncio
async def test_debug_pinecone_stats_error_handling() -> None:
    """GET /debug/pinecone-stats should return HTTP 500 when client raises exception."""
    with patch("app.api.routes.debug.get_pinecone_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.get_index_stats.side_effect = RuntimeError("Connection error")
        mock_get_client.return_value = mock_client

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/debug/pinecone-stats")

        assert response.status_code == 500
        assert "Connection error" in response.json()["detail"]
