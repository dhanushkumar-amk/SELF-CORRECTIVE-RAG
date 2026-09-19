"""Debug endpoints for development and verification.

TODO: Remove or gate behind an authenticated admin/DEBUG flag before production release.
"""

from fastapi import APIRouter, HTTPException

from app.core.logging import get_logger
from app.retrieval.pinecone_client import get_pinecone_client

logger = get_logger(__name__)
router = APIRouter()


@router.get("/pinecone-stats")
async def get_pinecone_stats() -> dict:
    """Retrieve current Pinecone index statistics (vector count, dimension, namespaces).

    Temporary endpoint for Phase 3 verification and debugging.
    """
    try:
        client = get_pinecone_client()
        stats = client.get_index_stats()
        return stats
    except Exception as exc:
        logger.error("Failed to fetch Pinecone index stats: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch Pinecone index stats: {exc}",
        ) from exc
