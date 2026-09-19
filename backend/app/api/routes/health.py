"""Health-check endpoint."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Return a simple health status.

    Returns:
        ``{"status": "ok"}`` with HTTP 200.
    """
    return {"status": "ok"}
