"""Health-check endpoint."""

from fastapi import APIRouter

from app.models.api_models import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness & readiness health check",
)
async def health_check() -> HealthResponse:
    """Return a simple health status.

    Returns:
        ``{"status": "ok"}`` with HTTP 200.
    """
    return HealthResponse(status="ok")
