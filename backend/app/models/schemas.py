"""
Pydantic schemas for request/response models.

This file will be populated in later phases as endpoints are implemented.
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response schema for the health check endpoint."""

    status: str
