"""
Pytest configuration for the backend test suite.

The suite makes many /api/v1/query requests across test modules. Since the
Phase 45 rate limiter is an in-process global, the default 10/minute cap would
cause spurious 429s. Every test therefore runs with the limit relaxed and the
limiter state reset; the dedicated rate-limiting test re-enables a strict limit
explicitly.
"""

import pytest

from app.core.config import settings
from app.core.rate_limit import reset_rate_limiters


@pytest.fixture(autouse=True)
def _relax_rate_limits_for_suite(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise the global query rate limit so unrelated tests never trip 429s."""
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_QUERY_PER_MINUTE", 1000)
    reset_rate_limiters()
    yield
