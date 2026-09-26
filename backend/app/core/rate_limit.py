"""
In-Process Rate Limiting for Phase 45: Abuse Protection on /api/v1/query.

Architecture & Design Decisions:
1. Sliding-Window Algorithm:
   A fixed 60-second sliding window keyed by client IP address; requests beyond
   ``RATE_LIMIT_QUERY_PER_MINUTE`` within the window are rejected with HTTP 429.

2. Why a custom in-process limiter (instead of slowapi):
   The project targets bleeding-edge Python 3.14 / Starlette 1.x where slowapi's
   pinned dependency chain does not yet declare compatibility. This module is a
   ~60-line dependency-free implementation that keeps the standard ErrorResponse
   contract. For multi-replica production deployments, swap the storage backend
   for a shared store (e.g. Redis) via slowapi/limits — the interface stays the
   same (`check(key) -> (allowed, retry_after)`).

3. Limit Rationale (default: 10 requests/minute/client):
   Each /api/v1/query call can trigger multiple billed LLM invocations
   (initial generation + up to CORRECTION_MAX_RETRIES=2 corrective re-generations
   + targeted re-retrieval), so a per-minute cap of 10 protects Groq/Gemini
   free-tier quotas from runaway clients while staying generous for interactive use.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from fastapi import Request

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "RateLimitExceeded",
    "SlidingWindowRateLimiter",
    "rate_limit_query",
    "reset_rate_limiters",
]


class RateLimitExceeded(Exception):
    """Raised when a client exceeds the configured per-minute query rate limit."""

    def __init__(self, limit: int, retry_after: float, client_key: str) -> None:
        super().__init__(
            f"Rate limit exceeded: maximum {limit} queries per minute per client."
        )
        self.limit = limit
        self.retry_after = retry_after
        self.client_key = client_key


class SlidingWindowRateLimiter:
    """Thread-safe fixed-window-sliding rate limiter backed by an in-memory deque.

    Args:
        limit: Maximum number of allowed hits within the window.
        window_seconds: Width of the sliding window in seconds (default 60).
    """

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, float]:
        """Record a hit for ``key`` and report whether it is allowed.

        Returns:
            Tuple of (allowed: bool, retry_after_seconds: float). When the hit
            is rejected, ``retry_after_seconds`` is the time until the oldest
            recorded hit expires from the window.
        """
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] > self.window_seconds:
                hits.popleft()

            if len(hits) >= self.limit:
                retry_after = self.window_seconds - (now - hits[0])
                return False, max(0.0, retry_after)

            hits.append(now)
            return True, 0.0

    def reset(self) -> None:
        """Clear all recorded hits (used by tests and operational tooling)."""
        with self._lock:
            self._hits.clear()


# Limiters are cached per configured limit so a settings change mid-process
# starts a fresh window instead of corrupting an existing one.
_LIMITERS: dict[int, SlidingWindowRateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def _get_limiter(limit: int) -> SlidingWindowRateLimiter:
    with _LIMITERS_LOCK:
        if limit not in _LIMITERS:
            _LIMITERS[limit] = SlidingWindowRateLimiter(limit=limit)
        return _LIMITERS[limit]


def reset_rate_limiters() -> None:
    """Drop all cached limiter instances (used by tests)."""
    with _LIMITERS_LOCK:
        _LIMITERS.clear()


def _client_key(request: Request) -> str:
    """Resolve a stable per-client key from X-Forwarded-For or the socket peer IP."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


async def rate_limit_query(request: Request) -> None:
    """FastAPI dependency enforcing the per-minute query rate limit.

    Raises:
        RateLimitExceeded: When the client has exhausted its per-minute quota.
    """
    if not settings.RATE_LIMIT_ENABLED:
        return

    limit = settings.RATE_LIMIT_QUERY_PER_MINUTE
    key = _client_key(request)
    limiter = _get_limiter(limit)

    allowed, retry_after = limiter.check(key)
    if not allowed:
        logger.warning(
            "Rate limit exceeded for client '%s' on %s (limit=%d/min).",
            key,
            request.url.path,
            limit,
        )
        raise RateLimitExceeded(limit=limit, retry_after=retry_after, client_key=key)
