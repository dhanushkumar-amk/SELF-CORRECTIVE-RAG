"""
Application configuration via pydantic-settings.

Reads values from environment variables and .env file.
Validates required configuration on startup and fails fast with clear errors.
"""

import sys

from pydantic import model_validator
from pydantic_settings import BaseSettings

from app.core.logging import get_logger

logger = get_logger(__name__)


class Settings(BaseSettings):
    """Global application settings.

    Required API keys are validated on instantiation. The application
    will refuse to start if critical configuration is missing, rather
    than failing later with a cryptic error deep in a retrieval call.
    """

    # ── Project ──────────────────────────────────────────────────────────
    PROJECT_NAME: str = "Self-Correcting RAG"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"

    # ── CORS ─────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # ── API Keys (populated via .env) ────────────────────────────────────
    PINECONE_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    # ── Pinecone ─────────────────────────────────────────────────────────
    PINECONE_ENVIRONMENT: str = "us-east-1"
    PINECONE_INDEX_NAME: str = "self-correcting-rag"

    # ── LangChain / LangSmith ────────────────────────────────────────────
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_TRACING_V2: bool = False

    # ── Feature flags ────────────────────────────────────────────────────
    REQUIRE_API_KEYS: bool = True  # set False in tests / CI

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }

    # ── Validation ───────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _validate_required_keys(self) -> "Settings":
        """Fail fast if required API keys are missing.

        This prevents the app from starting and then crashing deep inside
        a Pinecone or LLM call with an unhelpful error.

        Set REQUIRE_API_KEYS=false in .env to skip this check during
        development/testing when you don't need external services.
        """
        if not self.REQUIRE_API_KEYS:
            return self

        missing: list[str] = []

        if not self.PINECONE_API_KEY:
            missing.append("PINECONE_API_KEY")
        if not self.PINECONE_INDEX_NAME:
            missing.append("PINECONE_INDEX_NAME")
        if not self.GEMINI_API_KEY and not self.GROQ_API_KEY:
            missing.append("GEMINI_API_KEY or GROQ_API_KEY (at least one)")

        if missing:
            msg = (
                "\n╔══════════════════════════════════════════════════════╗\n"
                "║  MISSING REQUIRED ENVIRONMENT VARIABLES             ║\n"
                "╠══════════════════════════════════════════════════════╣\n"
            )
            for var in missing:
                msg += f"║  • {var:<50}║\n"
            msg += (
                "╠══════════════════════════════════════════════════════╣\n"
                "║  Copy .env.example → .env and fill in your keys.   ║\n"
                "║  Or set REQUIRE_API_KEYS=false to skip validation.  ║\n"
                "╚══════════════════════════════════════════════════════╝"
            )
            logger.error(msg)
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "See .env.example for required values."
            )

        return self

    @property
    def langsmith_enabled(self) -> bool:
        """Check if LangSmith tracing is configured and enabled."""
        return bool(self.LANGCHAIN_TRACING_V2 and self.LANGCHAIN_API_KEY)


def get_settings(**overrides: object) -> Settings:
    """Create a Settings instance with optional overrides.

    Useful in tests to override specific settings without touching .env.
    """
    return Settings(**overrides)


# ── Module-level singleton ────────────────────────────────────────────────
# REQUIRE_API_KEYS defaults to True but is set to False via .env
# during development when keys aren't needed yet.
settings = get_settings()
