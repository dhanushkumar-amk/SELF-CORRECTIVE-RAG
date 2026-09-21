"""
Application configuration via pydantic-settings.

Single centralized source of truth for all configuration and secrets across the project.
Validates required configuration on startup and fails fast with clear, actionable error messages.

Production Deployment Note (Phase 50):
    In local development, settings are read from the `backend/.env` file.
    In production environments (Docker, Kubernetes, AWS/GCP serverless), secrets
    should NOT be stored in a .env file. Instead, inject them as native environment
    variables or retrieve them via a cloud secrets manager (e.g., AWS Secrets Manager,
    GCP Secret Manager, or HashiCorp Vault). Pydantic-settings natively prioritizes
    system environment variables over .env files.
"""

from typing import Literal
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings

from app.core.logging import get_logger

logger = get_logger(__name__)

# Known dummy placeholder strings from .env.example or tutorials
PLACEHOLDER_SUBSTRINGS = (
    "your-",
    "your_",
    "placeholder",
    "change-me",
    "changeme",
    "replace-me",
    "todo",
    "xxx",
)


def _is_placeholder(value: str) -> bool:
    """Return True if the string looks like an unfilled template placeholder."""
    v = value.strip().lower()
    if not v:
        return False
    return any(p in v for p in PLACEHOLDER_SUBSTRINGS)


class Settings(BaseSettings):
    """Global application settings and secrets configuration.

    All settings across the backend must be accessed through this class.
    Do NOT use raw `os.getenv()` in application code.
    """

    # ── Project Metadata ──────────────────────────────────────────────────
    PROJECT_NAME: str = "Self-Correcting RAG"
    VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"

    # ── Runtime Environment ───────────────────────────────────────────────
    ENVIRONMENT: Literal["development", "staging", "production", "test"] = (
        "development"
    )
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ── CORS ─────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # ── Document Storage & Ingestion (Phase 6+) ───────────────────────────
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE_MB: int = 20

    # ── Vector DB: Pinecone (Required for Phase 3+) ────────────────────────
    # Sign up at https://app.pinecone.io (Free Starter tier)
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "self-correcting-rag"
    PINECONE_ENVIRONMENT: str = "us-east-1"
    PINECONE_TIMEOUT_SECONDS: int = 30
    # ── Reranking (Phase 22–24) ───────────────────────────────────────────
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_MIN_SCORE: float = -2.0  # Empirical logit threshold for relevance filtering
    RERANK_TOP_N: int = 5

    # ── LLM Providers (Phase 25 — Generation) ─────────────────────────────
    # Google Gemini: https://ai.google.dev/
    GEMINI_API_KEY: str = ""
    DEFAULT_GEMINI_MODEL: str = "gemini-2.5-flash"
    # Groq (Llama / OSS): https://console.groq.com/
    GROQ_API_KEY: str = ""
    DEFAULT_GROQ_MODEL: str = "openai/gpt-oss-120b"
    LLM_PRIMARY_PROVIDER: str = "groq"
    LLM_FALLBACK_PROVIDER: str = "gemini"

    # ── Verification & NLI (Phase 33+) ────────────────────────────────────
    NLI_MODEL_NAME: str = "cross-encoder/nli-deberta-v3-base"
    NLI_CONFIDENCE_THRESHOLD: float = 0.85

    # ── Correction Loop (Phase 40–41) ─────────────────────────────────────
    CORRECTION_MAX_RETRIES: int = 2

    # ── Observability: LangChain / LangSmith (Optional — Debugging) ───────
    # Observability platform for LangGraph traces: https://smith.langchain.com/
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "self-correcting-rag"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"

    # ── Safety / CI Flags ─────────────────────────────────────────────────
    # Set to False during local scaffolding or CI pipelines without external keys
    REQUIRE_API_KEYS: bool = True

    model_config = {
        "env_file": (".env", "backend/.env"),
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }

    # ── Field Validators ──────────────────────────────────────────────────

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def _normalize_log_level(cls, v: str) -> str:
        return v.upper().strip() if isinstance(v, str) else v

    # ── Model Validators (Fail-Fast Verification) ─────────────────────────

    @model_validator(mode="after")
    def _validate_startup_configuration(self) -> "Settings":
        """Fail fast on startup with clear, actionable guidance if configuration is invalid.

        Ensures errors are caught immediately during boot instead of failing silently
        or crashing with cryptic tracebacks deep inside worker threads.
        """
        if not self.REQUIRE_API_KEYS:
            return self

        errors: list[str] = []

        # 1. Validate Pinecone API Key
        if not self.PINECONE_API_KEY:
            errors.append(
                "* PINECONE_API_KEY is missing.\n"
                "  -> Action: Get a free key at https://app.pinecone.io/organizations/-/keys\n"
                "  -> Add to backend/.env: PINECONE_API_KEY=pcsk_..."
            )
        elif _is_placeholder(self.PINECONE_API_KEY):
            errors.append(
                f"* PINECONE_API_KEY is set to an example placeholder ('{self.PINECONE_API_KEY}').\n"
                "  -> Action: Replace it with your actual Pinecone API key in backend/.env\n"
                "  -> Get one free at: https://app.pinecone.io/organizations/-/keys"
            )

        # 2. Validate Pinecone Index Name
        if not self.PINECONE_INDEX_NAME:
            errors.append(
                "* PINECONE_INDEX_NAME is missing.\n"
                "  -> Action: Create a 384-dim cosine index named 'self-correcting-rag' in the Pinecone console\n"
                "  -> Add to backend/.env: PINECONE_INDEX_NAME=self-correcting-rag"
            )

        # 3. Validate Pinecone Environment/Region
        if not self.PINECONE_ENVIRONMENT:
            errors.append(
                "* PINECONE_ENVIRONMENT is missing.\n"
                "  -> Action: Set it to your cloud region (e.g. 'us-east-1') in backend/.env"
            )

        # 4. In production, check for at least one LLM key (reserved for Phase 25+)
        if self.ENVIRONMENT == "production":
            if not self.GEMINI_API_KEY and not self.GROQ_API_KEY:
                errors.append(
                    "* In production, at least one LLM key (GEMINI_API_KEY or GROQ_API_KEY) must be set.\n"
                    "  -> Gemini (free): https://ai.google.dev/\n"
                    "  -> Groq (free):   https://console.groq.com/"
                )

        if errors:
            border = "-" * 70
            msg = (
                f"\n+{border}+\n"
                f"|  CONFIGURATION & SECRET VALIDATION FAILED                           |\n"
                f"+{border}+\n"
            )
            for err in errors:
                for line in err.split("\n"):
                    msg += f"|  {line:<67}|\n"
            msg += (
                f"+{border}+\n"
                f"|  Fix: Copy backend/.env.example -> backend/.env and populate keys.   |\n"
                f"|  Or for offline dev/CI: set REQUIRE_API_KEYS=false in backend/.env. |\n"
                f"+{border}+\n"
            )
            logger.error(msg)
            raise ValueError(
                f"Configuration startup validation failed:\n" + "\n\n".join(errors)
            )

        return self

    @property
    def max_upload_size_bytes(self) -> int:
        """Return maximum allowed upload size in bytes."""
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def langsmith_enabled(self) -> bool:
        """Check if LangSmith tracing is configured and enabled."""
        return bool(self.LANGCHAIN_TRACING_V2 and self.LANGCHAIN_API_KEY)


def get_settings(**overrides: object) -> Settings:
    """Create a Settings instance with optional overrides.

    Useful in tests to verify validation rules and override specific keys
    without touching the filesystem .env.
    """
    return Settings(**overrides)


# ── Module-level singleton ────────────────────────────────────────────────────
# The primary configuration object imported across the application.
settings = get_settings()
