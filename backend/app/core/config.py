"""
Application configuration via pydantic-settings.

Reads values from environment variables and .env file.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global application settings."""

    # ── Project ──────────────────────────────────────────────────────────
    PROJECT_NAME: str = "Self-Correcting RAG"
    VERSION: str = "0.1.0"

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
    PINECONE_ENVIRONMENT: str = ""
    PINECONE_INDEX_NAME: str = ""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


settings = Settings()
