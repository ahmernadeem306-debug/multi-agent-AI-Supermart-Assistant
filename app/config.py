"""Application configuration loaded from environment variables / .env."""
from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.exceptions import ConfigError


class Settings(BaseSettings):
    groq_api_key: SecretStr
    groq_model: str = "llama-3.1-8b-instant"
    groq_timeout_seconds: int = 60

    database_url: str = "sqlite:///./data/bizagent.db"

    log_level: str = "INFO"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"

    chroma_persist_dir: str = "./data/chroma"

    tool_provider: str = "mcp"
    tool_provider_fallback: bool = False

    # --- RAG knowledge engine ---
    embedding_backend: str = "local"  # "groq" (unsupported, see app/rag/embeddings.py) | "local"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    knowledge_dir: str = "./data/knowledge"
    rag_collection_prefix: str = "bizagent_kb"
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 120
    rag_top_k: int = 5
    rag_min_score: float = 0.15

    # --- multi-agent orchestration ---
    agent_max_tool_calls: int = 3
    graph_recursion_limit: int = 25
    llm_temperature_routing: float = 0.0
    llm_temperature_synthesis: float = 0.35

    # --- forecasting & risk ---
    forecast_backend: str = "xgboost"  # "xgboost" | "seasonal_naive"
    forecast_horizon_days: int = 14
    forecast_max_horizon_days: int = 30
    forecast_min_history_days: int = 60
    model_dir: str = "./models"
    reorder_z_score: float = 1.65  # ~95% service level for safety stock
    stockout_critical_days: int = 3
    stockout_high_days: int = 7
    stockout_medium_days: int = 14

    # --- uploads ---
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB

    app_env: str = "development"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached Settings instance, loading it on first call.

    Fails fast with a readable ConfigError if required configuration
    (namely GROQ_API_KEY) is missing, instead of surfacing a raw
    pydantic ValidationError deep inside an agent run.
    """
    global _settings
    if _settings is not None:
        return _settings
    try:
        _settings = Settings()
    except Exception as exc:  # pydantic ValidationError on missing/invalid env
        raise ConfigError(
            "Failed to load application configuration. Make sure GROQ_API_KEY "
            "is set in your environment or in a .env file at the project root "
            "(see .env.example)."
        ) from exc
    return _settings
