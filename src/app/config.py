from pydantic import AnyUrl, AnyHttpUrl, field_validator, PostgresDsn, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List, Union
import secrets
import logging
import logging.config


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    PROJECT_NAME: str = "SentimentScope API"
    APP_NAME: str = PROJECT_NAME
    ENVIRONMENT: str = "development"

    DB_USER: Optional[str] = "postgres"
    DB_PASSWORD: Optional[str] = "postgres"
    DB_HOST: Optional[str] = "localhost"
    DB_PORT: Optional[int] = 5432
    DB_NAME: Optional[str] = "sentiment_db"
    DATABASE_URL: Optional[Union[PostgresDsn, str]] = None

    # --- Groq LLM Configs ---
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL_PRIMARY: str = "openai/gpt-oss-120b"
    GROQ_MODEL_FALLBACK: str = "openai/gpt-oss-20b"
    GROQ_MODEL_ROUTER: str = "qwen/qwen3.6-27b"

    # --- Sentiment Model Configs ---
    SENTIMENT_MODEL_REPO: str = "JohnAlbarkaIbrahim/afroxlmr-mini-nigerian-sentiment"
    SENTIMENT_MODEL_SUBFOLDER: str = "v1/onnx_int8"
    SENTIMENT_MODEL_DIR: str = "./models/sentiment_onnx_int8"

    # --- Embedding Model Configs ---
    EMBEDDING_MODEL_REPO: str = "Davlan/afro-xlmr-mini"
    EMBEDDING_MODEL_DIR: str = "./models/afro_mini_onnx_int8"

    # --- ChromaDB Vector Store Configs ---
    CHROMA_MODE: str = "cloud"  # 'local' or 'cloud'
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma"
    CHROMA_COLLECTION_NAME: str = "chat_vectors"
    CHROMA_API_KEY: Optional[str] = None
    CHROMA_TENANT: Optional[str] = "default_tenant"
    CHROMA_DATABASE: Optional[str] = "default_database"
    CHROMA_HOST: Optional[str] = None
    CHROMA_PORT: Optional[int] = None

    SECRET_KEY: str = secrets.token_urlsafe(32)
    IS_CELERY_WORKER: str = "false"
    CELERY_BROKER_URL: Optional[str] = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: Optional[str] = "redis://localhost:6379/0"
    CORS_ORIGINS: List[str] = []
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_DAYS: int = 7

    CORS_ALLOW_CREDENTIALS: bool = True
    TRUSTED_HOSTS: List[str] = ["*"]

    DB_CONNECT_RETRIES: int = 3
    DB_CONNECT_BACKOFF_SECONDS: int = 2

    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB

    @model_validator(mode="before")
    @classmethod
    def assemble_database_url(cls, values):
        if not isinstance(values, dict):
            return values
        db_url = values.get("DATABASE_URL")
        if db_url and str(db_url).strip() not in ("None", "", "null"):
            url = str(db_url).strip()
            if url.startswith("postgresql://") and "asyncpg" not in url:
                values["DATABASE_URL"] = url.replace(
                    "postgresql://", "postgresql+asyncpg://", 1
                )
        else:
            user = values.get("DB_USER") or "postgres"
            password = values.get("DB_PASSWORD") or "postgres"
            host = values.get("DB_HOST") or "localhost"
            port = values.get("DB_PORT") or 5432
            db = values.get("DB_NAME") or "sentiment_db"
            values["DATABASE_URL"] = (
                f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"
            )
        return values


settings = Settings()


def setup_logging():
    """Configure global logging level and format based on settings."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Suppress noisy loggers from dependencies
    # logging.getLogger("uvicorn").setLevel(logging.WARNING)
    # logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    # logging.getLogger("asyncio").setLevel(logging.WARNING)
    # logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)
