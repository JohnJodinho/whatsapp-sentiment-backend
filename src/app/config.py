from pydantic import AnyUrl, AnyHttpUrl, field_validator, PostgresDsn, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List
import secrets
import logging
import logging.config



class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    PROJECT_NAME: str = "SentimentScope API"
    APP_NAME: str = PROJECT_NAME  
    ENVIRONMENT: str = "development"

    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str 
    DB_PORT: int
    DB_NAME: str
    DATABASE_URL: Optional[PostgresDsn] = None

    # Azure OpenAI configs...
    AZURE_OPENAI_ENDPOINT: AnyHttpUrl
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_DEPLOYMENT: str
    AZURE_OPENAI_API_VERSION: str
    AZURE_OPENAI_ENDPOINT_CONTEXT_4: AnyHttpUrl
    AZURE_OPENAI_DEPLOYMENT_CONTEXT_4: str
    AZURE_OPENAI_API_KEY_CONTEXT_4: str
    AZURE_OPENAI_API_VERSION_CONTEXT_4: str
    AZURE_OPENAI_ENDPOINT_ROUTER: AnyHttpUrl
    AZURE_OPENAI_DEPLOYMENT_ROUTER: str
    AZURE_OPENAI_API_KEY_ROUTER: str
    AZURE_OPENAI_API_VERSION_ROUTER: str
    # AZURE_OPENAI_ENDPOINT_MAIN: AnyHttpUrl
    # AZURE_OPENAI_DEPLOYMENT_MAIN: str
    # AZURE_OPENAI_API_KEY_MAIN: str
    # AZURE_OPENAI_API_VERSION_MAIN: str
    AZURE_OPENAI_ENDPOINT_CONTEXT_4O: AnyHttpUrl
    AZURE_OPENAI_DEPLOYMENT_CONTEXT_4O: str
    AZURE_OPENAI_API_KEY_CONTEXT_4O: str
    AZURE_OPENAI_API_VERSION_CONTEXT_4O: str
    AZURE_OPENAI_ENDPOINT_MAIN_4O: AnyHttpUrl
    AZURE_OPENAI_DEPLOYMENT_MAIN_4O: str
    AZURE_OPENAI_API_KEY_MAIN_4O: str
    AZURE_OPENAI_API_VERSION_MAIN_4O: str
    AZURE_OPENAI_ENDPOINT_MAIN_4O_MINI: AnyHttpUrl
    AZURE_OPENAI_DEPLOYMENT_MAIN_4O_MINI: str
    AZURE_OPENAI_API_KEY_MAIN_4O_MINI: str
    AZURE_OPENAI_API_VERSION_MAIN_4O_MINI: str
    AZURE_LANGUAGE_ENDPOINT: AnyHttpUrl
    AZURE_LANGUAGE_KEY: str

    SECRET_KEY: str = secrets.token_urlsafe(32)
    IS_CELERY_WORKER: str = "false"
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None
    QDRANT_URL: str
    QDRANT_API_KEY: str
    CORS_ORIGINS: List[str] = []
    JWT_ALGORITHM: str
    JWT_ACCESS_TOKEN_EXPIRE_DAYS: int 
    
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
        # 1. Check if URL is provided directly
        if values.get("DATABASE_URL"):
            url = str(values.get("DATABASE_URL"))
            # If it starts with postgresql:// but not postgresql+asyncpg://, fix it
            if url.startswith("postgresql://") and "asyncpg" not in url:
                values["DATABASE_URL"] = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        
        # 2. If no URL provided, assemble it from parts (your existing logic)
        elif not values.get("DATABASE_URL"):
            user = values.get("DB_USER")
            password = values.get("DB_PASSWORD")
            host = values.get("DB_HOST")
            port = values.get("DB_PORT")
            db = values.get("DB_NAME")

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