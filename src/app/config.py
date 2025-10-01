from pydantic import AnyUrl, AnyHttpUrl, field_validator, PostgresDsn, model_validator
try:
    from pydantic import BaseSettings  # For pydantic v1
except ImportError:
    from pydantic_settings import BaseSettings  # For pydantic v2
from typing import Optional, List
import secrets

class Settings(BaseSettings):
    PROJECT_NAME: str = "WhatsApp Sentiment Dashboard"
    ENVIRONMENT: str = "development"

    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str 
    DB_PORT: int
    DB_NAME: str
    DATABASE_URL: Optional[PostgresDsn] = None

    SECRET_KEY: str = secrets.token_urlsafe(32)
    CELERY_BROKER_URL: Optional[str] = None
    CORS_ORIGINS: List[AnyHttpUrl] = []

    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB

    @model_validator(mode="before")
    @classmethod
    def assemble_database_url(cls, values):
        if not values.get("DATABASE_URL"):
           user = values.get("DB_USER")
           password = values.get("DB_PASSWORD")
           host = values.get("DB_HOST")
           port = values.get("DB_PORT")
           db = values.get("DB_NAME")

           values["DATABASE_URL"] = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"
        return values
    
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        raise ValueError(v)
    class Config:
        env_file = ".env"

settings = Settings()


