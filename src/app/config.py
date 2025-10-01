from pydantic import BaseSettings, AnyUrl, AnyHttpUrl
from typing import Optional
import secrets

class Settings(BaseSettings):
    PROJECT_NAME: str = "WhatsApp Sentiment Dashboard"
    DATABASE_URL: AnyUrl
    SECRET_KEY: str = "change-me"
    CELERY_BROKER_URL: str | None = None

    class Config:
        env_file = ".env"

settings = Settings()


