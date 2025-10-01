from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from src.app.config import settings
# from .base import Base


DATABASE_URL = str(settings.DATABASE_URL)

engine = create_async_engine(DATABASE_URL, echo=True, future=True)