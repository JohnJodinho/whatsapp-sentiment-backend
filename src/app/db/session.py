from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    create_async_engine, 
    async_sessionmaker
)


from sqlalchemy.pool import NullPool, QueuePool
import os

from src.app.config import settings
import logging


logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)

DATABASE_URL = str(settings.DATABASE_URL)

IS_CELERY_WORKER = os.getenv("IS_CELERY_WORKER", "false").lower() == "true"

engine_args = {
    "echo": False,
    "echo_pool": False,
    "logging_name": 'sqlalchemy.engine',
    "pool_pre_ping": True,
}

if IS_CELERY_WORKER:
    engine_args["poolclass"] = NullPool
else:
    engine_args["pool_size"] = 10
    engine_args["max_overflow"] = 20
    engine_args["pool_recycle"] = 3600

engine = create_async_engine(
    DATABASE_URL,

    **engine_args               
)


AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency injection to get a new async session.
    """
    async with AsyncSessionLocal() as session:
        yield session