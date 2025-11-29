from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    create_async_engine, 
    async_sessionmaker
)
from src.app.config import settings
import logging


logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)

DATABASE_URL = str(settings.DATABASE_URL)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    echo_pool=False,
    
    logging_name='sqlalchemy.engine', 
    pool_pre_ping=True,              
    pool_recycle=3600                
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