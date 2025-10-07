from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import insert
from typing import List
from src.app import models, schemas



async def create_chat(db: AsyncSession, )