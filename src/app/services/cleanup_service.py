import asyncio
import uuid
import logging
from sqlalchemy import select, delete
from src.app.db.session import AsyncSessionLocal  
from src.app import models, crud
from src.app.services.delete_embeddings_service import delete_chat_embeddings

log = logging.getLogger(__name__)


async def delete_single_chat_background(chat_id: int):
    """
    Background task to delete a single chat and its vector embeddings.
    """
    async with AsyncSessionLocal() as db: 
        try:
            log.info(f"Starting background deletion for chat_id={chat_id}")

            await crud.delete_chat(db, chat_id, should_commit=True)

            try:
                await delete_chat_embeddings(chat_id)
            except Exception as e:
                log.error(f"Failed to delete Qdrant vectors for chat {chat_id}: {e}")

            log.info(f"Background deletion completed for chat_id={chat_id}")

        except Exception as e:
            await db.rollback()
            log.error(f"Background chat deletion failed for chat_id={chat_id}: {e}")

async def cleanup_expired_user_data(user_uuid: uuid.UUID):
    """
    Background task to clean up user data and embeddings.
    Creates its own DB session so it runs independently of the request lifecycle.
    """
    async with AsyncSessionLocal() as db:
        try:
            log.info(f"Starting background cleanup for user {user_uuid}")

            result = await db.execute(
                select(models.Chat.id).where(models.Chat.owner_id == user_uuid)
            )
            chat_ids = result.scalars().all()

            #
            await db.execute(delete(models.User).where(models.User.id == user_uuid))
            await db.commit()

            if chat_ids:
                tasks = [delete_chat_embeddings(chat_id) for chat_id in chat_ids]
                await asyncio.gather(*tasks, return_exceptions=True)

            log.info(f"Background cleanup completed for user {user_uuid}")

        except Exception as e:
            await db.rollback()
            log.error(f"Background cleanup failed for user {user_uuid}: {e}")
            