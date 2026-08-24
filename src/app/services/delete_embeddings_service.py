# src/app/services/delete_embeddings_service.py

import logging
from src.app.services.vector_store import get_vector_store

log = logging.getLogger(__name__)


async def delete_chat_embeddings(chat_id: int):
    """Deletes all vector embeddings matching the given chat_id from VectorStore."""
    try:
        store = get_vector_store()
        await store.delete(where={"chat_id": {"$eq": chat_id}})
        log.info("Successfully deleted vector embeddings for chat_id=%s", chat_id)
    except Exception as e:
        log.error("Failed to delete vectors for chat_id=%s: %s", chat_id, e)
        raise e