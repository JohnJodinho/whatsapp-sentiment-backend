# src/app/services/embedding_worker.py

import asyncio
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

from qdrant_client import AsyncQdrantClient, models as qmodels
from src.app.config import settings


log = logging.getLogger(__name__)


QDRANT_COLLECTION = "chat_vectors"


# Initialize Client
qdrant = AsyncQdrantClient(
    url=str(settings.QDRANT_URL),
    api_key=settings.QDRANT_API_KEY
)



async def delete_chat_embeddings(chat_id: int):
    """
    Deletes all points in Qdrant matching the given chat_id.
    """
    try:
        # We use FilterSelector to delete by criteria (chat_id) rather than specific point IDs
        await qdrant.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="chat_id",
                            match=qmodels.MatchValue(value=chat_id)
                        )
                    ]
                )
            ),
            # Optional: wait=True if you need to confirm deletion immediately, 
            # but keep it False for performance on large deletions.
            wait=True 
        )
        log.info(f"Sent delete command for all vectors with chat_id={chat_id}")
    except Exception as e:
        log.error(f"Failed to delete vectors for chat_id={chat_id}: {e}")
        raise e