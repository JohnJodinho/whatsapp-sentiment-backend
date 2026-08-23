
import pytest
import asyncio
import uuid
from datetime import datetime, timezone
from sqlalchemy import text
from qdrant_client import AsyncQdrantClient, models as qmodels
from src.app.db.session import AsyncSessionLocal
from src.app.config import settings
from src.app.services.delete_embeddings_service import delete_chat_embeddings
from src.app import models

TEST_CHAT_ID = 141
# UNIQUE_KEYWORD = f"BlueBanana_{uuid.uuid4().hex[:6]}" # Unique marker to verify retrieval
# TEST_MESSAGE = f"The secret operation code is {UNIQUE_KEYWORD}."

@pytest.mark.asyncio
async def test_deletion():

    qdrant = AsyncQdrantClient(
        url=str(settings.QDRANT_URL),   
        api_key=settings.QDRANT_API_KEY,
        timeout=60.0
    )

    await delete_chat_embeddings(TEST_CHAT_ID)
    
    # Verify count is 0
    count_result = await qdrant.count(
        collection_name="chat_vectors",
        count_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(key="chat_id", match=qmodels.MatchValue(value=TEST_CHAT_ID))]
        )
    )
    assert count_result.count == 0
    print("✅ Deletion test passed successfully.")