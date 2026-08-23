
import pytest
import asyncio
import uuid
from datetime import datetime, timezone
from sqlalchemy import text
from qdrant_client import QdrantClient, models as qmodels
from src.app.db.session import AsyncSessionLocal
from src.app.services.embedding_worker import process_chat_ingestion, _ensure_collection_exists
from src.app.config import settings
from src.app import models

TEST_CHAT_ID = 141
# UNIQUE_KEYWORD = f"BlueBanana_{uuid.uuid4().hex[:6]}" # Unique marker to verify retrieval
# TEST_MESSAGE = f"The secret operation code is {UNIQUE_KEYWORD}."

@pytest.mark.asyncio
async def test_ingestion_pipeline():

   
    print("[Ingestion] Step 1: Running process_chat_ingestion...")
    await process_chat_ingestion(TEST_CHAT_ID)
    print("[Ingestion] Ingestion process completed.")
    # Verify ingestion by querying Qdrant directly
  

    # Verify ingestion by querying Qdrant directly
    qdrant = QdrantClient(
        url=str(settings.QDRANT_URL),   
        api_key=settings.QDRANT_API_KEY
    )

    print("[Verification] Querying Qdrant for ingested data...")
    
    # FIX: Use count() instead of search(). search() was removed in recent versions.
    # This also avoids the need for a dummy vector.
    count_result = qdrant.count(
        collection_name="chat_vectors",
        count_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="chat_id",
                    match=qmodels.MatchValue(value=TEST_CHAT_ID)
                )
            ]
        )
    )

    # Validate that we have at least one vector
    hits_count = count_result.count
    assert hits_count > 0, "No vectors found for the test chat in Qdrant."
    
    print(f"[Verification] Found {hits_count} vectors for Chat {TEST_CHAT_ID}.")
    print("✅ Ingestion test passed successfully.")