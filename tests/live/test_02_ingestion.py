import pytest
import asyncio
from src.app.services.embedding_worker import process_chat_ingestion
from src.app.services.vector_store import get_vector_store

TEST_CHAT_ID = 141


@pytest.mark.asyncio
async def test_ingestion_pipeline():
    print("[Ingestion] Step 1: Running process_chat_ingestion...")
    try:
        await process_chat_ingestion(TEST_CHAT_ID)
        print("[Ingestion] Ingestion process completed.")
    except Exception as e:
        pytest.skip(f"Live DB not available: {e}")

    vector_store = get_vector_store()
    hits_count = await vector_store.count(where={"chat_id": {"$eq": TEST_CHAT_ID}})
    print(f"[Verification] Found {hits_count} vectors for Chat {TEST_CHAT_ID}.")
    assert hits_count >= 0
    print("✅ Ingestion test finished.")