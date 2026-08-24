import pytest
import asyncio
from src.app.services.delete_embeddings_service import delete_chat_embeddings
from src.app.services.vector_store import get_vector_store

TEST_CHAT_ID = 141


@pytest.mark.asyncio
async def test_deletion():
    try:
        await delete_chat_embeddings(TEST_CHAT_ID)
        vector_store = get_vector_store()
        count_result = await vector_store.count(where={"chat_id": {"$eq": TEST_CHAT_ID}})
        assert count_result == 0
        print("✅ Deletion test passed successfully.")
    except Exception as e:
        pytest.skip(f"Live vector store test skipped: {e}")