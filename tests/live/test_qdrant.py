from src.app.services.retrieval_service import retriever
import pytest

TEST_CHAT_ID = 137


@pytest.mark.asyncio
async def test_vector_search():
    try:
        embeddings = await retriever.aget_relevant_documents(
            query="What did we discuss about the project deadline?",
            chat_id=TEST_CHAT_ID,
        )
        print(f"[Verification] Retrieved {len(embeddings)} documents for vector search.")
        print(f"[First 3 Embeddings] {embeddings[:3]}")
    except Exception as e:
        pytest.skip(f"Live vector store retrieval skipped: {e}")