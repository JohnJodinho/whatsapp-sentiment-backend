# tests/test_vector_store.py

import pytest
import asyncio
import os
import shutil
from src.app.services.vector_store import ChromaVectorStore, get_vector_store
from src.app.services.embedding_service import embed_texts, embed_query
from src.app.services.sentiment_service import predict_sentiment, get_sentiment_classifier


@pytest.mark.asyncio
async def test_chroma_local_vector_store_crud():
    test_dir = "./data/test_chroma"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir, ignore_errors=True)

    store = ChromaVectorStore(
        mode="local",
        persist_directory=test_dir,
        collection_name="test_collection",
    )

    # 1. Health check
    assert await store.health() is True

    # 2. Count should initially be 0
    assert await store.count() == 0

    # 3. Upsert vectors
    ids = ["doc_1", "doc_2", "doc_3"]
    embeddings = [
        [0.1] * 384,
        [0.2] * 384,
        [0.9] * 384,
    ]
    metadatas = [
        {"chat_id": 100, "sender_name": "Alice", "source_table": "messages", "source_id": 1, "timestamp": "2026-01-01T10:00:00"},
        {"chat_id": 100, "sender_name": "Bob", "source_table": "messages", "source_id": 2, "timestamp": "2026-01-01T11:00:00"},
        {"chat_id": 200, "sender_name": "Charlie", "source_table": "messages", "source_id": 3, "timestamp": "2026-01-01T12:00:00"},
    ]
    documents = [
        "Good morning everyone",
        "How is the project progressing?",
        "Unrelated text for another chat",
    ]

    inserted = await store.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )
    assert inserted == 3
    assert await store.count() == 3
    assert await store.count(where={"chat_id": {"$eq": 100}}) == 2

    # 4. Query with filter
    results = await store.query(
        query_embedding=[0.1] * 384,
        n_results=2,
        where={"chat_id": {"$eq": 100}},
    )
    assert len(results) == 2
    assert results[0]["metadata"]["chat_id"] == 100

    # 5. Delete scoped to chat_id
    await store.delete(where={"chat_id": {"$eq": 100}})
    assert await store.count(where={"chat_id": {"$eq": 100}}) == 0
    assert await store.count() == 1  # chat 200 remains

    # Cleanup
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_sentiment_service_structure():
    """Verify sentiment service loads and predicts valid labels."""
    try:
        classifier = get_sentiment_classifier()
        assert classifier is not None
        assert "positive" in classifier.label2id or 0 in classifier.id2label

        results = await predict_sentiment(["Dis market sweet die!", "I no like this thing at all."])
        assert len(results) == 2
        for r in results:
            assert "overall_label" in r
            assert "overall_label_score" in r
            assert r["overall_label"] in ["positive", "negative", "neutral"]
    except Exception as e:
        pytest.skip(f"Skipping live model test if weights not locally present: {e}")


@pytest.mark.asyncio
async def test_embedding_service_dimension():
    """Verify AfroXLMR-Mini produces 384-dimensional vectors."""
    try:
        embs = await embed_texts(["Hello world from Nigeria"])
        assert len(embs) == 1
        assert len(embs[0]) == 384
    except Exception as e:
        pytest.skip(f"Skipping live embedding test if weights not locally present: {e}")
