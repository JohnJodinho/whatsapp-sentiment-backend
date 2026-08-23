import pytest
import json
import uuid
import asyncio
from httpx import AsyncClient
from sqlalchemy import text
from typing import AsyncGenerator

# Import app components
from src.main import app  # Adjust import based on your actual main entry point
from src.app.db.session import get_db, AsyncSessionLocal
from src.app.security import get_current_user
from src.app import models

# --- Configuration ---
TEST_CHAT_ID = 98

# Dummy Analytics Payload for testing Tier 2
SAMPLE_ANALYTICS_JSON = {
    "general_dashboard": {
        "kpiMetrics": [
            {"label": "Total Messages", "value": 5000},
            {"label": "Active Days", "value": 120}
        ],
        "timeline": [],
        "activity": {"labels": ["Text", "Media"], "participants": []}
    },
    "sentiment_dashboard": {
        "kpiData": {
            "overallScore": 0.75,
            "positivePercent": 60.5,
            "negativePercent": 5.2,
            "neutralPercent": 34.3,
            "totalMessagesOrSegments": 5000
        }
    }
}

# --- Fixtures & Helpers ---

@pytest.fixture(scope="module")
async def db_session():
    async with AsyncSessionLocal() as session:
        yield session

@pytest.fixture(scope="module")
async def valid_user(db_session):
    """
    Fetches the real owner of Chat 98 to bypass auth.
    """
    result = await db_session.execute(
        text(f"SELECT owner_id FROM chats WHERE id = {TEST_CHAT_ID}")
    )
    row = result.fetchone()
    if not row:
        pytest.skip(f"Chat ID {TEST_CHAT_ID} does not exist in the live DB. Skipping tests.")
    
    owner_id = row[0]
    # Return a mock user object compliant with Pydantic/SQLAlchemy models
    user = models.User(id=owner_id, email="test@example.com")
    return user

@pytest.fixture(scope="module")
async def client(valid_user) -> AsyncGenerator[AsyncClient, None]:
    """
    Async HTTP Client with Auth Override.
    """
    # Override auth dependency to return the valid owner
    app.dependency_overrides[get_current_user] = lambda: valid_user
    
    async with AsyncClient(app=app, base_url="http://test") as c:
        yield c
    
    # Cleanup
    app.dependency_overrides = {}

async def parse_sse_stream(response):
    """
    Helper to consume SSE stream and extract the final JSON payload.
    Returns: (full_text_answer, final_metadata_json)
    """
    full_answer = ""
    final_payload = None
    
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            data_str = line.replace("data: ", "").strip()
            try:
                data = json.loads(data_str)
                # If it's a dict with 'route', it's the final payload
                if isinstance(data, dict) and "route" in data:
                    final_payload = data
                else:
                    # Otherwise it's a string token
                    full_answer += str(data)
            except json.JSONDecodeError:
                pass
                
    return full_answer, final_payload

# --- Live Tests ---

@pytest.mark.asyncio
async def test_tier1_fast_trap(client):
    """
    Intent: GREETING
    Route: TIER_1_FAST (Regex)
    Expectation: Instant response, no DB/LLM latency.
    """
    payload = {
        "question": "Hello there!",
        "analytics_json": {}
    }
    
    response = await client.post(f"/chat/{TEST_CHAT_ID}/query/streamed", json=payload)
    assert response.status_code == 200
    
    answer, metadata = await parse_sse_stream(response)
    
    print(f"\n[Fast Trap] Answer: {answer}")
    assert metadata is not None
    assert metadata["route"] == "TIER_1_FAST" or metadata["route"] == "FAST_TRAP"
    assert "Hello" in answer

@pytest.mark.asyncio
async def test_tier2_analytics_intent(client):
    """
    Intent: ANALYTICS
    Route: ANALYTICS_DASHBOARD
    Expectation: Uses the injected JSON, ignores SQL/Vector.
    """
    question = "What is the overall sentiment score?"
    payload = {
        "question": question,
        "analytics_json": SAMPLE_ANALYTICS_JSON
    }
    
    response = await client.post(f"/chat/{TEST_CHAT_ID}/query/streamed", json=payload)
    assert response.status_code == 200
    
    answer, metadata = await parse_sse_stream(response)
    
    print(f"\n[Analytics] Q: {question}")
    print(f"[Analytics] A: {answer}")
    
    assert metadata is not None
    assert metadata["route"] == "ANALYTICS_DASHBOARD"
    # It should mention 0.75 or 75%
    assert "0.75" in answer or "75" in answer

@pytest.mark.asyncio
async def test_tier3_sql_agent(client):
    """
    Intent: SQL_AGENT
    Route: SQL_AGENT
    Expectation: Generates SQL, executes against Supabase, returns count.
    """
    # Ask a question that requires counting rows (DB specific)
    question = "How many messages are there in total?"
    payload = {
        "question": question,
        "analytics_json": {} # Empty to force SQL lookup
    }
    
    response = await client.post(f"/chat/{TEST_CHAT_ID}/query/streamed", json=payload)
    assert response.status_code == 200
    
    answer, metadata = await parse_sse_stream(response)
    
    print(f"\n[SQL Agent] Q: {question}")
    print(f"[SQL Agent] A: {answer}")
    
    assert metadata is not None
    assert metadata["route"] == "SQL_AGENT"
    # Ensure it didn't refuse
    assert "cannot execute" not in answer
    # It should return a number
    assert any(char.isdigit() for char in answer)

@pytest.mark.asyncio
async def test_tier3_vector_search(client):
    """
    Intent: VECTOR_SEARCH
    Route: VECTOR_SEARCH
    Expectation: Queries Qdrant, returns sources.
    """
    # Ask something generic or specific to chat content
    question = "What is the main topic of discussion?"
    payload = {
        "question": question,
        "analytics_json": {}
    }
    
    response = await client.post(f"/chat/{TEST_CHAT_ID}/query/streamed", json=payload)
    assert response.status_code == 200
    
    answer, metadata = await parse_sse_stream(response)
    
    print(f"\n[Vector Search] Q: {question}")
    print(f"[Vector Search] A: {answer}")
    
    assert metadata is not None
    assert metadata["route"] == "VECTOR_SEARCH"
    # Check if we got sources back (Qdrant hit)
    # Note: If Qdrant is empty for this chat, sources might be empty, but route should be correct.
    assert isinstance(metadata["sources"], list)

@pytest.mark.asyncio
async def test_chat_history_contextualization(client):
    """
    Intent: CONTEXTUALIZATION + SQL/ANALYTICS
    Expectation: Router rewrites "he" to specific name based on history.
    """
    # 1. Seed History (Using the persist endpoint implicitly or mocking history)
    # Since this is a live test, we rely on the app's history fetcher.
    # To test this LIVE, we must assume there is some history or make two calls.
    
    # Call 1: Establish context
    await client.post(f"/chat/{TEST_CHAT_ID}/query/streamed", json={
        "question": "Who is the most active participant?",
        "analytics_json": SAMPLE_ANALYTICS_JSON
    })
    
    # Call 2: Follow up with pronoun
    question = "How many messages did HE send?" # "HE" refers to the answer from Call 1
    payload = {
        "question": question,
        "analytics_json": SAMPLE_ANALYTICS_JSON
    }
    
    response = await client.post(f"/chat/{TEST_CHAT_ID}/query/streamed", json=payload)
    answer, metadata = await parse_sse_stream(response)
    
    print(f"\n[Context] Q: {question}")
    print(f"[Context] A: {answer}")
    print(f"[Context] Route: {metadata['route']}")
    
    # The router logs should show the rewritten query (observable in console output)
    assert response.status_code == 200