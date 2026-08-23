import pytest
import json
import asyncio
from httpx import AsyncClient
from sqlalchemy import text

# --- Adjust these imports to match your project structure ---
from src.app.main import app
from src.app.db.session import AsyncSessionLocal
from src.app.security import get_current_user
from src.app import models

# --- Config ---
TEST_CHAT_ID = 98

# --- Fixtures (Setup) ---

@pytest.fixture(scope="module")
async def db_session():
    """Opens a real connection to Supabase."""
    async with AsyncSessionLocal() as session:
        yield session

@pytest.fixture(scope="module")
async def valid_user(db_session):
    """
    Dynamically finds the REAL owner of Chat 98.
    This ensures we don't get 403 Forbidden errors.
    """
    # Query the 'chats' table to find the owner_id
    result = await db_session.execute(
        text(f"SELECT owner_id FROM chats WHERE id = {TEST_CHAT_ID}")
    )
    row = result.fetchone()
    
    if not row:
        pytest.skip(f"CRITICAL: Chat ID {TEST_CHAT_ID} not found in DB. Cannot run live tests.")
    
    owner_id = row[0]
    print(f"\n   [Setup] Found Owner ID: {owner_id} for Chat {TEST_CHAT_ID}")
    
    # Return a mock User object that satisfies the dependency
    # We only need the ID to match for the route check
    return models.User(id=owner_id, email="live_test@example.com")

@pytest.fixture(scope="module")
async def client(valid_user):
    """
    Creates a test client that automagically authenticates as the correct user.
    """
    # FORCE the app to think this user is logged in
    app.dependency_overrides[get_current_user] = lambda: valid_user
    
    async with AsyncClient(app=app, base_url="http://test") as c:
        yield c
    
    # Cleanup: Remove the override so we don't break other tests later
    app.dependency_overrides = {}

async def parse_sse(response):
    """Helper to read the streaming response."""
    full_answer = ""
    route = None
    
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            content = line.replace("data: ", "").strip()
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "route" in data:
                    route = data["route"]
                else:
                    full_answer += str(data)
            except:
                pass
    return full_answer, route

# --- The Test ---

@pytest.mark.asyncio
async def test_fast_trap_greeting(client):
    """
    Scenario: User says 'Hello'.
    Expectation: 
    1. Status 200 OK.
    2. Route is 'TIER_1_FAST' (or 'FAST_TRAP').
    3. Response contains a static greeting.
    4. NO calls to OpenAI or Qdrant should happen here (Latency should be low).
    """
    print(f"\n   [Step 1] Sending 'Hello' to Chat {TEST_CHAT_ID}...")
    
    payload = {
        "question": "Hello",
        "analytics_json": {}
    }
    
    response = await client.post(
        f"/chat/{TEST_CHAT_ID}/query/streamed", 
        json=payload
    )
    
    assert response.status_code == 200, f"API Failed with {response.text}"
    
    answer, route = await parse_sse(response)
    
    print(f"   [Result] Route: {route}")
    print(f"   [Result] Answer: {answer}")
    
    # Assertions
    assert "TIER_1_FAST" in route or "FAST_TRAP" in route
    assert "Hello" in answer
    print("   ✅ Step 1 (Fast Trap) Passed!")