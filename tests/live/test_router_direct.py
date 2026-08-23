import pytest
import json
import logging
from langchain_core.messages import HumanMessage, AIMessage
from src.app.db.session import AsyncSessionLocal
from src.app.services import router_service

# --- Config ---
TEST_CHAT_ID = 98

# --- Dummy Data ---
SAMPLE_ANALYTICS = {
    "general_dashboard": {
        "kpiMetrics": [{"label": "Total Messages", "value": 5000}],
        "timeline": [],
        "activity": {"labels": ["Text"], "participants": []}
    }
}

# --- Helper to consume the Async Generator ---
async def consume_router_stream(generator):
    """
    Consumes the generator from router_service.route_and_process.
    Reassembles the answer and extracts the final metadata.
    """
    full_answer = ""
    final_metadata = {}
    
    async for event in generator:
        # Event format: "data: <json_string>\n\n"
        clean_data = event.replace("data: ", "").strip()
        if not clean_data:
            continue
            
        try:
            parsed = json.loads(clean_data)
            
            # Check if it's the final metadata payload
            if isinstance(parsed, dict) and "route" in parsed:
                final_metadata = parsed
            else:
                # It's a string token (part of the answer)
                full_answer += str(parsed)
        except json.JSONDecodeError:
            pass
            
    return full_answer, final_metadata

# --- Fixtures ---
@pytest.fixture(scope="module")
async def db_session():
    async with AsyncSessionLocal() as session:
        yield session

# --- Tests ---

@pytest.mark.asyncio
async def test_01_fast_trap(db_session):
    """Test the Regex Greeting Trap (Tier 1)"""
    print("\n[Test 01] Fast Trap")
    
    generator = router_service.route_and_process(
        query="Hello there",
        analytics_json={},
        chat_id=TEST_CHAT_ID,
        db=db_session,
        chat_history=[]
    )
    
    answer, meta = await consume_router_stream(generator)
    
    print(f"   -> Route: {meta.get('route')}")
    print(f"   -> Answer: {answer}")
    
    assert "TIER_1_FAST" in meta["route"] or "FAST_TRAP" in meta["route"]
    assert "Hello" in answer

@pytest.mark.asyncio
async def test_02_analytics_intent(db_session):
    """Test Analytics Dashboard Intent (Tier 2)"""
    print("\n[Test 02] Analytics Intent")
    
    query = "What are the total messages?"
    generator = router_service.route_and_process(
        query=query,
        analytics_json=SAMPLE_ANALYTICS,
        chat_id=TEST_CHAT_ID,
        db=db_session,
        chat_history=[]
    )
    
    answer, meta = await consume_router_stream(generator)
    
    print(f"   -> Query: {query}")
    print(f"   -> Route: {meta.get('route')}")
    print(f"   -> Answer: {answer}")
    
    assert meta["route"] == "ANALYTICS_DASHBOARD"
    assert "5000" in answer or "5,000" in answer

@pytest.mark.asyncio
async def test_03_sql_agent(db_session):
    """Test SQL Generation Agent (Tier 3)"""
    print("\n[Test 03] SQL Agent")
    
    # A question that definitely requires DB access, not dashboard
    query = "How many messages contain the word 'love'?"
    
    generator = router_service.route_and_process(
        query=query,
        analytics_json=SAMPLE_ANALYTICS,
        chat_id=TEST_CHAT_ID,
        db=db_session,
        chat_history=[]
    )
    
    answer, meta = await consume_router_stream(generator)
    
    print(f"   -> Query: {query}")
    print(f"   -> Route: {meta.get('route')}")
    print(f"   -> Answer: {answer}")
    
    assert meta["route"] == "SQL_AGENT"
    # We can't guarantee the count, but we check if it didn't fail
    assert "error" not in answer.lower()

@pytest.mark.asyncio
async def test_04_vector_search(db_session):
    """Test Semantic Search (Tier 3)"""
    print("\n[Test 04] Vector Search")
    
    query = "What was discussed about the meeting?"
    
    generator = router_service.route_and_process(
        query=query,
        analytics_json=SAMPLE_ANALYTICS,
        chat_id=TEST_CHAT_ID,
        db=db_session,
        chat_history=[]
    )
    
    answer, meta = await consume_router_stream(generator)
    
    print(f"   -> Query: {query}")
    print(f"   -> Route: {meta.get('route')}")
    
    assert meta["route"] == "VECTOR_SEARCH"
    assert isinstance(meta.get("sources"), list)

@pytest.mark.asyncio
async def test_05_contextualization(db_session):
    """Test Chat History Rewriting"""
    print("\n[Test 05] Contextualization")
    
    history = [
        HumanMessage(content="Who sent the most messages?"),
        AIMessage(content="John Doe sent the most.")
    ]
    query = "How many did he send?"
    
    # Enable logging to see the "Contextualized: ..." log
    # pytest -s will show stdout
    
    generator = router_service.route_and_process(
        query=query,
        analytics_json=SAMPLE_ANALYTICS,
        chat_id=TEST_CHAT_ID,
        db=db_session,
        chat_history=history
    )
    
    answer, meta = await consume_router_stream(generator)
    
    print(f"   -> Original: {query}")
    print(f"   -> Route: {meta.get('route')}")
    print(f"   -> Answer: {answer}")
    
    # The route should likely be SQL_AGENT or ANALYTICS, not GENERAL
    assert meta["route"] in ["SQL_AGENT", "ANALYTICS_DASHBOARD"]