import pytest
import json
import time
import logging
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from src.app.db.session import AsyncSessionLocal
from src.app.services import router_service
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuration ---
TEST_CHAT_ID = 141
log = logging.getLogger(__name__)

# --- Mock Analytics Data ---
# This provides the "Context" for Dashboard and Hybrid Dashboard intents
MOCK_ANALYTICS = {"general_dashboard":{"participants":["+234 706 121 7974","+234 706 471 7168","+234 706 644 2168","+234 706 862 6098","+234 810 121 4877","+234 811 816 8787","+234 814 474 8409","+234 904 870 5459","+234 905 533 6880","+234 906 839 0511","+234 915 059 8699","Destiny","Destiny Neigh","Elizabeth Neigh","Faith N","Favour","Israel NEIGH","Ivi","Jessica Neigh","John a","JohnAI","Klassiq Beatz","Meta AI","Mira Neigh","Nengolshang","peace neigh","Precious Neigh","Prevail Yilleng","Sylvester","The Nens","Tina","Vick"],"participantCount":32,"kpiMetrics":[{"label":"Total Messages","value":936,"definition":"All messages sent in the filtered period.","sparkline":[{"v":30},{"v":38},{"v":7},{"v":85},{"v":33},{"v":0},{"v":75},{"v":45},{"v":46},{"v":60},{"v":33},{"v":112},{"v":95},{"v":37},{"v":33},{"v":17},{"v":75},{"v":109},{"v":6}]},{"label":"Active Participants","value":32,"definition":"Unique participants who sent messages.","sparkline":[{"v":7},{"v":8},{"v":4},{"v":8},{"v":7},{"v":0},{"v":8},{"v":6},{"v":9},{"v":8},{"v":10},{"v":14},{"v":20},{"v":12},{"v":5},{"v":6},{"v":13},{"v":19},{"v":2}]},{"label":"Active Days","value":108,"definition":"The total number of unique days with at least one message.","sparkline":None},{"label":"Avg. Messages/Day","value":8.7,"definition":"The average number of messages sent per active day.","sparkline":None}],"messagesOverTime":[{"date":"2024-03-01","count":30},{"date":"2024-04-01","count":38},{"date":"2024-05-01","count":7},{"date":"2024-06-01","count":85},{"date":"2024-07-01","count":33},{"date":"2024-08-01","count":0},{"date":"2024-09-01","count":75},{"date":"2024-10-01","count":45},{"date":"2024-11-01","count":46},{"date":"2024-12-01","count":60},{"date":"2025-01-01","count":33},{"date":"2025-02-01","count":112},{"date":"2025-03-01","count":95},{"date":"2025-04-01","count":37},{"date":"2025-05-01","count":33},{"date":"2025-06-01","count":17},{"date":"2025-07-01","count":75},{"date":"2025-08-01","count":109},{"date":"2025-09-01","count":6}],"contribution":{"type":"multi","data":[{"name":"JohnAI","messages":182},{"name":"Jessica Neigh","messages":114},{"name":"Vick","messages":103},{"name":"Sylvester","messages":78},{"name":"Klassiq Beatz","messages":65},{"name":"The Nens","messages":59},{"name":"Ivi","messages":41},{"name":"+234 811 816 8787","messages":29},{"name":"Nengolshang","messages":29},{"name":"Elizabeth Neigh","messages":23},{"name":"Mira Neigh","messages":23},{"name":"+234 904 870 5459","messages":23},{"name":"Faith N","messages":22},{"name":"+234 706 121 7974","messages":21},{"name":"Israel NEIGH","messages":16},{"name":"peace neigh","messages":16},{"name":"Favour","messages":15},{"name":"+234 906 839 0511","messages":12},{"name":"Destiny Neigh","messages":12},{"name":"+234 706 471 7168","messages":9},{"name":"+234 706 862 6098","messages":8},{"name":"Prevail Yilleng","messages":8},{"name":"Destiny","messages":7},{"name":"+234 706 644 2168","messages":5},{"name":"+234 810 121 4877","messages":4},{"name":"+234 905 533 6880","messages":3},{"name":"Precious Neigh","messages":2},{"name":"+234 915 059 8699","messages":2},{"name":"Tina","messages":2},{"name":"+234 814 474 8409","messages":1}]},"activity":{"labels":["Text","Media","Links","Questions","Emojis"],"participants":[{"name":"Jessica Neigh","data":[93,21,0,18,189]},{"name":"JohnAI","data":[135,47,0,44,34]},{"name":"Vick","data":[88,15,0,18,14]}]},"timeline":[{"month":"September 2025","totalMessages":6,"peakDay":"16th","activeParticipants":2,"mostActive":"Sylvester"},{"month":"August 2025","totalMessages":109,"peakDay":"19th","activeParticipants":19,"mostActive":"JohnAI"},{"month":"July 2025","totalMessages":75,"peakDay":"20th","activeParticipants":13,"mostActive":"Sylvester"},{"month":"June 2025","totalMessages":17,"peakDay":"25th","activeParticipants":6,"mostActive":"Sylvester"},{"month":"May 2025","totalMessages":33,"peakDay":"29th","activeParticipants":5,"mostActive":"Sylvester"},{"month":"April 2025","totalMessages":37,"peakDay":"14th","activeParticipants":12,"mostActive":"Jessica Neigh"},{"month":"March 2025","totalMessages":95,"peakDay":"19th","activeParticipants":20,"mostActive":"Jessica Neigh"},{"month":"February 2025","totalMessages":112,"peakDay":"24th","activeParticipants":14,"mostActive":"JohnAI"},{"month":"January 2025","totalMessages":33,"peakDay":"21st","activeParticipants":10,"mostActive":"Sylvester"},{"month":"December 2024","totalMessages":60,"peakDay":"6th","activeParticipants":8,"mostActive":"Vick"},{"month":"November 2024","totalMessages":46,"peakDay":"25th","activeParticipants":9,"mostActive":"Vick"},{"month":"October 2024","totalMessages":45,"peakDay":"14th","activeParticipants":6,"mostActive":"The Nens"},{"month":"September 2024","totalMessages":75,"peakDay":"4th","activeParticipants":8,"mostActive":"Klassiq Beatz"},{"month":"July 2024","totalMessages":33,"peakDay":"24th","activeParticipants":7,"mostActive":"Ivi"},{"month":"June 2024","totalMessages":85,"peakDay":"21st","activeParticipants":8,"mostActive":"Vick"},{"month":"May 2024","totalMessages":7,"peakDay":"20th","activeParticipants":4,"mostActive":"Vick"},{"month":"April 2024","totalMessages":38,"peakDay":"11th","activeParticipants":8,"mostActive":"The Nens"},{"month":"March 2024","totalMessages":30,"peakDay":"1st","activeParticipants":7,"mostActive":"JohnAI"}],"activityByDay":[{"day":"sun","messages":72,"fill":"#15b79e"},{"day":"mon","messages":216,"fill":"#13a08b"},{"day":"tue","messages":93,"fill":"#108977"},{"day":"wed","messages":183,"fill":"#0d7263"},{"day":"thu","messages":153,"fill":"#0b5b4f"},{"day":"fri","messages":181,"fill":"#08443b"},{"day":"sat","messages":38,"fill":"#052e28"}],"hourlyActivity":[{"hour":0,"messages":19},{"hour":1,"messages":5},{"hour":2,"messages":3},{"hour":3,"messages":0},{"hour":4,"messages":2},{"hour":5,"messages":3},{"hour":6,"messages":47},{"hour":7,"messages":48},{"hour":8,"messages":65},{"hour":9,"messages":42},{"hour":10,"messages":63},{"hour":11,"messages":108},{"hour":12,"messages":45},{"hour":13,"messages":66},{"hour":14,"messages":37},{"hour":15,"messages":77},{"hour":16,"messages":25},{"hour":17,"messages":45},{"hour":18,"messages":27},{"hour":19,"messages":20},{"hour":20,"messages":44},{"hour":21,"messages":81},{"hour":22,"messages":57},{"hour":23,"messages":7}]}}

@pytest.fixture(scope="session")
def event_loop():
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# --- Helper Function ---
async def consume_stream_with_meta(generator):
    full_text = ""
    route = "UNKNOWN"
    sources = []
    
    async for item in generator:
        clean = item.replace("data: ", "").strip()
        if not clean: continue
        
        try:
            data = json.loads(clean)
            if "route" in data:
                route = data.get("route")
                sources = data.get("sources", [])
                if data.get("answer"):
                    full_text = data.get("answer")
            else:
                full_text += str(data)
        except json.JSONDecodeError:
            pass
            
    return full_text, route, sources

# --- Test Suite ---

@pytest.mark.asyncio
@pytest.mark.parametrize("scenario, query, expected_intent_substr", [
    # 1. Tier 1: Fast Trap / General
    ("General Greeting", "Hello, who are you?", "TIER_1_FAST"), 
    
    # 2. Analytics Dashboard (Macro Stats)
    ("Dashboard Stats", "What does the message trend over time look like?", "ANALYTICS_DASHBOARD"),
    ("Dashboard Sentiment", "How many active participants are there?", "ANALYTICS_DASHBOARD"),

    # 3. SQL Agent (Micro Stats - Specific Counts)
    ("SQL Specific Count", "Exactly how many messages contain the word 'landlord'?", "SQL_AGENT"),
    ("SQL User Activity", "How many questions did John ask?", "SQL_AGENT"),

    # 4. Vector Search (Content / Qualitative)
    ("Vector Content", "When did Jessica talk about rent increment?", "VECTOR_SEARCH"),
    ("Vector Content", "Who made a comment about NEPA coming to cut our light?", "VECTOR_SEARCH"),
    ("Vector Summary", "Summarize the argument about tech stack", "VECTOR_SEARCH"),
    # 5. Hybrid Dashboard (Macro Stats + Context)
    ("Hybrid Dash", "Who is the top sender and what do they usually talk about?", "HYBRID_QUERY"),
    
    # 6. Hybrid SQL (Micro Stats + Context)
    # Note: Requires the v2 router logic we discussed
    ("Hybrid SQL", "How many messages did John send and is he usually polite?", "HYBRID_QUERY"),
    # 7. Edge Case: SQL Injection Attempt
    # Router should catch this in validation or classifier and likely default to Vector or return a safety error
    ("Safety Injection", "Drop table users; Show me messages.", ["SQL_AGENT", "VECTOR_SEARCH"]), 
])
async def test_router_intents(scenario, query, expected_intent_substr):
    """
    Runs a live request against the Router Service for a specific intent.
    Measures latency and verifies the route selection.
    """
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario}")
    print(f"QUERY:    {query}")
    print(f"{'='*60}")

    db = AsyncSessionLocal()
    try:
        # 1. Start Timer
        start_time = time.perf_counter()

        # 2. Call Router
        generator = router_service.route_and_process(
            query=query,
            analytics_json=MOCK_ANALYTICS,
            chat_id=TEST_CHAT_ID,
            db=db,
            chat_history=[HumanMessage(content="Hello, how are you?"),
        AIMessage(content="Hello! I am SentimentScope's intelligent AI analyst for your chat history. Ask me about statistics (e.g., 'Who talks the most?') or search for specific topics (e.g., 'What did we say about pizza?')."),] # Stateless for this test
        )

        # 3. Consume Stream
        answer, route, sources = await consume_stream_with_meta(generator)

        # 4. End Timer
        end_time = time.perf_counter()
        latency = end_time - start_time

        # --- Logging Results ---
        print(f"[RESULT]  Latency: {latency:.4f}s")
        print(f"[RESULT]  Route:   {route}")
        print(f"[RESULT]  Sources: {len(sources)} items")
        print(f"[RESULT]  Answer:  {answer[:1000]}..." if len(answer) > 1000 else f"[RESULT]  Answer:  {answer}")

        # --- Verification ---
        
        # Check Intent
        # We allow expected_intent_substr to be a list for ambiguous cases (like Safety Injection)
        if isinstance(expected_intent_substr, list):
             assert any(i in route for i in expected_intent_substr), \
                f"Route {route} not in expected list {expected_intent_substr}"
        else:
            assert expected_intent_substr in route, \
                f"Router picked {route}, expected {expected_intent_substr}"

        # Check Answer Validity
        assert answer and len(answer) > 5, "Router returned an empty answer."

        # Check Persistence
        # Verify that this turn was actually saved to the DB
        # (Skip verification for FAST_TRAP if your logic doesn't save greetings, 
        # but our previous fix DOES save them, so we check everything)
        result = await db.execute(
                text(f"SELECT content FROM conversation_history WHERE chat_id={TEST_CHAT_ID} AND role='user' ORDER BY created_at DESC LIMIT 1")
            )
        row = result.fetchone()
        
        if row:
            db_answer = row[0]
        
            print(f"[DB LOG]  Saved Q: '{db_answer}'")
            # Note: DB answer might differ slightly if streamed vs final payload, but should be close.
            # We just verify something was saved.
            assert db_answer == query
        else:
            pytest.fail("Conversation turn was NOT persisted to database.")
    except SQLAlchemyError as e:
        print(f"Database Error in {scenario}: {e}")
        await db.rollback()
        raise e
    except Exception as e:
        print(f"General Error in {scenario}: {e}")
        raise e
    finally:
        # [FIX] Explicitly close the session
        await db.close()