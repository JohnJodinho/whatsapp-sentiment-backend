import pytest
import json
from langchain_core.messages import HumanMessage, AIMessage
from src.app.db.session import AsyncSessionLocal
from src.app.services import router_service
from sqlalchemy import text

TEST_CHAT_ID = 98

async def consume_stream(generator):
    full_text = ""
    async for item in generator:
        clean = item.replace("data: ", "").strip()
        try:
            data = json.loads(clean)
            if "route" not in data: full_text += str(data)
        except: pass
    return full_text

@pytest.mark.asyncio
async def test_multiturn_context():
    # 1. Setup Fake History
    # User asked about "Agent Smith" previously
    fake_history = [
        HumanMessage(content="Who John"),
        AIMessage(content="John is a participant in the chat."),
    ]
    
    # 2. Ask Ambiguous Question
    ambiguous_query = "How many messages does he have and what can you say about his conversation style?"
    
    print(f"\n[Multi-Turn] History: {[m.content for m in fake_history]}")
    print(f"[Multi-Turn] Current Query: '{ambiguous_query}'")
    
    async with AsyncSessionLocal() as db:
        # We expect the 'standalone_question' (internal) to become "Where does Agent Smith live?"
        # We can verify this implicitly by checking if the LLM attempts to answer about "Agent Smith" 
        # or simply by checking that it doesn't say "Who is 'he'?"
        
        generator = router_service.route_and_process(
            query=ambiguous_query,
            analytics_json={},
            chat_id=TEST_CHAT_ID,
            db=db,
            chat_history=fake_history
        )
        
        answer = await consume_stream(generator)
        print(f"[Multi-Turn] Answer: {answer}")
        
        # Verification:
        # If context failed, LLM would ask "Who are you referring to?"
        # If context worked, it will try to find info about Agent Smith (likely failing retrieval, but answering definitively)
        assert "who" not in answer.lower(), "Router failed to resolve pronoun 'he' from history."

        # 3. Verify Persistence
        # The router should have saved this interaction to Postgres
        # We check the last message in conversation_history
        result = await db.execute(
            text(f"SELECT content FROM conversation_history WHERE chat_id={TEST_CHAT_ID} ORDER BY created_at DESC LIMIT 1")
        )
        last_row = result.fetchone()
        saved_answer = last_row[0]
        
        print(f"[Multi-Turn] DB Persisted Answer: {saved_answer}")
        # The saved answer in DB should match the generated answer
        assert saved_answer == answer