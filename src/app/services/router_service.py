import logging
import re
import json
import asyncio
from typing import Dict, Any, List, AsyncGenerator

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_openai import AzureChatOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from sqlalchemy.exc import (
    SQLAlchemyError,
    ProgrammingError,
    OperationalError,
    DataError
)
from src.app.config import settings
from src.app.services.retrieval_service import retriever
from src.app.utils.serializers import serialize_analytics


from src.app.schemas import EmbeddingStatusEnum
from src.app import crud
log = logging.getLogger(__name__)


router_llm = AzureChatOpenAI(
    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_ROUTER,
    api_version=settings.AZURE_OPENAI_API_VERSION_ROUTER,
    azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT_ROUTER),
    api_key=settings.AZURE_OPENAI_API_KEY_ROUTER,
    temperature=0.0,
    max_tokens=200
)

main_llm = AzureChatOpenAI(
    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_ROUTER, 
    api_version=settings.AZURE_OPENAI_API_VERSION_ROUTER,
    azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT_ROUTER),
    api_key=settings.AZURE_OPENAI_API_KEY_ROUTER,
    temperature=0.2,
    max_tokens=800
)

FORBIDDEN_KEYWORDS = {
    'UPDATE', 'DELETE', 'INSERT', 'DROP', 'ALTER', 'TRUNCATE', 
    'CREATE', 'GRANT', 'REVOKE', 'EXEC', 'UNION', 'pg_sleep'
}
FORBIDDEN_TABLES = {'users', 'chats', 'embeddings', 'conversation_history'}

CONTEXTUALIZE_SYSTEM = """
Given a chat history and the latest user question which might reference context in the chat history, 
formulate a standalone question that can be understood without the chat history.
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.

Examples:
History: [User: Who is the most active?] [AI: John is.]
User: How many messages did HE send?
Standalone: How many messages did John send?

History: [User: What did we say about sushi?]
User: Summarize it.
Standalone: Summarize the discussion about sushi.
"""

contextualize_chain = (
    ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_SYSTEM),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    | router_llm
    | StrOutputParser()
)


DB_SCHEMA_CONTEXT = """
PostgreSQL Schema (Analytics Scope):

1. Table: participants (alias: p)
   - id: integer (PK)
   - chat_id: integer (FK)
   - name: varchar

2. Table: messages (alias: m)
   - id: integer (PK)
   - chat_id: integer (FK)
   - participant_id: integer (FK -> participants.id)
   - timestamp: datetime
   - content: text
   - word_count: integer
   - emojis_count: integer
   - links_count: integer
   - is_question: boolean
   - is_media: boolean

3. Table: message_sentiments (alias: s)
   - id: integer (PK)
   - message_id: integer (FK -> messages.id)
   - overall_label: varchar (Values: 'positive', 'negative', 'neutral')
   - overall_label_score: float
   - score_positive: float
   - score_negative: float
   - score_neutral: float

Relationships:
- m.participant_id = p.id
- s.message_id = m.id
"""

ROUTER_SYSTEM_PROMPT = """
You are the Central Dispatch of the SentimentScope Analysis Engine.
Analyze the User Query and the System Metadata to route the request to the correct worker.

### SYSTEM METADATA
- SQL Data Ready: {sql_ready} (Boolean)
- Embeddings Ready: {embeddings_ready} (Boolean)
- Dashboard Ready: {dashboard_ready} (Boolean)

### DASHBOARD CONTENTS (Pre-calculated Metrics)
If the user asks for these specific metrics, route to `analytics_dashboard`:
{dashboard_capabilities}

### INTENT DEFINITIONS
1. **analytics_dashboard**: 
   - Use for **Global Totals** (e.g., "How many participants?", "Total messages?").
   - Use for **Trends** (e.g., "Activity over time", "Who talks the most?", "Peak activity days").
   - Use for **Sentiment Summaries** (e.g., "Is the chat positive?", "Sentiment trends").

2. **sql_agent**: 
   - Use for **Filtered Counts** (e.g., "How many times did X say Y?", "Messages sent after 10 PM").
   - Use for **Specific Comparisons** not found in the dashboard.
   - *Requires SQL Data Ready*.

3. **vector_search**: 
   - Content retrieval (What did we say about X? Find messages about Y).
   - *Requires Embeddings Ready*.

4. **hybrid_query**: 
   - Complex questions needing BOTH stats and specific message context.
   - Example: "Who sent the most messages and what did they say?"
   - *Requires SQL Data Ready*.

5. **general**: Greetings, help requests, system questions.

6. **system_not_ready**: 
   - Trigger ONLY if the user asks for a specific data source that is `False`.

### OUTPUT FORMAT
Return strictly a JSON object, no markdown: {{"intent": "intent_name"}}

### USER QUERY
{question}
"""

SQL_GENERATION_PROMPT = f"""
You are a PostgreSQL Data Engineer. Generate a safe, read-only SQL query for the user question.

### SCHEMA
{DB_SCHEMA_CONTEXT}

### CRITICAL RULES
1. **Security:** ALWAYS filter by `chat_id = :chat_id`.
2. **Safety:** SELECT only. No UPDATE/DELETE/DROP.
3. **Limit:** AUTOMATICALLY append `LIMIT 10` to any query returning raw rows (not needed for COUNT/AVG).
4. **Scope:** - If the question asks for "meaning", "summary", or "topic" (uncountable qualitative data), set `valid_sql` to false.
   - If the question is about numbers, dates, counts, or specific rankings, set `valid_sql` to true.
5. **Text Matching:** ALWAYS use `ILIKE` for name or content comparisons to ensure case-insensitivity (e.g., `name ILIKE 'john'`).

### OUTPUT FORMAT (JSON)
{{{{
    "valid_sql": boolean,
    "sql": "SELECT ...", 
    "reasoning": "Brief explanation of why SQL applies or not."
}}}}

### USER QUERY
{{question}}
"""

MASTER_SYSTEM_PROMPT = """
### IDENTITY & PURPOSE
You are SentimentScope, an intelligent AI analyst for WhatsApp chat logs. 
Your goal is to provide insightful, data-driven answers about the user's conversation history.
Your tone is helpful, professional, and conversational. You are NOT a robot debugging a script.

### CURRENT SYSTEM STATE
[CRITICAL: Read this first to understand what data you have access to]
- Embedding Status: {embedding_status} (e.g., "completed", "processing", "pending", "failed")
- Data Available: {data_sources_available} (List)

### INPUT DATA STREAMS
You have been provided with the following data streams. If a stream is "None" or "Empty", ignore it.

1. **QUANTITATIVE DATA (SQL/Dashboard Stats):**
   *Use this for exact counts, dates, and hard numbers.*
   {structured_data}

2. **QUALITATIVE CONTEXT (Vector Search/RAG):**
   *Use this for "who said what", specific topics, tone, and context.*
   {rag_context}

### RESPONSE GUIDELINES

#### 1. HANDLING "NO DATA" SCENARIOS (Anti-Hallucination)
- If `rag_context` is empty/null: 
  - **Check `Embedding Status`**:
    - If "processing" or "pending": "I am currently processing the chat content for search. I can't look up specific topics yet, but I might be able to answer statistical questions."
    - If "completed" or "failed": "I couldn't find any specific messages matching that topic in your chat history."
  - Do **NOT** invent a citation like `[source_table:1]`.

#### 2. CITATION PROTOCOL (Strict)
- **Rule:** You must cite the specific message used for context.
- **Format:** Use `[source_table:id]` (e.g., `[messages:153]`, `[segments_sender:45]`).
- **Restriction:** - NEVER cite the SQL data, Dashboard, or System State. 
  - Only cite when you are quoting or summarizing a specific chunk from `rag_context`.

#### 3. SYNTHESIS (Hybrid Logic)
- If you have BOTH `structured_data` and `rag_context`:
  - Start with the hard facts (SQL/Stats).
  - Follow up with the context/examples (RAG).

### USER QUERY
{question}
"""

# Regex Trap
GREETING_PATTERNS = [
    r"^(hi|hello|hey|sup|greetings)\b",
    r"^who are you",
    r"^what can you do"
]

def _check_fast_trap(query: str) -> str | None:
    q = query.strip().lower()
    for p in GREETING_PATTERNS:
        if re.search(p, q):
            return "Hello! I am SentimentScope's intelligent AI analyst for your chat history. Ask me about statistics (e.g., 'Who talks the most?') or search for specific topics (e.g., 'What did we say about pizza?')."
    return None


def _get_dashboard_capabilities(analytics_json: Dict[str, Any] | None) -> str:
    if not analytics_json:
        return "caps:none"

    caps = ["caps:"]

    gen = analytics_json.get("general_dashboard")
    if gen:
        if gen.get("participants") is not None: caps.append("G:p")
        if gen.get("participantCount") is not None: caps.append("G:c")
        if gen.get("kpiMetrics"): caps.append("G:kpi")
        if gen.get("messagesOverTime"): caps.append("G:ts")
        if gen.get("activityByDay"): caps.append("G:day")
        if gen.get("hourlyActivity"): caps.append("G:hour")
        if gen.get("contribution"): caps.append("G:ctr")
        if gen.get("activity"): caps.append("G:radar")
        if gen.get("timeline"): caps.append("G:tl")

    sent = analytics_json.get("sentiment_dashboard")
    if sent:
        if sent.get("kpiData"): caps.append("S:kpi")
        if sent.get("trendData"): caps.append("S:trend")
        if sent.get("breakdownData"): caps.append("S:brk")
        if sent.get("dayData"): caps.append("S:day")
        if sent.get("hourData"): caps.append("S:hour")
        if sent.get("highlightsData"): caps.append("S:hl")

    return " ".join(caps) if len(caps) > 1 else "caps:none"




async def validate_sql_safety(sql: str) -> str:
    """
    Static analysis of SQL string to prevent destructive or unauthorized queries.
    Returns the cleaned SQL if safe, raises ValueError if unsafe.
    """
    normalized_sql = sql.strip().strip(';').replace('\n', ' ')
    upper_sql = normalized_sql.upper()

    # Destructive Command Check
    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(r'\b' + keyword + r'\b', upper_sql):
            raise ValueError(f"Security Alert: Query contains forbidden keyword '{keyword}'")
    # Forbidden Table Check
    for table in FORBIDDEN_TABLES:
        if re.search(r'\b' + table + r'\b', sql.lower()):
             raise ValueError(f"Security Alert: Access to restricted table '{table}' is denied.")

    # Scope Enforcement
    if ":chat_id" not in sql:
        raise ValueError("Security Alert: Query failed to bind 'chat_id' parameter.")

    # Performance Governor (Auto-Limit)
    if "LIMIT" not in upper_sql:
        normalized_sql += " LIMIT 20"
        log.warning("[SQL Agent] LLM forgot LIMIT. Injected 'LIMIT 20'.")
    else:
        pass

    return normalized_sql

async def generate_and_execute_sql(
    query: str, chat_id: int, db: AsyncSession
) -> str:
    """
    Agents sub-routine: Generates SQL, validates safety, executes, and returns raw results.
    """
    try:
        prompt = ChatPromptTemplate.from_template(SQL_GENERATION_PROMPT)
        chain = prompt | main_llm | StrOutputParser()
        raw_response = await chain.ainvoke({"question": query})
        
        cleaned_json = raw_response.strip().replace('```json', '').replace('```', '')
        
        try:
            parsed = json.loads(cleaned_json)
        except json.JSONDecodeError:
            log.warning(f"[SQL Agent] Failed to parse JSON: {raw_response}")
            return "REFUSE"

        if not parsed.get("valid_sql", False):
            return "REFUSE"

        sql_query = parsed.get("sql", "")
        log.info(f"[SQL Agent] Generated: {sql_query}")

        safe_sql = await validate_sql_safety(sql_query)

        stmt = text(safe_sql)
        result = await db.execute(stmt, {"chat_id": chat_id})
        rows = result.fetchall()
        
        if not rows:
            return "No data found matching the criteria."
        
        return str(rows[:20])

    except ProgrammingError as e:
        await db.rollback()
        
        error_details = str(e.orig) if hasattr(e, 'orig') else str(e)
        log.warning(f"[SQL Agent] Bad SQL generated. Query: {safe_sql} | Error: {error_details}")
        
        return "REFUSE"
    except OperationalError as e:
        await db.rollback()
        log.error(f"[SQL Agent] DB Operational Error (Connection/Timeout): {e}")
        return "REFUSE"

    except DataError as e:
        await db.rollback()
        log.warning(f"[SQL Agent] Data processing error: {e}")
        return "REFUSE"

    except SQLAlchemyError as e:
        # Catch-all for other DB errors
        await db.rollback()
        log.error(f"[SQL Agent] Generic DB Error: {e}")
        return "REFUSE"

    except Exception as e:
        # Non-DB errors 
        try:
            await db.rollback()
        except:
            pass 
            
        log.critical(f"[SQL Agent] Critical Python Error: {e}", exc_info=True)
        return "REFUSE"

async def run_vector_search(query: str, chat_id: int):
    docs = await retriever.aget_relevant_documents(query, chat_id)
    
    if not docs:
        
        return [], [], None 
        
    sources_list = [
        {"source_table": s.source_table, "source_id": s.source_id, "distance": s.distance, "text": s.text} 
        for s in docs
    ]
    
    context_text = "\n\n".join([f"[{d.source_table}:{d.source_id}] {d.text}" for d in docs])
    
    return docs, sources_list, context_text



async def route_and_process(
    query: str, 
    analytics_json: Dict[str, Any] | None, 
    chat_id: int,
    db: AsyncSession,
    chat_history: List[BaseMessage]
) -> AsyncGenerator[str, None]:
    """
    The Main Entry Point. Yields chunks of the answer (SSE format).
    Final yield is the structured JSON with metadata (RagQueryResponse).
    """
    fast_response = _check_fast_trap(query)
    if fast_response:
        yield f"data: {json.dumps(fast_response)}\n\n"
    
        await _save_turn(db, chat_id, query, fast_response, [])

        final_payload = {"answer": fast_response, "route": "TIER_1_FAST", "sources": []}
        yield f"data: {json.dumps(final_payload)}\n\n"
        return
    
    standalone_question = query
    if chat_history:
        try:
            standalone_question = await contextualize_chain.ainvoke({
                "chat_history": chat_history,
                "question": query
            })
        except Exception as e:
            log.error(f"Contextualization failed: {e}")
    

    status_str = await crud.get_chat_embedding_status(db, chat_id)
    
    is_sql_ready = (status_str is not None)
    has_dashboard = analytics_json is not None
    is_embeddings_ready = (status_str == EmbeddingStatusEnum.completed.value)
    
    dashboard_caps = _get_dashboard_capabilities(analytics_json)

    intent = "vector_search"
    try:
        router_input = {
            "question": standalone_question,
            "sql_ready": is_sql_ready,
            "embeddings_ready": is_embeddings_ready,
            "dashboard_ready": has_dashboard,
            "dashboard_capabilities": dashboard_caps # Inject guidance
        }
        
        raw_class = await (
            ChatPromptTemplate.from_template(ROUTER_SYSTEM_PROMPT) | router_llm | StrOutputParser()
        ).ainvoke(router_input)
        
        # Strip Markdown to fix the "Router failed: Expecting value" error
        cleaned_router = raw_class.strip().replace('```json', '').replace('```', '')
        parsed_intent = json.loads(cleaned_router)
        intent = parsed_intent.get("intent", "vector_search")


    except Exception as e:
        log.warning(f"Router failed: {e}")

        if is_embeddings_ready:
            intent = "vector_search"
        elif is_sql_ready:
            intent = "sql_agent"
        else:
            intent = "system_not_ready"

    log.info(f"Router Decision: {intent} (SQL: {is_sql_ready}, Embeddings: {is_embeddings_ready})")

    answer_accum = ""
    sources_list = []
    
    structured_data = "None"
    rag_context = None
    
    
    try:
        if intent == "system_not_ready":
             pass 

        if intent in ["analytics_dashboard", "hybrid_query"] and has_dashboard:
            structured_data = f"DASHBOARD STATS:\n{serialize_analytics(analytics_json)}"
            
        if intent in ["sql_agent", "hybrid_query"] and is_sql_ready:
            sql_res = await generate_and_execute_sql(standalone_question, chat_id, db)
            if sql_res != "REFUSE":
                if structured_data == "None":
                    structured_data = f"SQL RESULT: {sql_res}"
                else:
                    structured_data += f"\n\nSQL RESULT: {sql_res}"

        if intent in ["vector_search", "hybrid_query", "sql_agent", "general"] and is_embeddings_ready:
            _, sources_list, rag_context = await run_vector_search(standalone_question, chat_id)
            
        prompt_inputs = {
            "has_file": is_sql_ready, 
            "embedding_status": status_str if status_str else "unknown", 
            "data_sources_available": [k for k, v in [("SQL/Stats", structured_data != "None"), ("RAG", rag_context is not None)] if v],
            "structured_data": structured_data,
            "rag_context": rag_context if rag_context else "", 
            "question": standalone_question
        }
        
        chain = (
            ChatPromptTemplate.from_template(MASTER_SYSTEM_PROMPT) 
            | main_llm 
            | StrOutputParser()
        )

        async for chunk in chain.astream(prompt_inputs):
            answer_accum += chunk
            yield f"data: {json.dumps(chunk)}\n\n"

        if answer_accum:
            await _save_turn(db, chat_id, query, answer_accum, sources_list)
            
        final_resp = {
            "answer": answer_accum,
            "route": intent.upper(),
            "sources": sources_list 
        }
        yield f"data: {json.dumps(final_resp)}\n\n"

    except Exception as e:
        log.error(f"Routing Error: {e}", exc_info=True)
        err_msg = "I encountered an error processing your request."
        yield f"data: {json.dumps(err_msg)}\n\n"



async def _save_turn(
    db: AsyncSession,
    chat_id: int,
    q: str,
    a: str,
    sources: List[Dict[str, Any]]
):
    """"Helper to persist conversation turn safely."""
    try: 
        await crud.add_conversation_turn(
            db,
            chat_id=chat_id,
            user_q=q,
            ai_a=a,
            sources=sources
        )
    except Exception as e:
        log.error(f"Failed to save conversation turn for chat {chat_id}: {e}")