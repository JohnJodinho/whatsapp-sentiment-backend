# src/app/services/router_service.py

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
from src.app.schemas import RagSource
from src.app.crud import get_chat_embedding_status
from src.app.schemas import EmbeddingStatusEnum
from src.app import crud
log = logging.getLogger(__name__)

# gpt-4o-mini
router_llm = AzureChatOpenAI(
    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_ROUTER,
    api_version=settings.AZURE_OPENAI_API_VERSION_ROUTER,
    azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT_ROUTER),
    api_key=settings.AZURE_OPENAI_API_KEY_ROUTER,
    temperature=0.0,
    max_tokens=200
)

# Main LLM for answer generation (e.g., gpt-4o)
main_llm = AzureChatOpenAI(
    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_ROUTER, # Or a separate deployment
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

CLASSIFIER_SYSTEM = """
You are the Brain of a WhatsApp Chat Analyzer. 
Classify the user query into exactly one JSON object: {{"intent": "..."}}.

INTENTS:
1. "analytics_dashboard": Questions answerable by standard summary stats (e.g., "Who talks the most?", "Activity by hour?", "Overall sentiment").
2. "sql_agent": Specific/Deep statistical questions NOT in a standard dashboard. 
   Examples: "How many questions did John ask?", "Who has the highest negative sentiment score?", "Count messages with > 100 words".
3. "vector_search": Content questions. "What did we say about X?", "Find the message about Y".
4. "hybrid_dashboard": High-level stats + Content context. 
   (e.g., "Who is the most active user and what do they usually talk about?", "Why is the overall sentiment negative?").
5. "hybrid_sql": Specific counts/stats + Content context. 
   (e.g., "How many messages did John send and what is his communication style?", "Count the messages about 'sushi' and summarize them").
6. "general": Greetings, identity questions.

User Question: {question}

Output JSON:
"""

HYBRID_RESPONSE_SYSTEM = """
You are a Smart Analyst. 
I have provided two sources of information to answer the user's question:
1. DASHBOARD STATS: High-level metrics (activity, sentiment counts).
2. MESSAGE CONTEXT (RAG): Semantic search results from the chat history.

INSTRUCTIONS:
- Synthesize an answer that combines the hard numbers from the Stats with the qualitative details from the Context.
- If the Stats contradict the Context, prioritize the Stats for numbers and Context for meaning.

CITATION RULES:
1. **ONLY** cite the specific RAG documents used for context. Format: [source_table:id] (e.g., [messages:123], [segments:456]).
2. **NEVER** cite the "DASHBOARD", "STATS", or "JSON" as a source. The stats are facts, not clickable references.
"""

SQL_GENERATION_SYSTEM = f"""
You are a PostgreSQL Expert specializing in Chat Analytics.
Your goal is to convert User Questions into SQL queries using the schema below.

Context:
{DB_SCHEMA_CONTEXT}

CRITICAL RULES:
1. **Security:** EVERY query MUST filter by `chat_id = :chat_id` in the `WHERE` clause.
2. **Performance:** - ALWAYS use `LIMIT` for queries returning lists (Max 10 rows).
3. **Scope:** - IF the user asks for aggregations (count, average, max, min), dates, sentiment stats, or specific sender activity -> Generate SQL.
   - IF the question is **HYBRID** (asks for stats AND meaning): **IGNORE** the meaning/qualitative part and generate SQL for the stats part.
   - IF the question is **PURELY** about "meaning", "summary", "topic", "context", or "what was said" (with no countable stats) -> Return exactly: "REFUSE".
4. **Format:** Return ONLY the raw SQL string. Do not use Markdown (```). Do not explain.
5. **Joins:** - Use explicit JOINs. 
   - Join `participants` only if filtering by sender name.
   - Join `message_sentiments` only if filtering/averaging sentiment.
6. **Names:** When filtering by `p.name`, use `ILIKE` for case-insensitivity.

Examples:
Q: "How many messages sent by John?"
A: SELECT COUNT(*) FROM messages m JOIN participants p ON m.participant_id = p.id WHERE m.chat_id = :chat_id AND p.name ILIKE '%John%'

Q: "Who is the most negative person?"
A: SELECT p.name, AVG(s.score_negative) as neg_score FROM participants p JOIN messages m ON p.id = m.participant_id JOIN message_sentiments s ON m.id = s.message_id WHERE m.chat_id = :chat_id GROUP BY p.name ORDER BY neg_score DESC LIMIT 1

Q: "How many messages did John send and what is he talking about?"
A: SELECT COUNT(*) FROM messages m JOIN participants p ON m.participant_id = p.id WHERE m.chat_id = :chat_id AND p.name ILIKE '%John%'

Q: "What were they arguing about?"
A: REFUSE

User Question: {{question}}
"""

SQL_RESPONSE_SYSTEM = """
You are a helpful Data Analyst Assistant. 
You have just executed a SQL query to answer the user's question regarding their chat history.

User Question: {question}
SQL Query Result: {sql_result}

Instructions:
1. Synthesize the result into a clear, natural sentence.
2. Round decimal numbers to 2 places (e.g., sentiment scores).
3. If the result is empty or 0, politely inform the user no data matched.
4. Do not mention "SQL", "Database", or "Query ID". Just answer the question directly.

Example Input: [(15, 'positive')]
Example Output: "There were 15 messages classified as positive."
"""

ANALYTICS_SYSTEM = """
You are a Data Analyst. Answer using ONLY the provided JSON dashboard data.
Refuse to hallucinate. If the data isn't there, say so.
Refer to the data naturally (e.g., "The charts show..."). Do not mention "JSON".
"""

RAG_SYSTEM = """
You are a Chat Historian. Answer using the provided CONTEXT snippets.
Cite sources as [source_table:id].

CRITICAL CONSTRAINTS TO EXPLAIN TO USER IF RELEVANT:
- If the user asks for very old messages, remind them we only keep 24 months of history.
- If the user asks for short replies (e.g. "Did I say ok?"), remind them we filter out messages under 4 words.

Context:
{context}
"""

HYBRID_SQL_RESPONSE_SYSTEM = """
You are a Smart Analyst. 
I have provided two sources of information to answer the user's question:
1. PRECISE STATS (SQL): Exact numbers/lists from the database.
2. MESSAGE CONTEXT (RAG): Semantic search results from the chat history.

INSTRUCTIONS:
- Synthesize an answer that combines the exact numbers from SQL with the qualitative details from RAG.
- Use the SQL result for factual counts/lists (e.g., "John sent 183 messages").
- Use the Context for style, tone, and content summary.

CITATION RULES:
1. **ONLY** cite the specific RAG documents used for context. Format: [source_table:id] (e.g., [messages:123], [segments:456]).
2. **NEVER** cite the "SQL", "STATS", or "DATABASE" as a source. The stats are facts, not clickable references.
"""


# # --- 3. Chains ---
# classifier_chain = (
#     ChatPromptTemplate.from_template(CLASSIFIER_SYSTEM)
#     | router_llm
#     | StrOutputParser()
# )

# --- 4. Logic ---

# Tier 1: Regex Trap
GREETING_PATTERNS = [
    r"^(hi|hello|hey|sup|greetings)\b",
    r"^who are you",
    r"^what can you do"
]

def _check_fast_trap(query: str) -> str | None:
    q = query.strip().lower()
    for p in GREETING_PATTERNS:
        if re.search(p, q):
            return "Hello! I analyze your chat history. Ask me about statistics (e.g., 'Who talks the most?') or search for specific topics (e.g., 'What did we say about sushi?')."
    return None

async def validate_sql_safety(sql: str) -> str:
    """
    Static analysis of SQL string to prevent destructive or unauthorized queries.
    Returns the cleaned SQL if safe, raises ValueError if unsafe.
    """
    normalized_sql = sql.strip().strip(';').replace('\n', ' ')
    upper_sql = normalized_sql.upper()

    # 1. Destructive Command Check
    for keyword in FORBIDDEN_KEYWORDS:
        # Regex checks for whole word matches to avoid false positives (e.g., "UPDATE" in "UPDATED_AT")
        if re.search(r'\b' + keyword + r'\b', upper_sql):
            raise ValueError(f"Security Alert: Query contains forbidden keyword '{keyword}'")

    # 2. Table Access Control (Allow-list approach is safer, but Block-list is easier here)
    for table in FORBIDDEN_TABLES:
        if re.search(r'\b' + table + r'\b', sql.lower()):
             raise ValueError(f"Security Alert: Access to restricted table '{table}' is denied.")

    # 3. Scope Enforcement (Must use bound parameter)
    if ":chat_id" not in sql:
        raise ValueError("Security Alert: Query failed to bind 'chat_id' parameter.")

    # 4. Performance Governor (Auto-Limit)
    # If LIMIT is missing, append it. 
    if "LIMIT" not in upper_sql:
        normalized_sql += " LIMIT 20"
        log.warning("[SQL Agent] LLM forgot LIMIT. Injected 'LIMIT 20'.")
    else:
        # Optional: Regex to cap existing limits (e.g., change LIMIT 1000 to LIMIT 50)
        # This is complex to do reliably with Regex alone, keeping it simple for now.
        pass

    return normalized_sql

async def generate_and_execute_sql(
    query: str, chat_id: int, db: AsyncSession
) -> str:
    """
    Agents sub-routine: Generates SQL, validates safety, executes, and returns raw results.
    """
    try:
        # --- 1. Generate SQL ---
        prompt = ChatPromptTemplate.from_template(SQL_GENERATION_SYSTEM)
        chain = prompt | main_llm | StrOutputParser()
        
        # Ainvoke with the user query
        raw_response = await chain.ainvoke({"question": query})
        log.info(f"[SQL Agent] Generated Raw SQL: {raw_response}")
        # Basic string cleanup
        cleaned_response = raw_response.strip()
        cleaned_response = re.sub(r'```sql', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = cleaned_response.replace('```', '').strip()

        # Check for refusal *before* safety checks
        if "REFUSE" in cleaned_response:
            return "REFUSE" # Handle this upstream to trigger Vector Search

        log.info(f"[SQL Agent] Generated Raw: {cleaned_response}")

        # --- 2. High-End Safety Layer ---
        try:
            safe_sql = await validate_sql_safety(cleaned_response)
        except ValueError as e:
            log.warning(f"[SQL Agent] blocked malicious/unsafe query: {e}")
            return "I cannot execute this query due to safety constraints."

        # --- 3. Execute SQL (Safe Read-Only) ---
        stmt = text(safe_sql)
        
        # Execute using the session
        result = await db.execute(stmt, {"chat_id": chat_id})
        
        # Convert to list of dicts for cleaner formatting usually, or just tuples
        rows = result.fetchall()
        
        if not rows:
            return "No data found matching the criteria."
        
        # Convert rows (tuples) to string for the LLM to synthesize
        # Truncate if somehow we got too many rows despite LIMIT
        return str(rows[:20])

    except ProgrammingError as e:
        # Caused by: Syntax errors, missing columns, bad table names (LLM Hallucinations)
        # Action: Critical to rollback.
        await db.rollback()
        
        # Log the specific Postgres error code/message for debugging prompt engineering
        # e.orig often contains the raw driver error (e.g., asyncpg)
        error_details = str(e.orig) if hasattr(e, 'orig') else str(e)
        log.warning(f"[SQL Agent] Bad SQL generated. Query: {safe_sql} | Error: {error_details}")
        
        return "I tried to query the database, but the generated SQL was invalid. (Syntax Error)"

    except OperationalError as e:
        # Caused by: Timeouts, DB connection loss, server restart
        # Action: Rollback and alert infrastructure issues.
        await db.rollback()
        log.error(f"[SQL Agent] DB Operational Error (Connection/Timeout): {e}")
        return "The database is currently unreachable or timed out."

    except DataError as e:
        # Caused by: Division by zero, numeric overflow, invalid input syntax for types
        # Action: Rollback.
        await db.rollback()
        log.warning(f"[SQL Agent] Data processing error: {e}")
        return "The query attempted an invalid mathematical operation or data conversion."

    except SQLAlchemyError as e:
        # Catch-all for other DB errors
        await db.rollback()
        log.error(f"[SQL Agent] Generic DB Error: {e}")
        return "An unexpected database error occurred."

    except Exception as e:
        # Non-DB errors (e.g., Python logic, memory issues)
        # Note: If the session was active, we should still try to rollback just in case.
        try:
            await db.rollback()
        except:
            pass # Session might be closed already
            
        log.critical(f"[SQL Agent] Critical Python Error: {e}", exc_info=True)
        return "I encountered a system error while processing the data."

async def run_vector_search(query: str, chat_id: int):
    docs = await retriever.aget_relevant_documents(query, chat_id)
    sources_list = [
        {"source_table": s.source_table, "source_id": s.source_id, "distance": s.distance, "text": s.text} 
        for s in docs
    ]
    context_text = "\n\n".join([f"[{d.source_table}:{d.source_id}] {d.text}" for d in docs])
    if not context_text:
        context_text = "No relevant messages found (checked last 24 months)."
    
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
    
    # 1. Tier 1: Fast Trap
    fast_response = _check_fast_trap(query)
    if fast_response:
        # A. Stream the content immediately
        yield f"data: {json.dumps(fast_response)}\n\n"
        
        # B. [FIX] Persist the interaction to DB
        await _save_turn(db, chat_id, query, fast_response, [])

        # C. Stream final payload with metadata
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
            log.info(f"Contextualized: '{query}' -> '{standalone_question}'")
        except Exception as e:
            log.error(f"Contextualization failed: {e}")

    intent = "vector_search"  # Default intent
    try:
        raw_class = await (
            ChatPromptTemplate.from_template(CLASSIFIER_SYSTEM) | router_llm | StrOutputParser()
        ).ainvoke({"question": standalone_question })
        parsed = json.loads(raw_class.strip())
        log.info(f"Classifier Raw Output: {raw_class}") 
        intent = parsed.get("intent", "vector_search")
    except Exception as e:
        log.warning(f"Intent classification failed or produced invalid JSON. Defaulting to 'vector_search'. Error: {e}")
        

    log.info(f"Router Decision: {intent}")

    answer_accum = ""
    sources_list = []
    
    # 3. Tier 3: Handlers
    try:
        # --- A. Hybrid Dashboard (Macro Stats + Context) ---
        if intent == "hybrid_dashboard":
            stats_str = serialize_analytics(analytics_json)
            _, sources_list, context_text = await run_vector_search(standalone_question, chat_id)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", HYBRID_RESPONSE_SYSTEM),
                ("human", "STATS:\n{stats_str}\n\nCONTEXT:\n{context_text}\n\nQuestion: {question}")
            ])
            
            chain = prompt | main_llm | StrOutputParser()
            
            async for chunk in chain.astream({
                "stats_str": stats_str,
                "context_text": context_text,
                "question": standalone_question
            }):
                answer_accum += chunk
                yield f"data: {json.dumps(chunk)}\n\n"

        # --- B. Hybrid SQL (Micro Stats + Context) [NEW] ---
        elif intent == "hybrid_sql":
            # 1. Run SQL (Micro Stats)
            sql_result = await generate_and_execute_sql(standalone_question, chat_id, db)
            log.info(f"[Hybrid SQL] SQL Result: {sql_result}")
            # 2. Run Vector Search (Context)
            _, sources_list, context_text = await run_vector_search(standalone_question, chat_id)
            
            # 3. Synthesize
            # If SQL failed/refused, we still try to answer best effort, or fallback to standard RAG
            if sql_result == "REFUSE":
                 # Fallback to standard RAG if SQL refuses (unlikely given classification, but safe)
                 prompt = ChatPromptTemplate.from_messages([
                    ("system", RAG_SYSTEM), 
                    ("human", "Question: {question}")
                ])
                 chain = prompt | main_llm | StrOutputParser()
                 async for chunk in chain.astream({"context": context_text, "question": standalone_question}):
                    answer_accum += chunk
                    yield f"data: {json.dumps(chunk)}\n\n"
            else:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", HYBRID_SQL_RESPONSE_SYSTEM),
                    ("human", "SQL RESULT: {sql_result}\n\nCONTEXT:\n{context_text}\n\nQuestion: {question}")
                ])
                
                chain = prompt | main_llm | StrOutputParser()
                
                async for chunk in chain.astream({
                    "sql_result": sql_result,
                    "context_text": context_text,
                    "question": standalone_question
                }):
                    answer_accum += chunk
                    yield f"data: {json.dumps(chunk)}\n\n"

        elif intent == "sql_agent":
            # 1. Generate & Run SQL
            sql_result = await generate_and_execute_sql(standalone_question, chat_id, db)
            log.info(f"[SQL Agent] SQL Result: {sql_result}")
            # 2. Synthesize Answer
            if sql_result == "REFUSE":
                log.info("[SQL Agent] Refused query. Falling back to Vector Search.")
                intent = "vector_search" # Update intent for logging/metadata
                
                # Fallback logic
                docs, sources_list, context_text = await run_vector_search(standalone_question, chat_id)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", RAG_SYSTEM),
                    ("human", f"Question: {standalone_question}\n\nContext:\n{context_text}")
                ])
                chain = prompt | main_llm | StrOutputParser()
                async for chunk in chain.astream({
                    "context": context_text,
                    "question": standalone_question
                }):
                    answer_accum += chunk
                    yield f"data: {json.dumps(chunk)}\n\n"
            else:
                # Normal SQL Response
                prompt = ChatPromptTemplate.from_template(SQL_RESPONSE_SYSTEM)
                chain = prompt | main_llm | StrOutputParser()
                
                async for chunk in chain.astream({"sql_result": sql_result, "question": standalone_question}):
                    answer_accum += chunk
                    yield f"data: {json.dumps(chunk)}\n\n"

        elif intent == "analytics_dashboard":
            if not analytics_json:
                full_resp = "I need dashboard stats to answer that, but they aren't loaded."
                yield f"data: {json.dumps(full_resp)}\n\n"
                final_resp = {"answer": full_resp, "route": intent.upper(), "sources": []}
                yield f"data: {json.dumps(final_resp)}\n\n"
                return

            data_str = serialize_analytics(analytics_json)
            prompt = ChatPromptTemplate.from_messages([
                ("system", ANALYTICS_SYSTEM),
                ("human", "Data: {data_str}\n\nQuestion: {question}"),
            ])
            
            chain = prompt | main_llm | StrOutputParser()
            async for chunk in chain.astream({
                "data_str": data_str,
                "question": standalone_question
            }):
                answer_accum += chunk
                yield f"data: {json.dumps(chunk)}\n\n"

        elif intent == "vector_search":
            _, sources_list, context_text = await run_vector_search(standalone_question, chat_id)

            prompt = ChatPromptTemplate.from_messages([
                ("system", RAG_SYSTEM), # RAG_SYSTEM has {context} placeholder
                ("human", "Question: {question}")
            ])
            
            chain = prompt | main_llm | StrOutputParser()
            async for chunk in chain.astream({
                "context": context_text,
                "question": standalone_question
            }):
                answer_accum += chunk
                yield f"data: {json.dumps(chunk)}\n\n"
        else:
            chain = (
                ChatPromptTemplate.from_template("Answer politely: {question}") 
                | main_llm 
                | StrOutputParser()
            )
            async for chunk in chain.astream({"question": standalone_question}):
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