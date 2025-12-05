import logging
import re
import json
import asyncio
import tiktoken
from typing import Dict, Any, List, AsyncGenerator, Union
import random
from openai import RateLimitError

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


CTX_WINDOW_MAIN_4O = 32000     
CTX_WINDOW_MAIN_4O_MINI = 8000

router_llm = AzureChatOpenAI(
    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_ROUTER,
    api_version=settings.AZURE_OPENAI_API_VERSION_ROUTER,
    azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT_ROUTER),
    api_key=settings.AZURE_OPENAI_API_KEY_ROUTER,
    temperature=0.0,
    max_tokens=200
)

main_llm_primary = AzureChatOpenAI(
    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_MAIN_4O,
    api_version=settings.AZURE_OPENAI_API_VERSION_MAIN_4O,
    azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT_MAIN_4O),
    api_key=settings.AZURE_OPENAI_API_KEY_MAIN_4O,
    temperature=0.2,
    streaming=True 
)

main_llm_fallback = AzureChatOpenAI(
    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_MAIN_4O_MINI,
    api_version=settings.AZURE_OPENAI_API_VERSION_MAIN_4O_MINI,
    azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT_MAIN_4O_MINI),
    api_key=settings.AZURE_OPENAI_API_KEY_MAIN_4O_MINI,
    temperature=0.2,
    streaming=True
)
context_llm_fallback = AzureChatOpenAI(
    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_CONTEXT_4O,
    api_version=settings.AZURE_OPENAI_API_VERSION_CONTEXT_4O,
    azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT_CONTEXT_4O),
    api_key=settings.AZURE_OPENAI_API_KEY_CONTEXT_4O,
    temperature=0.2,
    max_tokens=500
)

# Fallback Context LLM (GPT-4 / Smaller model)
context_llm = AzureChatOpenAI(
    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_CONTEXT_4,
    api_version=settings.AZURE_OPENAI_API_VERSION_CONTEXT_4,
    azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT_CONTEXT_4),
    api_key=settings.AZURE_OPENAI_API_KEY_CONTEXT_4,
    temperature=0.2,
    max_tokens=500
)

FORBIDDEN_KEYWORDS = {
    'UPDATE', 'DELETE', 'INSERT', 'DROP', 'ALTER', 'TRUNCATE', 
    'CREATE', 'GRANT', 'REVOKE', 'EXEC', 'pg_sleep'
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
    "reasoning": "A concise label for the data being retrieved (e.g. 'Count of messages from John')."
}}}}

### USER QUERY
{{question}}
"""

MASTER_SYSTEM_PROMPT = """
### IDENTITY
You are SentimentScope, a warm and insightful AI companion analyzing a WhatsApp chat.
Your goal is to answer the user's questions directly and naturally.

### [SESSION_FLOW]
(Use this recent context to maintain tone and continuity, but DO NOT refer to it explicitly.)
{chat_context}

### KNOWLEDGE BASE
[STATISTICS & FACTS]
{structured_data}

[CONVERSATION EXCERPTS]
{rag_context}

### STRICT RESPONSE RULES (ANTI-LEAK)
1. **Be Direct & Conversational:** - Never explain *how* you found the answer. 
   - BAD: "Based on the session flow..." or "Looking at the SQL results..."
   - GOOD: "John sent 182 messages."

2. **No Technical Jargon:** 
   - NEVER use words/phrases like: "SQL", "RAG", "Vector Search", "Embeddings", "Database", "Query", "Tuple", "[CONVERSATION EXCERPTS]", "session flow", "knowledge base", "[EXCERPTS]", "[STATISTICS]".
   - If the data is missing, say: "I'm not sure," or "I don't see that in the history." Do NOT say: "The SQL result is empty."

3. **Handle Discrepancies:**
   - If [STATISTICS] says "0 messages" but [EXCERPTS] shows John talking, trust the [EXCERPTS] and say: "I see John chatting, but I don't have his exact message count right now."

4. **Citations:**
   - Only cite specific quotes from [CONVERSATION EXCERPTS] using `[source_table:id]`.
   - Do not cite statistics.

### USER QUESTION
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


async def generate_standalone_question(user_question: str, chat_history: List[BaseMessage]) -> str:
    """
    Generates a standalone question from user input and chat history
    using primary and fallback context LLMs, with intelligent retries.
    """
    # 1. Format History for the Prompt
    history_str = ""
    for msg in chat_history[-6:]:  # Keep context window manageable
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_str += f"{role}: {msg.content}\n"

    # 2. Compose Prompt
    prompt_content = (
        f"Given the chat history:\n{history_str}\n"
        f"Rewrite the user's question as a standalone question:\n{user_question}"
    )
    
    messages = [HumanMessage(content=prompt_content)]

    # 3. Execution with Backoff Strategy
    retries = 0
    max_retries = 5
    base_delay = 0.5 

    while retries <= max_retries:
        try:
            # Attempt 1: Primary LLM
            response = await context_llm.ainvoke(messages)
            return response.content.strip()
        
        except RateLimitError:
            log.warning(f"Primary Context LLM Rate Limited. Attempting Fallback... (Retry {retries})")
            try:
                # Attempt 2: Fallback LLM
                response = await context_llm_fallback.ainvoke(messages)
                return response.content.strip()
            
            except RateLimitError:
                # Both failed: Exponential Backoff
                retries += 1
                if retries > max_retries:
                    break
                
                sleep_time = base_delay * (2 ** retries) + random.uniform(0, 0.2)
                log.warning(f"Rate limit hit on both LLMs. Retrying in {sleep_time:.2f}s...")
                await asyncio.sleep(sleep_time)
        
        except Exception as e:
            # Catch non-rate-limit errors (e.g. Context Length exceeded) and log
            log.error(f"Contextualization Error: {e}")
            return user_question # Fail safe: return original question

    # If all retries exhausted, return original question to allow flow to continue
    log.error("Failed to contextualize question after maximum retries.")
    return user_question

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


def _calculate_dynamic_config(prompt_text: str):
    """
    Determines which model to use and the safe max_tokens value based on input size.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    prompt_tokens = len(enc.encode(prompt_text))
    
    # Logic: Prefer 4o, Fallback to Mini if context is too tight (or explicitly switched later)
    # Note: You can adjust this threshold. Here we use 4o unless it's getting full, 
    # but the retry logic below handles the cost/rate-limit fallback.
    
    # Calculate available tokens for 4o
    if prompt_tokens < (CTX_WINDOW_MAIN_4O - 1000):
        # Default to Primary
        llm = main_llm_primary
        safe_max_tokens = CTX_WINDOW_MAIN_4O - prompt_tokens - 500
    else:
        # Fallback immediately if prompt is massive
        llm = main_llm_fallback
        safe_max_tokens = min(CTX_WINDOW_MAIN_4O_MINI - prompt_tokens - 200, 4096)

    # Cap max_tokens to avoid crazy outputs
    if safe_max_tokens > 4096: 
        safe_max_tokens = 4096
        
    return llm, safe_max_tokens, prompt_tokens

async def execute_resilient_main_llm(
    prompt_messages: List[BaseMessage], 
    stream: bool = False
) -> Union[str, AsyncGenerator[str, None]]:
    """
    Executes LLM call with:
    1. Dynamic Token Calculation
    2. Primary -> Fallback Model Switching
    3. Exponential Backoff for Rate Limits
    """
    
    # Convert messages to string for token counting
    msg_texts = []
    for m in prompt_messages:
        if hasattr(m, 'content'):
            msg_texts.append(m.content)
        elif isinstance(m, tuple) and len(m) >= 2:
            msg_texts.append(str(m[1]))
        elif isinstance(m, str):
            msg_texts.append(m)
        else:
            msg_texts.append(str(m))
    
    full_text = " ".join(msg_texts)
    
    # Initial Config
    current_llm, dynamic_max, _ = _calculate_dynamic_config(full_text)
    
    retries = 0
    max_retries = 5
    base_delay = 0.5

    while retries <= max_retries:
        try:
            # Apply Dynamic Max Tokens (Create a copy/runtime config if needed, 
            # but binding directly works for single-request scope in standard usage)
            # Langchain invoke accepts `max_tokens` in bind or call options usually, 
            # but modifying the object property is the direct way per your pseudo-code.
            current_llm.max_tokens = dynamic_max

            if stream:
                # For Streaming (Final Answer) - return the generator immediately
                # Note: If rate limit hits *during* stream, it raises error in the consumption loop.
                # This try/catch primarily protects the connection establishment.
                return current_llm.astream(prompt_messages)
            else:
                # For Standard (SQL)
                response = await current_llm.ainvoke(prompt_messages)
                return response.content

        except RateLimitError:
            log.warning(f"Rate Limit Hit on {current_llm.azure_deployment}. Retries: {retries}")
            
            # Switch Strategy: If on Primary, downgrade to Fallback
            if current_llm == main_llm_primary:
                log.info("Switching to Fallback Model (Mini)...")
                current_llm = main_llm_fallback
                # Recalculate max tokens for the smaller model
                _, _, p_tokens = _calculate_dynamic_config(full_text)
                dynamic_max = min(CTX_WINDOW_MAIN_4O_MINI - p_tokens - 200, 4096)
            else:
                # Already on fallback, just backoff
                retries += 1
                sleep_time = base_delay * (2 ** retries) + random.uniform(0, 0.2)
                await asyncio.sleep(sleep_time)
        
        except Exception as e:
            log.error(f"LLM Execution Error: {e}")
            raise e # Non-rate-limit errors should bubble up

    raise Exception("Main LLM failed after maximum retries.")


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

async def generate_and_execute_sql(query: str, chat_id: int, db: AsyncSession) -> str:
    safe_sql = "N/A"
    try:
        prompt_template = ChatPromptTemplate.from_template(SQL_GENERATION_PROMPT)
        prompt_messages = prompt_template.format_messages(question=query)
        raw_response = await execute_resilient_main_llm(prompt_messages, stream=False)
        cleaned_json = raw_response.strip().replace('```json', '').replace('```', '')
        try:
            parsed = json.loads(cleaned_json)
        except json.JSONDecodeError:
            log.warning(f"[SQL Agent] Failed to parse JSON: {raw_response}")
            return "REFUSE"

        if not parsed.get("valid_sql", False):
            return "REFUSE"

        sql_query = parsed.get("sql", "")
        # Extract Reasoning
        reasoning = parsed.get("reasoning", "Database Query Result")
        
        log.info(f"[SQL Agent] Generated: {sql_query} | Reasoning: {reasoning}")

        safe_sql = await validate_sql_safety(sql_query)

        stmt = text(safe_sql)
        result = await db.execute(stmt, {"chat_id": chat_id})
        rows = result.fetchall()
        
        if not rows:
            return f"[QUERY GOAL: {reasoning}] RESULT: No records found."
        
        # Unwrap Logic
        formatted_data = str(rows[:20])
        if len(rows) == 1 and len(rows[0]) == 1:
            formatted_data = str(rows[0][0])
        elif len(rows) > 0 and len(rows[0]) == 1:
            formatted_data = ", ".join([str(r[0]) for r in rows[:20]])

        return f"[QUERY GOAL: {reasoning}] RESULT: {formatted_data}"

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
        standalone_question = await generate_standalone_question(query, chat_history)
        log.info(f"Contextualized: '{query}' -> '{standalone_question}'")

    # --- SESSION CONTEXT FORMATTING (FIXED) ---
    chat_context_str = ""
    if chat_history:
        recent_history = chat_history[-6:] 
        formatted_turns = []
        for msg in recent_history:
            role = "User" if isinstance(msg, HumanMessage) else "SentimentScope"
            # Truncate content for token safety
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            formatted_turns.append(f"{role}: {content}")
        chat_context_str = "\n".join(formatted_turns)
    else:
        # Pass empty string instead of "No history" to prevent LLM from commenting on it
        chat_context_str = ""
    

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
            raw_stats = serialize_analytics(analytics_json)
            
            if len(raw_stats) > 20000:
                raw_stats = raw_stats[:20000] + "..."
            
            structured_data = f"DASHBOARD STATS:\n{raw_stats}"
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
            "structured_data": structured_data,
            "rag_context": rag_context if rag_context else "", 
            "chat_context": chat_context_str, 
            "question": standalone_question
        }
        
        # Format messages using the Template
        final_messages = ChatPromptTemplate.from_template(MASTER_SYSTEM_PROMPT).format_messages(**prompt_inputs)

        try:
            # EXECUTE RESILIENT STREAM
            stream_generator = await execute_resilient_main_llm(final_messages, stream=True)
            
            # Consume Stream
            async for chunk in stream_generator:
                # Langchain stream yields ChatGenerationChunk, we need the content
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                answer_accum += content
                yield f"data: {json.dumps(content)}\n\n"

        except Exception as e:
            log.error(f"Streaming Error: {e}")
            yield f"data: {json.dumps('System is busy. Please try again.')}\n\n"

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