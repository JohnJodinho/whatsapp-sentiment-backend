# src/app/services/router_service.py

import logging
import re
import json
import asyncio
import random
from typing import Dict, Any, List, AsyncGenerator, Union, Tuple, Optional
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from sqlalchemy.exc import (
    SQLAlchemyError,
    ProgrammingError,
    OperationalError,
    DataError,
)

from src.app.config import settings
from src.app.services.retrieval_service import retriever
from src.app.services.llm_factory import (
    get_router_llm,
    get_context_llm,
    get_main_llm_primary,
    get_main_llm_fallback,
    execute_resilient_llm,
)
from src.app.utils.serializers import serialize_analytics
from src.app.schemas import EmbeddingStatusEnum
from src.app import crud

log = logging.getLogger(__name__)

FORBIDDEN_KEYWORDS = {
    "UPDATE", "DELETE", "INSERT", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "GRANT", "REVOKE", "EXEC", "pg_sleep",
}
FORBIDDEN_TABLES = {"users", "chats", "embeddings", "conversation_history"}

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

FILTER_EXTRACTION_PROMPT = """
You are a Query Analyst. Extract search filters from the user query.

### DATE CONTEXT
Today's Date: {current_date}

### RULES
1. **Senders:** If specific people are mentioned (e.g., "John", "Sarah"), extract them into a list.
2. **Time Ranges:** If time periods are mentioned (e.g., "yesterday", "last week", "in December"), calculate the precise ISO 8601 start and end timestamps.
   - Return a list of lists: [[start_iso, end_iso], ...]
3. If no filters exist, return null/None for those fields.

### OUTPUT FORMAT (Strict JSON)
{{
    "sender_names": ["Name1", "Name2"] or null,
    "time_ranges": [["2023-01-01T00:00:00", "2023-01-01T23:59:59"]] or null
}}

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
1. **Be Direct & Conversational:**
   - Never explain *how* you found the answer.
   - BAD: "Based on the session flow..." or "Looking at the SQL results..."
   - GOOD: "John sent 182 messages."

2. **No Technical Jargon:** 
   - NEVER use words/phrases like: "SQL", "RAG", "Vector Search", "Embeddings", "Database", "Query", "Tuple", "[CONVERSATION EXCERPTS]", "session flow", "knowledge base", "[EXCERPTS]", "[STATISTICS]".
   - If the data is missing, say: "I'm not sure," or "I don't see that in the history." Do NOT say: "The SQL result is empty."

3. **Handle Discrepancies:**
   - If [STATISTICS] says "0 messages" but [EXCERPTS] shows John talking, trust the [EXCERPTS] and say: "I see John chatting, but I don't have his exact message count right now."

4. **Citations (CRITICAL):**
   - **STRICT FORMAT:** You must use the format `[table_name:id]`.
   - **VARIABLE TABLE NAMES:** The `table_name` MUST match the source provided in the excerpts (usually `messages` or `segments_sender`).
   - **FORBIDDEN:** Do NOT add words like "Source:", "Reference:", or "Ref:" inside the brackets.
   - **EXAMPLES:**
      - ✅ CORRECT (Message): "The user asked for help [messages:284619]"
      - ✅ CORRECT (Segment): "They discussed the scholarship deadline [segments_sender:4401]"
      - ❌ WRONG (Extra text): "The user asked for help [Source: messages:284619]"
   - **MULTIPLE CITATIONS:** If citing multiple sources at same time, put them in ONE bracket separated by **COMMAS** or leave in separate brackets.
      - ✅ CORRECT: `[messages:291521] [segments_sender:271355]`
      - ✅ CORRECT: `[messages:291521, segments_sender:271355]`
   - Only cite specific quotes from [CONVERSATION EXCERPTS]. Do not cite statistics.

### USER QUESTION
{question}
"""

GREETING_PATTERNS = [
    r"^(hi|hello|hey|sup|greetings)\b",
    r"^who are you",
    r"^what can you do",
]


def _check_fast_trap(query: str) -> Optional[str]:
    q = query.strip().lower()
    for p in GREETING_PATTERNS:
        if re.search(p, q):
            return "Hello! I am SentimentScope's intelligent AI analyst for your chat history. Ask me about statistics (e.g., 'Who talks the most?') or search for specific topics (e.g., 'What did we say about pizza?')."
    return None


async def generate_standalone_question(user_question: str, chat_history: List[BaseMessage]) -> str:
    """Generates a standalone question from user input and chat history."""
    if not chat_history:
        return user_question

    history_str = ""
    for msg in chat_history[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_str += f"{role}: {msg.content}\n"

    prompt_content = (
        f"Given the chat history:\n{history_str}\n"
        f"Rewrite the user's question as a standalone question:\n{user_question}"
    )

    try:
        context_llm = get_context_llm()
        response = await context_llm.ainvoke([HumanMessage(content=prompt_content)])
        return response.content.strip() if hasattr(response, "content") else str(response).strip()
    except Exception as e:
        log.error("Contextualization error: %s", e)
        return user_question


def _get_dashboard_capabilities(analytics_json: Optional[Dict[str, Any]]) -> str:
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
    normalized_sql = sql.strip().strip(";").replace("\n", " ")
    upper_sql = normalized_sql.upper()

    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(r"\b" + keyword + r"\b", upper_sql):
            raise ValueError(f"Security Alert: Query contains forbidden keyword '{keyword}'")

    for table in FORBIDDEN_TABLES:
        if re.search(r"\b" + table + r"\b", sql.lower()):
            raise ValueError(f"Security Alert: Access to restricted table '{table}' is denied.")

    if ":chat_id" not in sql:
        raise ValueError("Security Alert: Query failed to bind 'chat_id' parameter.")

    if "LIMIT" not in upper_sql:
        normalized_sql += " LIMIT 20"

    return normalized_sql


def _extract_json_payload(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", str(raw_text), flags=re.IGNORECASE).strip()
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()
    else:
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace:last_brace + 1].strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return None


async def generate_and_execute_sql(query: str, chat_id: int, db: AsyncSession) -> str:
    safe_sql = "N/A"
    try:
        prompt_template = ChatPromptTemplate.from_template(SQL_GENERATION_PROMPT)
        prompt_messages = prompt_template.format_messages(question=query)

        router_llm = get_router_llm()
        raw_response = await router_llm.ainvoke(prompt_messages)
        content_str = raw_response.content if hasattr(raw_response, "content") else str(raw_response)

        parsed = _extract_json_payload(content_str)
        if not parsed:
            log.warning("[SQL Agent] Failed to parse JSON: %s", content_str)
            return "REFUSE"

        if not parsed.get("valid_sql", False):
            return "REFUSE"

        sql_query = parsed.get("sql", "")
        reasoning = parsed.get("reasoning", "Database Query Result")
        log.info("[SQL Agent] Generated: %s | Reasoning: %s", sql_query, reasoning)

        safe_sql = await validate_sql_safety(sql_query)
        stmt = text(safe_sql)
        result = await db.execute(stmt, {"chat_id": chat_id})
        rows = result.fetchall()

        if not rows:
            return f"[QUERY GOAL: {reasoning}] RESULT: No records found."

        formatted_data = str(rows[:20])
        if len(rows) == 1 and len(rows[0]) == 1:
            formatted_data = str(rows[0][0])
        elif len(rows) > 0 and len(rows[0]) == 1:
            formatted_data = ", ".join([str(r[0]) for r in rows[:20]])

        return f"[QUERY GOAL: {reasoning}] RESULT: {formatted_data}"

    except Exception as e:
        await db.rollback()
        log.warning("[SQL Agent] DB or Execution error: %s", e)
        return "REFUSE"


async def run_vector_search(
    query: str,
    chat_id: int,
    sender_names: Optional[List[str]] = None,
    time_ranges: Optional[List[Tuple[datetime, datetime]]] = None,
):
    docs = await retriever.aget_relevant_documents(
        query, chat_id, sender_names=sender_names, time_ranges=time_ranges
    )
    if not docs:
        return [], [], None

    sources_list = [
        {
            "source_table": s.source_table,
            "source_id": s.source_id,
            "distance": s.distance,
            "text": s.text,
            "sender_name": s.sender_name,
            "timestamp": s.timestamp,
        }
        for s in docs
    ]

    context_text = "\n\n".join([
        f"[{d.source_table}:{d.source_id}, message_sender: {d.sender_name}, time_sent: {d.timestamp}] {d.text}"
        for d in docs
    ])

    return docs, sources_list, context_text


async def extract_search_filters(query: str) -> Dict[str, Any]:
    current_date = datetime.now().isoformat()
    try:
        prompt_template = ChatPromptTemplate.from_template(FILTER_EXTRACTION_PROMPT)
        prompt_messages = prompt_template.format_messages(
            current_date=current_date, question=query
        )

        router_llm = get_router_llm()
        raw_response = await router_llm.ainvoke(prompt_messages)
        content_str = raw_response.content if hasattr(raw_response, "content") else str(raw_response)

        parsed = _extract_json_payload(content_str)
        if not parsed:
            return {}

        final_filters: Dict[str, Any] = {}
        if parsed.get("sender_names"):
            final_filters["sender_names"] = parsed["sender_names"]

        if parsed.get("time_ranges"):
            processed_ranges = []
            for range_pair in parsed["time_ranges"]:
                if isinstance(range_pair, list) and len(range_pair) == 2:
                    try:
                        s_dt = datetime.fromisoformat(range_pair[0])
                        e_dt = datetime.fromisoformat(range_pair[1])
                        processed_ranges.append((s_dt, e_dt))
                    except ValueError:
                        continue
            if processed_ranges:
                final_filters["time_ranges"] = processed_ranges

        return final_filters
    except Exception as e:
        log.warning("[Filter Extraction] Failed: %s", e)
        return {}


def _sanitize_sources(sources: Optional[List[Any]]) -> List[Dict[str, Any]]:
    if not sources:
        return []

    cleaned = []
    for src in sources:
        if hasattr(src, "model_dump"):
            data = src.model_dump()
        elif hasattr(src, "dict"):
            data = src.dict()
        elif isinstance(src, dict):
            data = src.copy()
        else:
            continue

        if "timestamp" in data and isinstance(data["timestamp"], datetime):
            data["timestamp"] = data["timestamp"].isoformat()

        cleaned.append(data)
    return cleaned


async def _save_turn(db: AsyncSession, chat_id: int, q: str, a: str, sources: List[Dict[str, Any]]):
    try:
        await crud.add_conversation_turn(
            db, chat_id=chat_id, user_q=q, ai_a=a, sources=sources
        )
    except Exception as e:
        log.error("Failed to save conversation turn for chat %s: %s", chat_id, e)


async def route_and_process(
    query: str,
    analytics_json: Optional[Dict[str, Any]],
    chat_id: int,
    db: AsyncSession,
    chat_history: List[BaseMessage],
) -> AsyncGenerator[str, None]:
    """Main RAG Entry Point. Yields SSE event stream chunks."""
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
        log.info("Contextualized: '%s' -> '%s'", query, standalone_question)

    chat_context_str = ""
    if chat_history:
        recent_history = chat_history[-6:]
        formatted_turns = []
        for msg in recent_history:
            role = "User" if isinstance(msg, HumanMessage) else "SentimentScope"
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            formatted_turns.append(f"{role}: {content}")
        chat_context_str = "\n".join(formatted_turns)

    status_str = await crud.get_chat_embedding_status(db, chat_id)
    is_sql_ready = status_str is not None
    has_dashboard = analytics_json is not None
    is_embeddings_ready = status_str == EmbeddingStatusEnum.completed.value
    dashboard_caps = _get_dashboard_capabilities(analytics_json)

    intent = "vector_search"
    try:
        router_input = {
            "question": standalone_question,
            "sql_ready": is_sql_ready,
            "embeddings_ready": is_embeddings_ready,
            "dashboard_ready": has_dashboard,
            "dashboard_capabilities": dashboard_caps,
        }

        router_llm = get_router_llm()
        raw_class = await (
            ChatPromptTemplate.from_template(ROUTER_SYSTEM_PROMPT)
            | router_llm
            | StrOutputParser()
        ).ainvoke(router_input)

        parsed_intent = _extract_json_payload(raw_class)
        intent = parsed_intent.get("intent", "vector_search") if parsed_intent else "vector_search"
    except Exception as e:
        log.warning("Router classification failed: %s", e)
        if is_embeddings_ready:
            intent = "vector_search"
        elif is_sql_ready:
            intent = "sql_agent"
        else:
            intent = "system_not_ready"

    log.info("Router Decision: %s (SQL: %s, Embeddings: %s)", intent, is_sql_ready, is_embeddings_ready)

    answer_accum = ""
    sources_list: List[Dict[str, Any]] = []
    structured_data = "None"
    rag_context = None

    try:
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
            search_filters = await extract_search_filters(standalone_question)
            if search_filters:
                log.info("Applying filters: %s", search_filters)

            _, sources_list, rag_context = await run_vector_search(
                standalone_question,
                chat_id,
                sender_names=search_filters.get("sender_names"),
                time_ranges=search_filters.get("time_ranges"),
            )

        prompt_inputs = {
            "structured_data": structured_data,
            "rag_context": rag_context if rag_context else "",
            "chat_context": chat_context_str,
            "question": standalone_question,
        }

        final_messages = ChatPromptTemplate.from_template(MASTER_SYSTEM_PROMPT).format_messages(
            **prompt_inputs
        )

        try:
            stream_generator = await execute_resilient_llm(final_messages, stream=True)
            async for chunk in stream_generator:
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                answer_accum += content
                yield f"data: {json.dumps(content)}\n\n"
        except Exception as e:
            log.error("Streaming error: %s", e)
            yield f"data: {json.dumps('System is busy. Please try again.')}\n\n"

        if answer_accum:
            sanitized_sources = _sanitize_sources(sources_list)
            await _save_turn(db, chat_id, query, answer_accum, sanitized_sources)

        final_resp = {
            "answer": answer_accum,
            "route": intent.upper(),
            "sources": _sanitize_sources(sources_list),
        }
        yield f"data: {json.dumps(final_resp)}\n\n"

    except Exception as e:
        log.error("Routing execution error: %s", e, exc_info=True)
        err_msg = "I encountered an error processing your request."
        yield f"data: {json.dumps(err_msg)}\n\n"