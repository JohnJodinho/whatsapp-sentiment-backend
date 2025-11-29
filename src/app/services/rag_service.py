import logging
import json
from typing import List, Dict, Any, Optional, AsyncGenerator

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI


from src.app.config import settings
from src.app.schemas import RagQueryRequest, RagQueryResponse, RagSource, QueryRoute
from src.app.services import router_service
from src.app.services.retrieval_service import SQLAlchemyVectorRetriever
from src.app.db.session import AsyncSessionLocal
from src.app import crud

log = logging.getLogger(__name__)

llm = AzureChatOpenAI(
    api_version=settings.AZURE_OPENAI_API_VERSION_ROUTER,
    azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT_ROUTER),
    api_key=settings.AZURE_OPENAI_API_KEY_ROUTER,
    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_ROUTER,
    temperature=0.0, 
    max_tokens=2000
)


retriever = SQLAlchemyVectorRetriever(
    db_session_factory=AsyncSessionLocal,
    top_k=8 
)


# This rewrites the question to be standalone based on history
CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
that can be understood without the chat history. \
Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

CHAT_PROMPT = ChatPromptTemplate.from_template(
"""
You are a helpful assistant for answering questions about a chat history.
Use only the provided CONTEXT snippets (each from a message or chat segment) to answer the user's question.
Cite your sources inline using [source_table:source_id] (e.g., [messages:123]).
If the answer is not in the context, say "I could not find an answer in the provided chat history."

CONTEXT:
{context_snippets}

QUESTION:
{user_question}

ANSWER:
"""
)

# Template for ANALYTICS-only questions
ANALYTICS_PROMPT = ChatPromptTemplate.from_template(
"""
You are a helpful assistant for answering questions about chat analytics.
Use only the provided DATA to answer the user's question.

RULES:
1. Provide a detailed answer based *only* on the data provided.
2. NEVER mention "ANALYTICS_JSON", "JSON", or internal field names like "kpiMetrics".
3. Instead, refer to the data naturally, e.g., "According to the dashboard," "The data shows," or "The sentiment trends indicate."
4. If the answer is not in the provided data, say "I don't have that data available in the dashboard."

DATA:
{analytics_json_snippet}

QUESTION:
{user_question}

ANSWER:
"""
)

# Template for BOTH (Hybrid) questions
BOTH_PROMPT = ChatPromptTemplate.from_template(
"""
You are a helpful assistant. Use *both* the CONTEXT snippets and the ANALYTICS DATA
to answer the user's question.

RULES:
1. Cite your sources for *chat content* inline using [source_table:source_id] (e.g., [messages:123]).
2. For *analytics data*, refer to it naturally (e.g., "The dashboard shows...", "Statistics indicate..."). DO NOT cite JSON keys or mention "ANALYTICS_JSON".
3. Combine insights from both sources to provide a comprehensive answer.
4. If the answer is not in the provided context or data, say "I don't have enough information to answer that."

CONTEXT:
{context_snippets}

ANALYTICS DATA:
{analytics_json_snippet}

QUESTION:
{user_question}

ANSWER:
"""
)

# --- 3. Define Chains ---
chat_chain = CHAT_PROMPT | llm | StrOutputParser()
analytics_chain = ANALYTICS_PROMPT | llm | StrOutputParser()
both_chain = BOTH_PROMPT | llm | StrOutputParser()


def _format_docs_to_context(docs: List[Document]) -> str:
    """Converts retrieved documents into a string for the prompt."""
    snippets = []
    for doc in docs:
        # Create the header with an f-string
        header = f"[{doc.metadata['source_table']}:{doc.metadata['source_id']}]\n"

        
        # Use simple string concatenation for the content.
        # This prevents errors if the chat content has curly braces.
        snippet = header + doc.page_content
        
        snippets.append(snippet)
    return "\n---\n".join(snippets)



def _serialize_analytics(analytics_json: dict | None) -> str:
    """
    Intelligently summarizes the analytics JSON 
    """
    if not analytics_json:
        return "null"
    
    clean_data = {}

    # --- 1. Process General Dashboard ---
    # Based on dashboardData.ts structure
    if "general_dashboard" in analytics_json:
        gd = analytics_json["general_dashboard"]
        clean_gd = {}
        
        # A. KPIs: Keep label/value, REMOVE sparklines (visual noise)
        if "kpiMetrics" in gd:
            clean_gd["kpiMetrics"] = [
                {k: v for k, v in m.items() if k != "sparkline"} 
                for m in gd.get("kpiMetrics", [])
            ]
            
        # B. Contribution: Keep fully (Answer "Who sent the most?")
        if "contribution" in gd:
            clean_gd["contribution"] = gd["contribution"]
            
        # C. Activity: Keep fully (Answer "Who sends links/media?")
        if "activity" in gd:
            clean_gd["activity"] = gd["activity"]
            
        # D. Timeline: High value, but limit history (last 12 months)
        #    Removes 'messagesOverTime' as this table provides better summary
        if "timeline" in gd:
            # Only keep essential fields from ChatSegmentBase/Multi/Two
            clean_gd["timeline_summary"] = [
                {
                    k: v for k, v in item.items() 
                    if k in ["month", "totalMessages", "peakDay", "activeParticipants", "mostActive", "conversationBalance"]
                }
                for item in gd.get("timeline", [])[:12] # Top 12 most recent months
            ]
            
        # E. Heatmaps: Keep compact (Answer "When are they active?")
        if "activityByDay" in gd:
             # Remove 'fill' color code, keep day/messages
             clean_gd["activityByDay"] = [
                 {k: v for k, v in d.items() if k != "fill"}
                 for d in gd.get("activityByDay", [])
             ]
             
        if "hourlyActivity" in gd:
             clean_gd["hourlyActivity"] = gd.get("hourlyActivity")

        clean_data["general"] = clean_gd

    # --- 2. Process Sentiment Dashboard ---
    # Based on sentimentDashboardData.ts structure
    if "sentiment_dashboard" in analytics_json:
        sd = analytics_json["sentiment_dashboard"]
        clean_sd = {}
        
        # A. KPIs: Keep fully
        if "kpiData" in sd:
            clean_sd["kpiData"] = sd["kpiData"]
            
        # B. Breakdown: Keep fully (Answer "Who is most positive?")
        if "breakdownData" in sd:
            clean_sd["breakdownData"] = sd["breakdownData"]
            
        # C. Highlights: CRITICAL (Answer "Why is it negative?")
        if "highlightsData" in sd:
            clean_sd["highlights"] = sd["highlightsData"]
            
        # D. Time/Trend: Remove 'trendData' (too verbose). 
        #    Keep day/hour aggregates as they are smaller.
        if "dayData" in sd:
            clean_sd["dayData"] = sd["dayData"]
        
        #    Summarize hourly to just top 3 peaks to save tokens?
        #    Or keep as is (24 items is manageable). Let's keep as is.
        if "hourData" in sd:
            clean_sd["hourData"] = sd["hourData"]

        clean_data["sentiment"] = clean_sd

    # Return the slimmed-down version
    # separators=(",", ":") removes all whitespace to save tokens
    return json.dumps(clean_data, separators=(",", ":"))

# --- 5. Main Service Function ---

async def stream_rag_query(chat_id: int, payload: RagQueryRequest) -> AsyncGenerator[str, None]:
    """
    Streams the RAG response token-by-token using SSE format.
    Yields 'data: <token>\n\n' for content.
    Yields 'data: <json_payload>\n\n' as the final message.
    """
    async with AsyncSessionLocal() as db:
        
        # 1. Fetch Chat History
        history_objs = await crud.get_conversation_history(db, chat_id)
        chat_history = []
        for h in history_objs:
            if h.role == "user":
                chat_history.append(HumanMessage(content=h.content))
            else:
                chat_history.append(AIMessage(content=h.content))

        # 2. Reformulate Query
        if chat_history:
            standalone_question = await contextualize_q_chain.ainvoke({
                "chat_history": chat_history,
                "question": payload.question
            })
            log.info(f"Reformulated Query: '{payload.question}' -> '{standalone_question}'")
        else:
            standalone_question = payload.question
    
        # 3. Route the query
        route = await router_service.route_query(standalone_question)
        
        sources: List[RagSource] = []
        docs = []
        input_data = {}
        selected_chain = None

        try:
            # Prepare Context/Data based on Route
            if route == "CHAT":
                docs = await retriever._aget_relevant_documents(
                    query=standalone_question,
                    chat_id=chat_id,
                    run_manager=None
                )
                context_snippets = _format_docs_to_context(docs)
                input_data = {
                    "context_snippets": context_snippets,
                    "user_question": standalone_question
                }
                selected_chain = chat_chain

            elif route == "ANALYTICS":
                analytics_snippet = _serialize_analytics(payload.analytics_json)
                input_data = {
                    "analytics_json_snippet": analytics_snippet,
                    "user_question": standalone_question
                }
                selected_chain = analytics_chain

            elif route == "BOTH":
                docs = await retriever._aget_relevant_documents(
                    query=standalone_question,
                    chat_id=chat_id,
                    run_manager=None
                )
                context_snippets = _format_docs_to_context(docs)
                analytics_snippet = _serialize_analytics(payload.analytics_json)
                input_data = {
                    "context_snippets": context_snippets,
                    "analytics_json_snippet": analytics_snippet,
                    "user_question": standalone_question
                }
                selected_chain = both_chain
            
            # Prepare Sources object
            for doc in docs:
                source = RagSource(
                    source_table=doc.metadata["source_table"],
                    source_id=doc.metadata["source_id"],
                    distance=doc.metadata["distance"],
                    text=doc.page_content
                )
                sources.append(source)

            # 4. Run Streaming Chain
            full_answer = ""
            
            # Stream the tokens
            async for chunk in selected_chain.astream(input_data):

                full_answer += chunk
                yield f"data: {json.dumps(chunk)}\n\n"

            # 5. Persist to DB
            sources_dict = [s.model_dump() for s in sources] if sources else None
            await crud.add_conversation_turn(db, chat_id, payload.question, full_answer, sources=sources_dict)

            # 6. Send Final Payload
            final_response = RagQueryResponse(
                answer=full_answer,
                sources=sources,
                route=route
            )
            
            # Serialize to JSON
            final_json = final_response.model_dump_json()
            yield f"data: {final_json}\n\n"

        except Exception as e:
            log.error(f"Error processing RAG query for chat {chat_id}: {e}", exc_info=True)
            # Send error message to user
            error_msg = "I'm sorry, I encountered an error while processing your request."
            yield f"data: {error_msg}\n\n"
            
            # Still try to return a valid structure so frontend doesn't crash
            final_response = RagQueryResponse(
                answer=error_msg,
                sources=[],
                route=route or "ERROR"
            )
            yield f"data: {final_response.model_dump_json()}\n\n"