import logging
import json
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage, AIMessage
from sqlalchemy import select
from typing import List, Optional
from src.app.db.session import get_db
from src.app import crud, models
from src.app.schemas import RagQueryRequest, RagQueryResponse, ConversationHistoryItem
from src.app.security import get_current_user
from src.app.services import router_service
from src.app.limiter import limiter

router = APIRouter()
log = logging.getLogger(__name__)

# --- NEW ENDPOINT START ---
@router.get("/chat/{chat_id}/status")
async def get_chat_status(
    chat_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    # Query chat id and embeddings_status together to disambiguate "chat not found" from "status is NULL"
    query = select(models.Chat.id, models.Chat.embeddings_status).where(
        models.Chat.id == chat_id,
        models.Chat.owner_id == current_user.id
    )
    result = await db.execute(query)
    row = result.first()

    if row is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    emb_status = row.embeddings_status or "pending"
    return JSONResponse(content={"status": emb_status})


@router.post(
    "/chat/{chat_id}/query/streamed"
)
@limiter.limit("5/minute")
async def query_chat_streamed(
    request: Request,
    chat_id: int,
    payload: RagQueryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Main endpoint for asking questions to a chat (Streaming).
    """

    chat = await crud.get_chat(db, chat_id=chat_id)
    if not chat or chat.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Chat not found")

    
    
    history_objs = await crud.get_conversation_history(db, chat_id, limit=6)
    chat_history_messages = []
    for h in history_objs:
        if h.role == "user":
            chat_history_messages.append(HumanMessage(content=h.content))
        else:
            chat_history_messages.append(AIMessage(content=h.content))

    analytics_data = payload.analytics_json or {}
    return StreamingResponse(
        router_service.route_and_process(
            query=payload.question,
            analytics_json=analytics_data,
            chat_id=chat_id,
            db=db,
            chat_history=chat_history_messages
        ),
        media_type="text/event-stream"
    )


@router.post(
    "/chat/{chat_id}/query",
    response_model=RagQueryResponse
)
@limiter.limit("5/minute")
async def query_chat_sync(
    request: Request,
    chat_id: int,
    payload: RagQueryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Synchronous non-streaming query endpoint used as a fallback.
    """
    chat = await crud.get_chat(db, chat_id=chat_id)
    if not chat or chat.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Chat not found")

    history_objs = await crud.get_conversation_history(db, chat_id, limit=6)
    chat_history_messages = [
        HumanMessage(content=h.content) if h.role == "user" else AIMessage(content=h.content)
        for h in history_objs
    ]

    analytics_data = payload.analytics_json or {}
    final_payload = None

    async for chunk in router_service.route_and_process(
        query=payload.question,
        analytics_json=analytics_data,
        chat_id=chat_id,
        db=db,
        chat_history=chat_history_messages
    ):
        if chunk.startswith("data: "):
            content_str = chunk[6:].strip()
            try:
                parsed = json.loads(content_str)
                if isinstance(parsed, dict) and "route" in parsed:
                    final_payload = parsed
            except Exception:
                pass

    if not final_payload:
        final_payload = {
            "answer": "No response generated.",
            "route": "UNKNOWN",
            "sources": []
        }

    return final_payload





@router.delete("/chat/{chat_id}/history", status_code=204)
@limiter.limit("10/minute")
async def clear_chat_history(
    request: Request,
    chat_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Clears the conversation memory for this chat.
    """

    chat = await crud.get_chat(db, chat_id=chat_id)
    if not chat or chat.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Chat not found")


    await crud.clear_conversation_history(db, chat_id)
    
    return 

@router.get(
    "/chat/{chat_id}/history",
    response_model=List[ConversationHistoryItem]
)
async def get_chat_history(
    chat_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    chat = await crud.get_chat(db, chat_id=chat_id)
    if not chat or chat.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    history = await crud.get_full_conversation_history(db, chat_id)
    return history