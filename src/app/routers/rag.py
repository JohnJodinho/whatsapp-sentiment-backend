import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Optional
from src.app.db.session import get_db
from src.app import crud, models
from src.app.schemas import RagQueryRequest, ConversationHistoryItem
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
    """
    Checks the embedding processing status of a chat.
    Returns: { "status": "pending" | "processing" | "completed" | "failed" }
    """
    chat = await crud.get_chat(db, chat_id=chat_id)
    if not chat or chat.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Assuming crud.get_chat_embedding_status(db, chat_id) exists per instructions
    status = await crud.get_chat_embedding_status(db, chat_id)
    
    # Default to pending if None
    return JSONResponse(content={"status": status or "pending"})
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