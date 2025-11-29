# src/app/routers/rag.py

import logging
from fastapi import APIRouter, Depends, HTTPException

from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from src.app.db.session import get_db
from src.app import crud
from src.app.schemas import RagQueryRequest, RagQueryResponse
from src.app.services import rag_service_prev
from src.app.schemas import ConversationHistoryItem

router = APIRouter()
log = logging.getLogger(__name__)

@router.post(
    "/chat/{chat_id}/query",
    response_model=RagQueryResponse
)
async def query_chat(
    chat_id: int,
    payload: RagQueryRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Main endpoint for asking questions to a chat.

    """

    chat = await crud.get_chat(db, chat_id=chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
   
    try:
        response = await rag_service_prev.process_rag_query(chat_id, payload)
        return response
    except Exception as e:
        log.error(f"RAG endpoint failed for chat {chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing your question.")


@router.delete("/chat/{chat_id}/history", status_code=204)
async def clear_chat_history(
    chat_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Clears the conversation memory for this chat.
    """

    chat = await crud.get_chat(db, chat_id=chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")


    await crud.clear_conversation_history(db, chat_id)
    
    return 

@router.get(
    "/chat/{chat_id}/history",
    response_model=List[ConversationHistoryItem]
)
async def get_chat_history(
    chat_id: int,
    db: AsyncSession = Depends(get_db)
):
    chat = await crud.get_chat(db, chat_id=chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    history = await crud.get_full_conversation_history(db, chat_id)
    return history