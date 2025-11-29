from fastapi import APIRouter, Request, HTTPException, status, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import json 
import logging
import uuid
from redis import asyncio as aioredis
import os
from src.app.config import settings
from src.app.db.session import AsyncSessionLocal, get_db
from src.app import crud, models, schemas
from src.app.schemas import SentimentStatusEnum 
from src.app.security import get_current_user, get_current_user_ws

router = APIRouter()
log = logging.getLogger(__name__)

REDIS_URL = settings.CELERY_BROKER_URL
async def redis_event_generator(chat_id: int, request: Request):
    try:
        redis_client = aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
        pubsub = redis_client.pubsub()
        channel = f"chat_progress_{chat_id}"
        
        await pubsub.subscribe(channel)
        
        
        yield f"event: connected\ndata: {json.dumps({'msg': 'Connected'})}\n\n"

        async for message in pubsub.listen():
            if await request.is_disconnected():
                break

            if message["type"] == "message":
                payload = json.loads(message["data"])
                
        
                event_type = payload.get("status", "progress") 
                data_body = json.dumps(payload.get("data", {}))
                
                
                yield f"event: {event_type}\ndata: {data_body}\n\n"

                if event_type in ["completed", "failed", "error", "cancelled"]:
                    break
                    
    except Exception as e:
        log.error(f"SSE Error: {e}")
        yield f"event: error\ndata: {json.dumps({'error': 'Stream internal error'})}\n\n"
    finally:
        try:
            await pubsub.unsubscribe(channel)
            await redis_client.close()
        except:
            pass


@router.get("/progress/{chat_id}")
async def sentiment_progress_stream(request: Request, chat_id: int, token: str = Query(...)):
    async with AsyncSessionLocal() as db:
        user = await get_current_user_ws(token, db)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")

    return StreamingResponse(
        redis_event_generator(chat_id, request), 
        media_type="text/event-stream"
    )


@router.post("/cancel/{chat_id}", status_code=status.HTTP_202_ACCEPTED)
async def cancel_sentiment_analysis(
    chat_id: int, 
    db: AsyncSession = Depends(get_db), 
    current_user: models.User = Depends(get_current_user)
):
    """
    Cancels the job 
    """
    chat = await crud.get_chat(db, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    if chat.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Unauthorized")
        
    if chat.sentiment_status == schemas.SentimentStatusEnum.processing.value:
        chat.cancel_requested = True
        db.add(chat)
        await db.commit()
        
        try:
            redis_client = aioredis.from_url(REDIS_URL)
            await redis_client.publish(
                f"chat_progress_{chat_id}", 
                json.dumps({"status": "cancelled", "data": {"error": "Cancelled by user"}})
            )
            await redis_client.close()
        except Exception as e:
            log.error(f"Failed to publish cancel event: {e}")

        return {"status": "cancel_requested", "chat_id": chat_id}
    
    return {"status": "job_not_processing", "chat_id": chat_id}