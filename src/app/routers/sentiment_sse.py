from fastapi import APIRouter, Request, HTTPException, status, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import json 
import logging
from redis import asyncio as aioredis

from src.app.config import settings
from src.app.db.session import AsyncSessionLocal, get_db
from src.app import crud, models, schemas

from src.app.security import get_current_user, get_current_user_ws
from src.app.utils.redis_helper import get_redis_client

router = APIRouter()
log = logging.getLogger(__name__)

REDIS_URL = settings.CELERY_BROKER_URL

import time

async def redis_event_generator(chat_id: int, request: Request):
    # 1. Get the best available Redis (Upstash or Local Fallback)
    redis_client = await get_redis_client(use_async=True)
    pubsub = redis_client.pubsub()
    channel = f"chat_progress_{chat_id}"
    
    try: 
        await pubsub.subscribe(channel)

        # 2. Check durable Redis state first (captures queued/provisioning status)
        try:
            redis_state = await redis_client.hgetall(f"chat_progress_state:{chat_id}")
            if redis_state:
                state_status = redis_state.get("status")
                if state_status in ["queued", "provisioning"]:
                    yield f"event: {state_status}\ndata: {json.dumps(redis_state)}\n\n"
                elif state_status == "completed":
                    yield f"event: completed\ndata: {json.dumps({'percent': 100, 'status': 'done'})}\n\n"
                    return
                elif state_status == "failed":
                    yield f"event: error\ndata: {json.dumps({'error': redis_state.get('error', 'Analysis failed')})}\n\n"
                    return
        except Exception as r_err:
            log.warning("Could not check durable Redis state for chat %s: %s", chat_id, r_err)

        # 3. Source-of-Truth Sync from Database
        async with AsyncSessionLocal() as db:
            chat = await crud.get_chat(db, chat_id)
            if chat:
                # Handle terminal states immediately
                if chat.sentiment_status == "completed":
                    yield f"event: completed\ndata: {json.dumps({'percent': 100, 'status': 'done'})}\n\n"
                    return 
                
                elif chat.sentiment_status == "failed":
                    yield f"event: error\ndata: {json.dumps({'error': 'Analysis failed previously'})}\n\n"
                    return

                # Calculate detailed progress for 'processing' or 'pending' states
                progress = await crud.get_sentiment_progress(db, chat_id)
                total = progress["messages_total"] + progress["segments_total"]
                done = progress["messages_scored"] + progress["segments_scored"]
                percent = int(100 * (done / total)) if total > 0 else 0
                
                # Send the full payload so the UI has all counters immediately
                initial_data = {
                    "percent": percent,
                    "messages_done": progress["messages_scored"],
                    "messages_total": progress["messages_total"],
                    "segments_done": progress["segments_scored"],
                    "segments_total": progress["segments_total"],
                    "total": total
                }
                yield f"event: progress\ndata: {json.dumps(initial_data)}\n\n"

        # 4. Listen for Real-time Updates with keep-alive heartbeats
        last_ping = time.time()
        while True:
            if await request.is_disconnected():
                break

            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message and message.get("type") == "message":
                try:
                    payload = json.loads(message["data"])
                    event_type = payload.get("status", "progress")
                    data_body = json.dumps(payload.get("data", {}))
                    
                    yield f"event: {event_type}\ndata: {data_body}\n\n"

                    # Stop the stream on terminal events
                    if event_type in ["completed", "failed", "error", "cancelled"]:
                        break
                except Exception as parse_err:
                    log.warning("Error parsing pubsub payload: %s", parse_err)

            now = time.time()
            if now - last_ping >= 15.0:
                # Standard SSE comment ping to keep proxy connection alive
                yield ": keep-alive\n\n"
                last_ping = now
                    
    except Exception as e:
        log.error(f"SSE Error: {e}")
        yield f"event: error\ndata: {json.dumps({'error': 'Stream internal error'})}\n\n"
    finally:
        await pubsub.unsubscribe(channel)
        await redis_client.close()


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
    chat = await crud.get_chat(db, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    if chat.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Unauthorized")
        
    if chat.sentiment_status == schemas.SentimentStatusEnum.processing.value:
        # 1. DB Update (Persistence)
        chat.cancel_requested = True
        db.add(chat)
        await db.commit()
        
        redis_client = await get_redis_client(use_async=True)
        try:
            # 2. Redis Kill Switch (Speed)
            # Set a flag that expires in 1 hour. Worker checks this frequently.
            await redis_client.setex(f"stop_signal_{chat_id}", 3600, "1")

            # 3. Optimistic UI Update (UX)
            # Tell frontend "It's done" immediately, even if worker takes 2 more seconds to die.
            await redis_client.publish(
                f"chat_progress_{chat_id}", 
                json.dumps({"status": "cancelled", "data": {"error": "Cancelled by user"}})
            )
        except Exception as e:
            log.error(f"Failed to publish cancel event: {e}")
        finally:
            await redis_client.close()

        return {"status": "cancel_requested", "chat_id": chat_id}
    
    return {"status": "job_not_processing", "chat_id": chat_id}