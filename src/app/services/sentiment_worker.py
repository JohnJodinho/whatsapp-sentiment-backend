import asyncio
import logging
import json
import os
import redis
from redis import ConnectionPool
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool
from azure.ai.textanalytics.aio import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

from src.app import crud, models
from src.app.celery_app import celery_app
from src.app.config import settings

# --- CONFIGURATION ---
try:
    DATABASE_URL = str(settings.DATABASE_URL)
except ImportError:
    DATABASE_URL = os.getenv("DATABASE_URL")

AZURE_ENDPOINT = str(settings.AZURE_LANGUAGE_ENDPOINT)
AZURE_KEY = settings.AZURE_LANGUAGE_KEY

log = logging.getLogger(__name__)

BROKER_URL = settings.CELERY_BROKER_URL
redis_pool = ConnectionPool.from_url(BROKER_URL, decode_responses=True)

# --- REMOVED GLOBAL CLIENT TO FIX EVENT LOOP ERROR ---

def get_redis_sync():
    """Get a sync redis client from pool"""
    return redis.Redis(connection_pool=redis_pool)

def should_stop(chat_id: int) -> bool:
    try:
        r = get_redis_sync()
        if r.exists(f"stop_signal_{chat_id}"):
            return True
    except Exception:
        pass 
    return False

def publish_progress(chat_id: int, status_key: str, data: dict):
    try:
        r = get_redis_sync()
        channel = f"chat_progress_{chat_id}"
        message = {"status": status_key, "data": data}
        r.publish(channel, json.dumps(message))
    except Exception as e:
        log.error(f"Redis Publish Error: {e}")

def map_azure_result(doc_result):
    sentiment = doc_result.sentiment
    scores = doc_result.confidence_scores

    if sentiment == "positive":
        return "positive", scores.positive
    elif sentiment == "negative":
        return "negative", scores.negative
    elif sentiment == "neutral":
        return "neutral", scores.neutral
    elif sentiment == "mixed":
        return "neutral", scores.neutral
    
    return "neutral", 0.0

async def _process_smart_buffer(db, chat_id, buffer, create_func, get_text_func, client):
    if not buffer:
        return

    buffer.sort(key=lambda x: len(get_text_func(x)))
    
    BATCH_SIZE = 10 
    processed_count = 0
    UPDATE_FREQUENCY = 20
    
    MAX_RETRIES = 3
    BASE_DELAY = 2

    for i in range(0, len(buffer), BATCH_SIZE):
        if should_stop(chat_id):
            raise Exception("Cancelled by user")
        
        batch_items = buffer[i : i + BATCH_SIZE]
        
        azure_batch = [
            {"id": str(item.id), "text": get_text_func(item)} 
            for item in batch_items
        ]

        response = None
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await client.analyze_sentiment(documents=azure_batch)
                break 
            except HttpResponseError as e:
                if e.status_code == 429:
                    if attempt == MAX_RETRIES:
                        log.error(f"Max retries ({MAX_RETRIES}) reached for batch. Failing.")
                        raise e
                    
                    delay = BASE_DELAY * (2 ** attempt)
                    if e.response and 'Retry-After' in e.response.headers:
                        try:
                            retry_after = int(e.response.headers['Retry-After'])
                            delay = max(delay, retry_after)
                        except (ValueError, TypeError):
                            pass
                    
                    log.warning(f"Rate limited (429). Retrying batch in {delay}s (Attempt {attempt + 1}/{MAX_RETRIES})...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    log.error(f"Azure API Error: {e.message}")
                    raise e
            except Exception as e:
                log.error(f"Unexpected batch error: {e}")
                raise e

        if response:
            try:
                for idx, doc_result in enumerate(response):
                    if doc_result.is_error:
                        log.error(f"Document Error (ID: {doc_result.id}): {doc_result.error.code} - {doc_result.error.message}")
                        label, score = "neutral", 0.0
                    else:
                        label, score = map_azure_result(doc_result)

                    payload = {
                        "overall_label": label,
                        "overall_label_score": score,
                    }
                    
                    item_obj = batch_items[idx]
                    await create_func(db, item_obj.id, payload, should_commit=False)

            except Exception as map_exc:
                log.error(f"Result mapping failed: {map_exc}")
                raise map_exc
        
        processed_count += len(batch_items)

        if processed_count % UPDATE_FREQUENCY == 0 or processed_count == len(buffer):
            await db.commit() 
            
            progress = await crud.get_sentiment_progress(db, chat_id)
            total = progress["messages_total"] + progress["segments_total"]
            done = progress["messages_scored"] + progress["segments_scored"]
            percent = int(100 * (done / total)) if total > 0 else 0
            
            publish_progress(chat_id, "progress", {
                "percent": percent,
                "messages_done": progress["messages_scored"],
                "messages_total": progress["messages_total"],
                "segments_done": progress["segments_scored"],
                "segments_total": progress["segments_total"],
                "total": total
            })

async def _process_stage_batch(db, chat_id, stream_func, create_func, get_text_func, client, buffer_size=100):
    buffer = []
    
    async for item in stream_func(db, chat_id):
        if len(buffer) % 50 == 0:
             if should_stop(chat_id):
                 raise Exception("Cancelled by user")

        buffer.append(item)
        
        if len(buffer) >= buffer_size:
            log.info(f"Processing buffer of {len(buffer)} items...")
            await _process_smart_buffer(db, chat_id, buffer, create_func, get_text_func, client)
            buffer = [] 

    if buffer:
        await _process_smart_buffer(db, chat_id, buffer, create_func, get_text_func, client)


async def process_chat_logic(chat_id: int):
    # --- FIX: Instantiate Client INSIDE the task loop ---
    credential = AzureKeyCredential(AZURE_KEY)
    
    # Use 'async with' to ensure the client (and its aiohttp session) 
    # is properly closed when this task finishes.
    async with TextAnalyticsClient(endpoint=AZURE_ENDPOINT, credential=credential) as client:
        
        worker_engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
        WorkerSession = async_sessionmaker(worker_engine, expire_on_commit=False)

        try:
            async with WorkerSession() as db:
                chat = await crud.get_chat(db, chat_id)
                if not chat: return
                
                if should_stop(chat_id) or chat.cancel_requested:
                     raise Exception("Cancelled by user")

                if chat.sentiment_status != models.SentimentStatusEnum.processing.value:
                    chat.sentiment_status = models.SentimentStatusEnum.processing.value
                    db.add(chat)
                    await db.commit()

                try:
                    await _process_stage_batch(
                        db=db, 
                        chat_id=chat_id,
                        stream_func=crud.stream_unscored_messages,
                        create_func=crud.create_message_sentiment,
                        get_text_func=lambda x: x.content,
                        client=client,
                        buffer_size=50 
                    )

                    await _process_stage_batch(
                        db=db, 
                        chat_id=chat_id,
                        stream_func=crud.stream_unscored_sender_segments,
                        create_func=crud.create_segment_sentiment,
                        get_text_func=lambda x: x.combined_text,
                        client=client,
                        buffer_size=50 
                    )

                    chat.sentiment_status = models.SentimentStatusEnum.completed.value
                    db.add(chat)
                    await db.commit()
                    
                    publish_progress(chat_id, "completed", {
                        "percent": 100, 
                        "status": "done"
                    })

                except Exception as e:
                    await db.rollback()
                    if str(e) == "Cancelled by user":
                        log.info(f"Chat {chat_id} stopped via Kill Switch.")
                    else:
                        log.error(f"Processing failed: {e}", exc_info=True)
                        chat.sentiment_status = models.SentimentStatusEnum.failed.value
                        db.add(chat)
                        await db.commit()
                        publish_progress(chat_id, "error", {"error": str(e)})
                    raise e
        finally:
            await worker_engine.dispose()

@celery_app.task(name="src.app.services.sentiment_worker.analyze_sentiment_task", bind=True, max_retries=3)
def analyze_sentiment_task(self, chat_id: int):
    try:
        asyncio.run(process_chat_logic(chat_id))
    except Exception as exc:
        if str(exc) == "Cancelled by user":
            return
        self.retry(exc=exc, countdown=5)