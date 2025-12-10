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

# Access Azure settings via Pydantic model as requested
AZURE_ENDPOINT = str(settings.AZURE_LANGUAGE_ENDPOINT)
AZURE_KEY = settings.AZURE_LANGUAGE_KEY

log = logging.getLogger(__name__)

BROKER_URL = settings.CELERY_BROKER_URL
redis_pool = ConnectionPool.from_url(BROKER_URL, decode_responses=True)

# --- GLOBAL CLIENT ---
_AZURE_CLIENT = None

def get_azure_client():
    """
    Lazy-loads the Azure Text Analytics Async Client.
    """
    global _AZURE_CLIENT
    if _AZURE_CLIENT is None:
        log.info("Initializing Azure Text Analytics Client...")
        try:
            credential = AzureKeyCredential(AZURE_KEY)
            _AZURE_CLIENT = TextAnalyticsClient(
                endpoint=AZURE_ENDPOINT, 
                credential=credential
            )
            log.info("Azure Client initialized successfully.")
        except Exception as e:
            log.error(f"Failed to initialize Azure Client: {e}")
            raise e
    return _AZURE_CLIENT

def get_redis_sync():
    """Get a sync redis client from pool"""
    return redis.Redis(connection_pool=redis_pool)

def should_stop(chat_id: int) -> bool:
    """
    Checks Redis for the 'kill switch'. 
    """
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
    """
    Maps Azure result to internal schema:
    - positive/negative/neutral -> kept as is.
    - mixed -> mapped to neutral.
    - Score uses the confidence score of the chosen label.
    """
    sentiment = doc_result.sentiment  # 'positive', 'negative', 'neutral', 'mixed'
    scores = doc_result.confidence_scores

    if sentiment == "positive":
        return "positive", scores.positive
    elif sentiment == "negative":
        return "negative", scores.negative
    elif sentiment == "neutral":
        return "neutral", scores.neutral
    elif sentiment == "mixed":
        # Requirement: Map mixed to neutral.
        # We use the neutral score to be consistent, even if it is low.
        return "neutral", scores.neutral
    
    # Fallback
    return "neutral", 0.0

async def _process_smart_buffer(db, chat_id, buffer, create_func, get_text_func, client):
    if not buffer:
        return

    # 1. Sort by length (Performance Optimization)
    buffer.sort(key=lambda x: len(get_text_func(x)))
    
    # Azure Limit: Max 10 documents per request
    BATCH_SIZE = 10 
    processed_count = 0
    UPDATE_FREQUENCY = 20
    
    # Retry Configuration
    MAX_RETRIES = 3
    BASE_DELAY = 2

    # 2. Iterate in batches
    for i in range(0, len(buffer), BATCH_SIZE):
        # --- CRITICAL: Check cancellation BEFORE every API call ---
        if should_stop(chat_id):
            raise Exception("Cancelled by user")
        
        batch_items = buffer[i : i + BATCH_SIZE]
        
        azure_batch = [
            {"id": str(item.id), "text": get_text_func(item)} 
            for item in batch_items
        ]

        response = None
        
        # --- RETRY LOGIC START ---
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await client.analyze_sentiment(documents=azure_batch)
                break # Success, exit retry loop
            
            except HttpResponseError as e:
                # Handle Rate Limiting (429)
                if e.status_code == 429:
                    if attempt == MAX_RETRIES:
                        log.error(f"Max retries ({MAX_RETRIES}) reached for batch. Failing.")
                        raise e
                    
                    # Calculate Backoff: Use 'Retry-After' header or default exponential backoff
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
                    # Non-retriable error (e.g., 401 Unauthorized, 400 Bad Request)
                    log.error(f"Azure API Error: {e.message}")
                    raise e
            except Exception as e:
                log.error(f"Unexpected batch error: {e}")
                raise e
        # --- RETRY LOGIC END ---

        # Process results if response exists
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

        # 3. Commit & Publish
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
        # --- Check cancellation during fetching ---
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
    # Get the shared Azure client
    client = get_azure_client()
    
    worker_engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
    WorkerSession = async_sessionmaker(worker_engine, expire_on_commit=False)

    try:
        async with WorkerSession() as db:
            # Initial Status Update
            chat = await crud.get_chat(db, chat_id)
            if not chat: return
            
            # Double check cancel before starting
            if should_stop(chat_id) or chat.cancel_requested:
                 raise Exception("Cancelled by user")

            if chat.sentiment_status != models.SentimentStatusEnum.processing.value:
                chat.sentiment_status = models.SentimentStatusEnum.processing.value
                db.add(chat)
                await db.commit()

            try:
                # Reduced buffer size to 50 ensures frequent updates 
                # and aligns with Azure's 10-doc limit (5 calls per buffer)
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

                # Finalize
                final_progress = await crud.get_sentiment_progress(db, chat_id)
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
        # We do NOT close the global _AZURE_CLIENT here, as it is shared across tasks.
        # It will be closed when the application/worker shuts down.
        await worker_engine.dispose()

@celery_app.task(name="src.app.services.sentiment_worker.analyze_sentiment_task", bind=True, max_retries=3)
def analyze_sentiment_task(self, chat_id: int):
    try:
        asyncio.run(process_chat_logic(chat_id))
    except Exception as exc:
        if str(exc) == "Cancelled by user":
            return
        # Exponential backoff for retries
        self.retry(exc=exc, countdown=5)