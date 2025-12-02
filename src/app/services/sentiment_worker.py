import asyncio
import logging
import json
import os
import psutil  
import time
from typing import List, Any
from celery import Celery
import redis
from redis import ConnectionPool
import time
from src.app.config import settings
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline
from onnxruntime import SessionOptions, GraphOptimizationLevel
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool 
from src.app import crud, models


try:
    from src.app.config import settings
    DATABASE_URL = str(settings.DATABASE_URL)
except ImportError:
    DATABASE_URL = os.getenv("DATABASE_URL")

from src.app import crud, models

log = logging.getLogger(__name__)


BROKER_URL = settings.CELERY_BROKER_URL
BACKEND_URL = settings.CELERY_RESULT_BACKEND

redis_pool = ConnectionPool.from_url(BROKER_URL, decode_responses=True)
celery_app = Celery(
    "sentiment_worker",
    broker=BROKER_URL,
    backend=BACKEND_URL,
    include=['src.app.services.embedding_worker']
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1 
)

# Optimized Model Loading 
_PIPELINE = None
_MODEL_PATH = "./onnx_model_optimized"
_REDIS_CLIENT = None

def get_pipeline():
    """
    Loads model with Container-Safe Optimizations.
    """
    global _PIPELINE
    if _PIPELINE is None:
        log.info(f"Loading ONNX model from {_MODEL_PATH}...")
        try:
            sess_options = SessionOptions()
            
            # CRITICAL FIX for Containers:
            # Set to 1 to prevent thread thrashing. Let Celery workers provide parallelism.
            # If you are not using Celery concurrency, set this to 2 or 4 max.
            sess_options.intra_op_num_threads = 1 
            sess_options.inter_op_num_threads = 1
            sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
            
            ort_model = ORTModelForSequenceClassification.from_pretrained(
                _MODEL_PATH,
                session_options=sess_options
            )
            tokenizer = AutoTokenizer.from_pretrained(_MODEL_PATH)
            
            _PIPELINE = pipeline(
                "text-classification",
                model=ort_model,
                tokenizer=tokenizer,
                top_k=None,
                device=-1, # CPU
                truncation=True,
                max_length=512
            )
            log.info("ONNX Model loaded with single-threaded worker optimization.")
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            raise e
    return _PIPELINE

# Redis Pub/Sub Helper 
def publish_progress(chat_id: int, status_key: str, data: dict):
    """
    Publishes using the global connection pool for low latency.
    """
    try:
        r = redis.Redis(connection_pool=redis_pool)
        channel = f"chat_progress_{chat_id}"
        message = {"status": status_key, "data": data}
        # Publish and forget - don't block waiting for listeners count logic
        r.publish(channel, json.dumps(message))
    except Exception as e:
        log.error(f"Redis Publish Error: {e}")


async def _process_smart_buffer(db, chat_id, buffer, create_func, get_text_func, pipe):
    """
    Sorts by length for speed, but commits and updates progress frequently for UX.
    """
    if not buffer:
        return

    # 1. Sort by length (Shortest -> Longest) for Dynamic Padding efficiency
    buffer.sort(key=lambda x: len(get_text_func(x)))
    
    # Increased batch size slightly for ONNX vectorization
    BATCH_SIZE = 16 
    
    # 2. Track local progress to update UI frequently
    processed_count = 0
    UPDATE_FREQUENCY = 50 # Update Redis every 50 items

    # Create a batch container
    for i in range(0, len(buffer), BATCH_SIZE):
        batch_items = buffer[i : i + BATCH_SIZE]
        texts = [get_text_func(item) for item in batch_items]

        # Inference
        preds = pipe(texts, batch_size=len(texts), padding=True, truncation=True)

        for item_obj, pred in zip(batch_items, preds):
            p = pred[0] if isinstance(pred, list) else pred
            payload = {
                "overall_label": p.get("label"),
                "overall_label_score": p.get("score_positive", p["score"]) if isinstance(p, dict) else p["score"],
            }
            # Queue the insert in the session
            await create_func(db, item_obj.id, payload, should_commit=False)
        
        processed_count += len(batch_items)

        # 3. Intermediate Commit & Update (Solves the "Lag" issue)
        # We commit every few batches so the DB doesn't lock for too long
        # and the user sees the progress bar moving.
        if processed_count % UPDATE_FREQUENCY == 0 or processed_count == len(buffer):
            await db.commit() 
            
            # Fetch current stats for accurate percentage
            progress = await crud.get_sentiment_progress(db, chat_id)
            total = progress["messages_total"] + progress["segments_total"]
            done = progress["messages_scored"] + progress["segments_scored"]
            percent = int(100 * (done / total)) if total > 0 else 0
            
            publish_progress(chat_id, "progress", {
                "percent": percent,
                "messages_done": progress["messages_scored"],
                "messages_total": progress["messages_total"]
            })

async def _process_stage_batch(db, chat_id, stream_func, create_func, get_text_func, pipe, buffer_size=1000):
    buffer = []
    
    chat = await crud.get_chat(db, chat_id)
    
    async for item in stream_func(db, chat_id):
        buffer.append(item)
        
        # Check cancellation less frequently to save DB calls
        if len(buffer) % 200 == 0:
            await db.refresh(chat)
            if chat.cancel_requested:
                raise Exception("Cancelled by user")

        if len(buffer) >= buffer_size:
            log.info(f"Processing buffer of {len(buffer)} items...")
            await _process_smart_buffer(db, chat_id, buffer, create_func, get_text_func, pipe)
            buffer = [] 

    # Remaining items
    if buffer:
        await _process_smart_buffer(db, chat_id, buffer, create_func, get_text_func, pipe)


async def process_chat_logic(chat_id: int):
    # Initialize pipeline once per worker process lifespan
    pipe = get_pipeline()
    
    worker_engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
    WorkerSession = async_sessionmaker(worker_engine, expire_on_commit=False)

    try:
        async with WorkerSession() as db:
            chat = await crud.get_chat(db, chat_id)
            if not chat:
                return

            if chat.sentiment_status != models.SentimentStatusEnum.processing.value:
                chat.sentiment_status = models.SentimentStatusEnum.processing.value
                db.add(chat)
                await db.commit()

            try:
                # Process Messages
                await _process_stage_batch(
                    db=db, 
                    chat_id=chat_id,
                    stream_func=crud.stream_unscored_messages,
                    create_func=crud.create_message_sentiment,
                    get_text_func=lambda x: x.content,
                    pipe=pipe,
                    buffer_size=500 # Reduced buffer size for better responsiveness
                )

                # Process Segments
                await _process_stage_batch(
                    db=db, 
                    chat_id=chat_id,
                    stream_func=crud.stream_unscored_sender_segments,
                    create_func=crud.create_segment_sentiment,
                    get_text_func=lambda x: x.combined_text,
                    pipe=pipe,
                    buffer_size=200 
                )

                # Finalize
                final_progress = await crud.get_sentiment_progress(db, chat_id)
                chat.sentiment_status = models.SentimentStatusEnum.completed.value
                db.add(chat)
                await db.commit()
                
                publish_progress(chat_id, "completed", {
                    "percent": 100, 
                    "status": "done",
                    "messages_done": final_progress["messages_scored"]
                })

            except Exception as e:
                await db.rollback()
                if str(e) == "Cancelled by user":
                    log.info(f"Chat {chat_id} stopped.")
                else:
                    log.error(f"Processing failed: {e}", exc_info=True)
                    chat.sentiment_status = models.SentimentStatusEnum.failed.value
                    db.add(chat)
                    await db.commit()
                    publish_progress(chat_id, "error", {"error": str(e)})
                raise e
    finally:
        await worker_engine.dispose()

@celery_app.task(name="analyze_sentiment", bind=True, max_retries=3)
def analyze_sentiment_task(self, chat_id: int):
    try:
        # Use asyncio.run for the async entry point
        asyncio.run(process_chat_logic(chat_id))
    except Exception as exc:
        if str(exc) == "Cancelled by user":
            return
        self.retry(exc=exc, countdown=5)