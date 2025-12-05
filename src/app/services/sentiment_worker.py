import asyncio
import logging
import json
import os
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
from src.app.celery_app import celery_app
from src.app.config import settings


try:
    DATABASE_URL = str(settings.DATABASE_URL)
except ImportError:
    DATABASE_URL = os.getenv("DATABASE_URL")


log = logging.getLogger(__name__)


BROKER_URL = settings.CELERY_BROKER_URL

redis_pool = ConnectionPool.from_url(BROKER_URL, decode_responses=True)

# Optimized Model Loading 
_PIPELINE = None
_MODEL_PATH = "./onnx_model_optimized"


def get_pipeline():
    """
    Loads model with Container-Safe Optimizations.
    """
    global _PIPELINE
    if _PIPELINE is None:
        # --- DEBUGGING BLOCK START ---
        model_file = os.path.join(_MODEL_PATH, "model.onnx")
        if os.path.exists(model_file):
            size = os.path.getsize(model_file)
            log.info(f"DEBUG: Found model at {model_file}")
            log.info(f"DEBUG: File size is {size} bytes")
            
            if size < 5000: # If less than 5KB, it's definitely a pointer file
                with open(model_file, 'r') as f:
                    content = f.read()
                log.error(f"CRITICAL: Model file is too small! Content preview: {content}")
                raise Exception("Model file is a Git LFS pointer, not the actual binary.")
        else:
            log.error(f"CRITICAL: Model file not found at {model_file}")

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

def get_redis_sync():
    """Get a sync redis client from pool"""
    return redis.Redis(connection_pool=redis_pool)

def should_stop(chat_id: int) -> bool:
    """
    Checks Redis for the 'kill switch'. 
    Much faster than DB query, safe to call in loops.
    """
    try:
        r = get_redis_sync()
        # Check if the stop signal key exists
        if r.exists(f"stop_signal_{chat_id}"):
            return True
    except Exception:
        pass # If redis fails, rely on the eventual DB check
    return False

def publish_progress(chat_id: int, status_key: str, data: dict):
    try:
        r = get_redis_sync()
        channel = f"chat_progress_{chat_id}"
        message = {"status": status_key, "data": data}
        r.publish(channel, json.dumps(message))
    except Exception as e:
        log.error(f"Redis Publish Error: {e}")


async def _process_smart_buffer(db, chat_id, buffer, create_func, get_text_func, pipe):
    if not buffer:
        return

    # 1. Sort by length (Performance Optimization)
    buffer.sort(key=lambda x: len(get_text_func(x)))
    
    BATCH_SIZE = 16 
    processed_count = 0
    UPDATE_FREQUENCY = 25 # Updated: More frequent visual updates

    # 2. Iterate in batches
    for i in range(0, len(buffer), BATCH_SIZE):
        # --- CRITICAL: Check cancellation BEFORE every inference batch ---
        if should_stop(chat_id):
            raise Exception("Cancelled by user")
        
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
            await create_func(db, item_obj.id, payload, should_commit=False)
        
        processed_count += len(batch_items)

        # 3. Commit & Publish
        if processed_count % UPDATE_FREQUENCY == 0 or processed_count == len(buffer):
            await db.commit() 
            
            # Use a lightweight progress calculation if possible, or fetch from DB
            progress = await crud.get_sentiment_progress(db, chat_id)
            total = progress["messages_total"] + progress["segments_total"]
            done = progress["messages_scored"] + progress["segments_scored"]
            percent = int(100 * (done / total)) if total > 0 else 0
            
            publish_progress(chat_id, "progress", {
                "percent": percent,
                "messages_done": progress["messages_scored"],
                "messages_total": progress["messages_total"], # <--- ADD THIS
                "segments_done": progress["segments_scored"],
                "segments_total": progress["segments_total"], # <--- ADD THIS
                "total": total
            })


async def _process_stage_batch(db, chat_id, stream_func, create_func, get_text_func, pipe, buffer_size=100):
    buffer = []
    
    # We remove the initial `chat` DB fetch here to save time. 
    # We rely on Redis for immediate cancellation checks.
    
    async for item in stream_func(db, chat_id):
        # --- Check cancellation during fetching ---
        # We check every 50 items or via Redis instantly
        if len(buffer) % 50 == 0:
             if should_stop(chat_id):
                 raise Exception("Cancelled by user")

        buffer.append(item)
        
        if len(buffer) >= buffer_size:
            log.info(f"Processing buffer of {len(buffer)} items...")
            await _process_smart_buffer(db, chat_id, buffer, create_func, get_text_func, pipe)
            buffer = [] 

    if buffer:
        await _process_smart_buffer(db, chat_id, buffer, create_func, get_text_func, pipe)


async def process_chat_logic(chat_id: int):
    pipe = get_pipeline()
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
                # Reduced buffer size from 500 -> 100 for CPU
                # This ensures the user sees progress within the first few seconds
                await _process_stage_batch(
                    db=db, 
                    chat_id=chat_id,
                    stream_func=crud.stream_unscored_messages,
                    create_func=crud.create_message_sentiment,
                    get_text_func=lambda x: x.content,
                    pipe=pipe,
                    buffer_size=100 
                )

                await _process_stage_batch(
                    db=db, 
                    chat_id=chat_id,
                    stream_func=crud.stream_unscored_sender_segments,
                    create_func=crud.create_segment_sentiment,
                    get_text_func=lambda x: x.combined_text,
                    pipe=pipe,
                    buffer_size=100 
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
                    # Note: We don't need to publish "cancelled" here because 
                    # the API endpoint already did it 'optimistically'.
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