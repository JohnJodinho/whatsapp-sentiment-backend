import asyncio
import logging
import json
import os
import redis
from redis import ConnectionPool
from onnxruntime import SessionOptions, GraphOptimizationLevel
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool 
from src.app import crud, models
from src.app.celery_app import celery_app
from src.app.config import settings
from src.app.utils.redis_helper import get_redis_client


log = logging.getLogger(__name__)

# --- CONFIG ---
DATABASE_URL = str(settings.DATABASE_URL)
BROKER_URL = settings.CELERY_BROKER_URL
MODEL_PATH = "./onnx_model_optimized" # Local path in Docker container

redis_pool = ConnectionPool.from_url(BROKER_URL, decode_responses=True)

# --- MODEL LOADER ---
_PIPELINE = None

def get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        log.info(f"Loading ONNX model from {MODEL_PATH}...")
        try:
            sess_options = SessionOptions()
            
            # --- PERFORMANCE FIX ---
            # Set to 0 to let ONNX use all available cores (Hugging Face gives 2 vCPUs)
            # This doubles your inference speed compared to setting it to 1.
            sess_options.intra_op_num_threads = 0 
            sess_options.inter_op_num_threads = 0
            sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
            
            ort_model = ORTModelForSequenceClassification.from_pretrained(
                MODEL_PATH,
                session_options=sess_options,
                provider="CPUExecutionProvider"
            )
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            
            _PIPELINE = pipeline(
                "text-classification",
                model=ort_model,
                tokenizer=tokenizer,
                top_k=None,
                device=-1, # CPU
                truncation=True,
                max_length=512
            )
            log.info("✅ ONNX Model loaded with MULTI-THREADING enabled.")
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            raise e
    return _PIPELINE


def should_stop(chat_id: int) -> bool:
    """
    Checks the currently active Redis (Upstash or Fallback) for a stop signal.
    """
    try:
        r = get_redis_client() 
        if r.exists(f"stop_signal_{chat_id}"):
            log.info(f"🛑 Stop signal detected for chat {chat_id}")
            return True
    except Exception as e:
        log.warning(f"Could not check stop signal: {e}")
    return False

def publish_progress(chat_id: int, status_key: str, data: dict):
    """Resilient publishing that finds the active Redis instance."""
    try:
        r = get_redis_client()
        r.publish(f"chat_progress_{chat_id}", json.dumps({"status": status_key, "data": data}))
    except Exception as e:
        log.error(f"Failed to publish progress to any Redis: {e}")

async def _process_batch(db, chat_id, buffer, create_func, get_text_func, pipe):
    if not buffer: return

    # Sort to minimize padding (Speedup)
    buffer.sort(key=lambda x: len(get_text_func(x)))
    
    # Batch size 32 is a sweet spot for "Large" models on 2 vCPUs
    BATCH_SIZE = 32 
    processed_count = 0

    for i in range(0, len(buffer), BATCH_SIZE):
        if should_stop(chat_id): raise Exception("Cancelled by user")
        
        batch_items = buffer[i : i + BATCH_SIZE]
        texts = [get_text_func(item) for item in batch_items]

        # Run Inference
        preds = pipe(texts, batch_size=len(texts), truncation=True)

        for item_obj, pred in zip(batch_items, preds):
            # Handle list output from pipeline
            if isinstance(pred, list):
                # Find the label with the highest score
                best = max(pred, key=lambda x: x['score'])
                label = best['label']
                score = best['score']
            else:
                label = pred['label']
                score = pred['score']

            payload = {"overall_label": label, "overall_label_score": score}
            await create_func(db, item_obj.id, payload, should_commit=False)
        
        processed_count += len(batch_items)

    # Commit ONCE per buffer (Massive DB Speedup)
    await db.commit()
    
    # Update Progress
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

async def process_chat_logic(chat_id: int):
    pipe = get_pipeline()
    worker_engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
    WorkerSession = async_sessionmaker(worker_engine, expire_on_commit=False)

    async with WorkerSession() as db:
        chat = await crud.get_chat(db, chat_id)
        if not chat: return
        
        # Mark as processing
        # Reconciliation: Always ensure DB shows 'processing' when task starts/resumes
        chat.sentiment_status = models.SentimentStatusEnum.processing.value
        await db.commit()

        try:
            # 1. Process Messages (Resumes where sentiment is NULL)
            buffer = []
            async for item in crud.stream_unscored_messages(db, chat_id):
                buffer.append(item)
                if len(buffer) >= 100:
                    await _process_batch(db, chat_id, buffer, crud.create_message_sentiment, lambda x: x.content, pipe)
                    buffer = []
            if buffer: await _process_batch(db, chat_id, buffer, crud.create_message_sentiment, lambda x: x.content, pipe)
           
            # 2. Process Segments (Same logic)
            buffer = []
            async for item in crud.stream_unscored_sender_segments(db, chat_id):
                buffer.append(item)
                if len(buffer) >= 100:
                    await _process_batch(db, chat_id, buffer, crud.create_segment_sentiment, lambda x: x.combined_text, pipe)
                    buffer = []
            if buffer: await _process_batch(db, chat_id, buffer, crud.create_segment_sentiment, lambda x: x.combined_text, pipe)

            # 3. Finish
            chat.sentiment_status = models.SentimentStatusEnum.completed.value
            db.add(chat)
            await db.commit()
            publish_progress(chat_id, "completed", {"percent": 100, "status": "done"})

        except Exception as e:
            await db.rollback()
            if str(e) == "Cancelled by user":
                log.info(f"Chat {chat_id} Cancelled.")
            else:
                log.error(f"Error: {e}")
                chat.sentiment_status = models.SentimentStatusEnum.failed.value
                db.add(chat)
                await db.commit()
                publish_progress(chat_id, "error", {"error": str(e)})

@celery_app.task(name="src.app.services.sentiment_worker.analyze_sentiment_task", bind=True)
def analyze_sentiment_task(self, chat_id: int):
    asyncio.run(process_chat_logic(chat_id))