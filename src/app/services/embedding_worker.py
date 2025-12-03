# src/app/services/embedding_worker.py

import asyncio
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, AsyncGenerator

from qdrant_client import AsyncQdrantClient, models as qmodels
from src.app.config import settings
from src.app.services.sentiment_worker import celery_app
from src.app.db.session import AsyncSessionLocal
from src.app import crud, schemas

from src.app.services.embedding_service import embed_texts

log = logging.getLogger(__name__)

# --- Configuration ---
QDRANT_COLLECTION = "chat_vectors"
VECTOR_SIZE = 1536  
HARD_CAP_LIMIT = 5000  
HISTORY_LIMIT_DAYS = 730  
MIN_WORD_COUNT = 4  
DB_BATCH_SIZE = 1000
MAX_CONCURRENT_EMBEDS = 5
EMBED_BATCH_SIZE = 100

_qdrant_client: AsyncQdrantClient | None = None


def get_qdrant_client() -> AsyncQdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = AsyncQdrantClient(
            url=str(settings.QDRANT_URL),
            api_key=settings.QDRANT_API_KEY,
            timeout=60.0,
            prefer_grpc=True # Enable gRPC for better performance if supported
        )
    return _qdrant_client



async def _ensure_collection_exists(client: AsyncQdrantClient):
    """Idempotent collection setup."""
    if not await client.collection_exists(QDRANT_COLLECTION):
        log.info(f"Creating Qdrant collection: {QDRANT_COLLECTION}")
        await client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=VECTOR_SIZE,
                distance=qmodels.Distance.COSINE
            )
        )
        # Create Payload Index for faster filtering
        await client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="chat_id",
            field_schema=qmodels.PayloadSchemaType.INTEGER
        )

@celery_app.task(name="generate_embeddings", bind=True, acks_late=True)
def generate_embeddings_task(self, chat_id: int):
    """
    Celery task to orchestrate intelligent ingestion into Qdrant.
    """
    try:
        log.info(f"Starting Qdrant Ingestion for Chat {chat_id}")
       
        asyncio.run(process_chat_ingestion(chat_id))
        
        
        log.info(f"✅ Qdrant Ingestion finished for Chat {chat_id}")
    except Exception as e:
        log.error(f"Embedding task failed for Chat {chat_id}: {e}", exc_info=True)
   


async def process_chat_ingestion(chat_id: int):
    client = get_qdrant_client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_EMBEDS)
    
    try:
        await _ensure_collection_exists(client)

        # 1. Hard Cap Check
        count_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(key="chat_id", match=qmodels.MatchValue(value=chat_id))]
        )
        count_result = await client.count(QDRANT_COLLECTION, count_filter=count_filter)
        
        if count_result.count >= HARD_CAP_LIMIT:
            log.warning(f"[Hard Cap] Chat {chat_id} has {count_result.count} vectors. Skipping.")
            return

        # Calculate remaining quota
        quota = HARD_CAP_LIMIT - count_result.count
        cutoff_aware = datetime.now(timezone.utc) - timedelta(days=HISTORY_LIMIT_DAYS)
        cutoff_date = cutoff_aware.replace(tzinfo=None)
        
        async with AsyncSessionLocal() as db:
            # Update status to processing
            await crud.update_chat_embedding_status(
                db, chat_id, schemas.EmbeddingStatusEnum.processing.value, should_commit=True
            )

            total_ingested = 0
            
            # We create a generator pipeline
            # Generator -> Batches -> Concurrent Processing
            
            async for batch in data_stream_generator(db, chat_id, cutoff_date, quota):
                # Process a batch of text (e.g., 100 items)
                async with semaphore:
                    processed_count = await process_and_upload_batch(client, batch, chat_id)
                    total_ingested += processed_count
                
                # Check quota mid-stream to stop early if we hit the limit
                quota -= processed_count
                if quota <= 0:
                    break

            # Final Success Status
            await crud.update_chat_embedding_status(
                db, chat_id, schemas.EmbeddingStatusEnum.completed.value, should_commit=True
            )
            log.info(f"✅ Ingestion Complete. Total {total_ingested} vectors for Chat {chat_id}")

    except Exception as e:
        log.error(f"Error logic trace: {e}")
        async with AsyncSessionLocal() as db:
            await crud.update_chat_embedding_status(
                db, chat_id, schemas.EmbeddingStatusEnum.failed.value, should_commit=True
            )
        raise e


async def data_stream_generator(db, chat_id: int, cutoff_date: datetime, max_items: int) -> AsyncGenerator[List[Dict], None]:
    """
    Yields batches of data (Messages + Segments) to be processed.
    Uses LIMIT/OFFSET pagination to prevent OOM.
    """
    current_batch = []
    total_processed_count = 0
    
    # ==========================================
    # PHASE 1: Process Messages
    # ==========================================
    msg_offset = 0
    
    while total_processed_count < max_items:
        # Fetch batch from DB
        raw_msgs = await crud.get_messages_batch(
            db, chat_id=chat_id, limit=DB_BATCH_SIZE, offset=msg_offset, min_date=cutoff_date
        )
        
        if not raw_msgs:
            break # No more messages

        for msg in raw_msgs:
            if total_processed_count >= max_items:
                break

            # Junk Filter
            if not msg.content or len(msg.content.split()) < MIN_WORD_COUNT:
                continue

            # Timezone fix
            ts = msg.timestamp if msg.timestamp.tzinfo else msg.timestamp.replace(tzinfo=timezone.utc)
            
            # Add to buffer
            current_batch.append({
                "type": "message",
                "id": msg.id,
                "text": msg.content,
                "timestamp": ts,
                "source_table": "messages"
            })
            total_processed_count += 1

            # Yield if buffer is full
            if len(current_batch) >= EMBED_BATCH_SIZE:
                yield current_batch
                current_batch = []

        msg_offset += DB_BATCH_SIZE

    # ==========================================
    # PHASE 2: Process Segments
    # ==========================================
    # Only start if we haven't hit the hard cap yet
    if total_processed_count < max_items:
        seg_offset = 0
        
        while total_processed_count < max_items:
            # Fetch batch from DB
            raw_segs = await crud.get_segments_batch(
                db, chat_id=chat_id, limit=DB_BATCH_SIZE, offset=seg_offset, min_date=cutoff_date
            )
            
            if not raw_segs:
                break # No more segments

            for seg in raw_segs:
                if total_processed_count >= max_items:
                    break
                
                # Check for content existence
                if not seg.combined_text:
                    continue

                # Timezone fix (Segments usually link to TimeSegment)
                # Ensure your CRUD eager loads 'time_segment'
                seg_ts = seg.time_segment.start_time
                if seg_ts.tzinfo is None:
                    seg_ts = seg_ts.replace(tzinfo=timezone.utc)

                # Add to buffer
                current_batch.append({
                    "type": "segment",
                    "id": seg.id,
                    "text": seg.combined_text,
                    "timestamp": seg_ts,
                    "source_table": "segments_sender"
                })
                total_processed_count += 1

                # Yield if buffer is full
                if len(current_batch) >= EMBED_BATCH_SIZE:
                    yield current_batch
                    current_batch = []

            seg_offset += DB_BATCH_SIZE

    # ==========================================
    # PHASE 3: Flush Remaining
    # ==========================================
    # Yield whatever is left in the buffer (even if it's just 1 item)
    if current_batch:
        yield current_batch


async def process_and_upload_batch(client: AsyncQdrantClient, batch: List[Dict], chat_id: int) -> int:
    """
    Embeds texts and upserts to Qdrant. Returns count of uploaded items.
    """
    texts = [item["text"] for item in batch]
    
    try:
        embeddings = await embed_texts(texts)
    except Exception as e:
        log.error(f"Embedding API failed: {e}")
        return 0

    points = []
    for i, item in enumerate(batch):
        embedding = embeddings[i]
        if not embedding: continue

        # Deterministic UUID
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{item['source_table']}_{item['id']}"))
        
        payload = {
            "source_table": item["source_table"],
            "source_id": item["id"],
            "chat_id": chat_id,
            "text": item["text"],
            "timestamp": item["timestamp"].isoformat()
        }

        points.append(qmodels.PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload
        ))

    if points:
        await client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points,
            wait=False # Fire and forget for speed
        )
    
    return len(points)