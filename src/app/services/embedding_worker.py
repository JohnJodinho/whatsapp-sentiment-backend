# src/app/services/embedding_worker.py

import asyncio
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

from qdrant_client import AsyncQdrantClient, models as qmodels
from src.app.config import settings
from src.app.services.sentiment_worker import celery_app
from src.app.db.session import AsyncSessionLocal
from src.app import crud

from src.app.services.embedding_service import embed_texts

log = logging.getLogger(__name__)

# --- Configuration ---
QDRANT_COLLECTION = "chat_vectors"
VECTOR_SIZE = 1536  # Standard for text-embedding-ada-002 / small-3
HARD_CAP_LIMIT = 2000  # Strict limit per chat
HISTORY_LIMIT_DAYS = 730  # 2 years
MIN_WORD_COUNT = 4  # Junk filter

# Initialize Client
qdrant = AsyncQdrantClient(
    url=str(settings.QDRANT_URL),
    api_key=settings.QDRANT_API_KEY,
    timeout=60.0
)

async def _ensure_collection_exists():
    """
    Checks if collection exists; if not, creates it.
    ALWAYS ensures the necessary payload indices exist.
    """
    # 1. Create Collection if it doesn't exist
    if not await qdrant.collection_exists(QDRANT_COLLECTION):
        log.info(f"Creating Qdrant collection: {QDRANT_COLLECTION}")
        await qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=VECTOR_SIZE,
                distance=qmodels.Distance.COSINE
            )
        )

    # 2. Create Payload Index (Safe to run multiple times, Qdrant handles idempotency)
    # We need this to filter by chat_id during the count check
    await qdrant.create_payload_index(
        collection_name=QDRANT_COLLECTION,
        field_name="chat_id",
        field_schema=qmodels.PayloadSchemaType.INTEGER
    )

@celery_app.task(name="generate_embeddings", bind=True)
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
        # We don't re-raise to avoid Celery retry loops on fatal errors
        # (You can add retry logic if needed)


async def process_chat_ingestion(chat_id: int):
    """
    Main logic: Fetch -> Filter -> Embed -> Upload
    """
    await _ensure_collection_exists()
    # --- 1. Hard Cap Check ---
    # Count how many vectors this chat already has in Qdrant
    count_filter = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="chat_id",
                match=qmodels.MatchValue(value=chat_id)
            )
        ]
    )
    
    count_result = await qdrant.count(
        collection_name=QDRANT_COLLECTION,
        count_filter=count_filter
    )
    existing_count = count_result.count

    if existing_count >= HARD_CAP_LIMIT:
        log.warning(f"[Hard Cap] Chat {chat_id} already has {existing_count} vectors. Skipping ingestion.")
        return

    async with AsyncSessionLocal() as db:
        # --- 2. Fetch Source Data ---
        # Fetch raw messages and summarized segments
        messages = await crud.get_messages_by_chat(db, chat_id=chat_id)
        segments = await crud.get_all_sender_segments_for_chat(db, chat_id=chat_id)
        
        items_to_embed: List[Dict[str, Any]] = []
        
        # Calculate cutoff date (timezone aware)
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=HISTORY_LIMIT_DAYS)

        # --- 3. The Filtering Pipeline ---
        
        # A. Process Messages
        for msg in messages:
            # FIX: Ensure timestamp is timezone-aware before comparison
            msg_ts = msg.timestamp
            if msg_ts.tzinfo is None:
                msg_ts = msg_ts.replace(tzinfo=timezone.utc)

            # Time Filter
            if msg_ts < cutoff_date:
                continue
            
            # Junk Filter (Word Count)
            if not msg.content or len(msg.content.split()) < MIN_WORD_COUNT:
                continue
            
            items_to_embed.append({
                "type": "message",
                "id": msg.id,
                "text": msg.content,
                "timestamp": msg_ts, # Use the fixed timestamp
                "source_table": "messages"
            })

        # B. Process Segments (Summaries are high value, usually keep all if within time)
        # Segments link to TimeSegment, so we need to check the time_segment start_time
        # Note: crud.get_all_sender_segments_for_chat joins TimeSegment, 
        # so we assume msg.time_segment.start_time is available if eager loaded, 
        # or we might need to adjust crud. check crud.py -> It eager loads!
        for seg in segments:
            # FIX: Ensure timestamp is timezone-aware before comparison
            seg_ts = seg.time_segment.start_time
            if seg_ts.tzinfo is None:
                seg_ts = seg_ts.replace(tzinfo=timezone.utc)

            if seg_ts < cutoff_date:
                continue
                
            if seg.combined_text:
                items_to_embed.append({
                    "type": "segment",
                    "id": seg.id,
                    "text": seg.combined_text,
                    "timestamp": seg_ts, # Use the fixed timestamp
                    "source_table": "segments_sender"
                })

        # --- 4. Batch Embedding & Upload ---
        
        # Sort by timestamp (optional, but good for sequential processing)
        items_to_embed.sort(key=lambda x: x["timestamp"])

        remaining_quota = HARD_CAP_LIMIT - existing_count
        if len(items_to_embed) > remaining_quota:
            log.info(f"[Hard Cap] Truncating input from {len(items_to_embed)} to {remaining_quota} items.")
            items_to_embed = items_to_embed[-remaining_quota:] # Keep newest

        if not items_to_embed:
            log.info(f"No valid items to embed for Chat {chat_id} after filtering.")
            return

        # Process in batches of 100
        BATCH_SIZE = 100
        total_uploaded = 0
        
        for i in range(0, len(items_to_embed), BATCH_SIZE):
            batch = items_to_embed[i : i + BATCH_SIZE]
            texts = [item["text"] for item in batch]
            
            # A. Generate Embeddings (Azure)
            try:
                embeddings = await embed_texts(texts) #
            except Exception as e:
                log.error(f"Azure Embedding API failed on batch {i}: {e}")
                continue # Skip this batch, try next

            # B. Prepare Qdrant Points
            points = []
            for j, item in enumerate(batch):
                embedding = embeddings[j]
                if not embedding: 
                    continue # Skip failed embeddings

                # Qdrant requires UUID or Integer IDs. 
                # Since we have multiple tables (messages, segments) with overlapping Int IDs,
                # we must generate a deterministic UUID based on type+id.
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{item['source_table']}_{item['id']}"))
                
                payload = {
                    "source_table": item["source_table"],
                    "source_id": item["id"],
                    "chat_id": chat_id,
                    "text": item["text"], # Store text in Qdrant payload for retrieval
                    "timestamp": item["timestamp"].isoformat()
                }

                points.append(qmodels.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                ))

            # C. Upsert to Qdrant
            if points:
                await qdrant.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=points,
                    wait=False
                )
                total_uploaded += len(points)
                log.info(f"Uploaded batch {i//BATCH_SIZE + 1} ({len(points)} vectors) for Chat {chat_id}")

        log.info(f"Total {total_uploaded} vectors ingested for Chat {chat_id}")