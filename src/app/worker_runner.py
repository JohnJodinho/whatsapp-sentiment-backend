# src/app/worker_runner.py

import sys
import os
import time
import json
import asyncio
import logging
import argparse

from src.app.config import settings
from src.app.utils.redis_helper import get_redis_client
from src.app.services.sentiment_worker import process_chat_logic
from src.app.services.embedding_worker import process_chat_ingestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("worker_runner")

ACTIVE_RUNNERS_KEY = "active_runners_set"
PENDING_QUEUE_KEY = "queue:pending_chats"


async def run_worker_lifecycle(initial_chat_id: int):
    """
    Executes the compute job for a chat, and reuses the warm environment
    to process any subsequent queued chats before shutting down.
    """
    current_chat_id = initial_chat_id
    redis = await get_redis_client(use_async=True)

    try:
        while current_chat_id:
            log.info("🚀 [RUNNER] Processing started for Chat %s", current_chat_id)

            # 1. Update State to Active
            await redis.sadd(ACTIVE_RUNNERS_KEY, str(current_chat_id))
            await redis.hset(
                f"chat_progress_state:{current_chat_id}",
                mapping={
                    "status": "processing",
                    "percent": 5,
                    "updated_at": str(time.time()),
                },
            )
            await redis.publish(
                f"chat_progress_{current_chat_id}",
                json.dumps({
                    "status": "progress",
                    "data": {
                        "percent": 5,
                        "status": "processing",
                        "message": "Analyzing sentiments and communication patterns...",
                    },
                }),
            )

            # 2. Phase 1: Sentiment Analysis
            log.info("📊 [Phase 1/2] Running Sentiment Analysis for Chat %s...", current_chat_id)
            try:
                await process_chat_logic(current_chat_id)
                log.info("✅ Sentiment Analysis finished for Chat %s", current_chat_id)
            except Exception as e:
                log.error("❌ Sentiment Analysis failed for Chat %s: %s", current_chat_id, e, exc_info=True)
                await redis.hset(
                    f"chat_progress_state:{current_chat_id}",
                    mapping={"status": "failed", "error": str(e)},
                )
                await redis.srem(ACTIVE_RUNNERS_KEY, str(current_chat_id))
                # Check for next queued chat before exiting
                next_queued = await redis.lpop(PENDING_QUEUE_KEY)
                current_chat_id = int(next_queued) if next_queued else None
                continue

            # 3. Phase 2: Vector Embedding & Ingestion
            log.info("🧠 [Phase 2/2] Generating Vector Embeddings for Chat %s...", current_chat_id)
            await redis.hset(
                f"chat_progress_state:{current_chat_id}",
                mapping={
                    "status": "embedding",
                    "percent": 70,
                    "updated_at": str(time.time()),
                },
            )
            await redis.publish(
                f"chat_progress_{current_chat_id}",
                json.dumps({
                    "status": "progress",
                    "data": {
                        "percent": 70,
                        "status": "embedding",
                        "message": "Generating semantic vector embeddings...",
                    },
                }),
            )

            try:
                await process_chat_ingestion(current_chat_id)
                log.info("✅ Vector Ingestion finished for Chat %s", current_chat_id)
            except Exception as e:
                log.error("❌ Vector Ingestion failed for Chat %s: %s", current_chat_id, e, exc_info=True)
                # We continue even if embeddings fail, since sentiment succeeded

            # 4. Final Success State
            await redis.hset(
                f"chat_progress_state:{current_chat_id}",
                mapping={
                    "status": "completed",
                    "percent": 100,
                    "updated_at": str(time.time()),
                },
            )
            await redis.publish(
                f"chat_progress_{current_chat_id}",
                json.dumps({
                    "status": "completed",
                    "data": {
                        "percent": 100,
                        "status": "done",
                        "message": "All analyses and vectors generated successfully.",
                    },
                }),
            )

            # Free this chat from active runners
            await redis.srem(ACTIVE_RUNNERS_KEY, str(current_chat_id))
            log.info("🎉 Completed all tasks for Chat %s", current_chat_id)

            # 5. RUNNER REUSE: Drain the waiting queue if any other chats arrived!
            next_chat_str = await redis.lpop(PENDING_QUEUE_KEY)
            if next_chat_str:
                log.info("🔄 [REUSE] Warm runner claiming queued Chat %s (Zero cold start!)", next_chat_str)
                current_chat_id = int(next_chat_str)
            else:
                log.info("🏁 [DRAIN] Queue is empty. Runner self-terminating cleanly.")
                current_chat_id = None

    except Exception as exc:
        log.critical("Runner encountered unhandled fatal exception: %s", exc, exc_info=True)
        if current_chat_id:
            await redis.srem(ACTIVE_RUNNERS_KEY, str(current_chat_id))
        raise exc
    finally:
        await redis.close()


def main():
    parser = argparse.ArgumentParser(description="Distributed WhatsApp Sentiment & Embedding Runner")
    parser.add_argument("--chat-id", type=int, default=None, help="Target Chat ID to process")
    args = parser.parse_args()

    chat_id = args.chat_id or int(os.environ.get("CHAT_ID", 0))
    if not chat_id:
        log.error("No --chat-id specified and CHAT_ID environment variable not found.")
        sys.exit(1)

    log.info("Initializing worker runner for Chat ID: %d", chat_id)
    asyncio.run(run_worker_lifecycle(chat_id))
    log.info("Worker process completed with exit code 0.")


if __name__ == "__main__":
    main()
