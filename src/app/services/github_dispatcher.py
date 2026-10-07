# src/app/services/github_dispatcher.py

import json
import time
import logging
import httpx
from typing import Dict, Any, Optional

from src.app.config import settings
from src.app.utils.redis_helper import get_redis_client

log = logging.getLogger(__name__)

ACTIVE_RUNNERS_KEY = "active_runners_set"
PENDING_QUEUE_KEY = "queue:pending_chats"


async def dispatch_chat_worker(chat_id: int) -> Dict[str, Any]:
    """
    Orchestrates distributed worker allocation with concurrency hardcapping and queue management.
    """
    redis = await get_redis_client(use_async=True)

    try:
        active_count = await redis.scard(ACTIVE_RUNNERS_KEY)
        max_runners = getattr(settings, "MAX_CONCURRENT_RUNNERS", 3)

        # 1. Capacity Available -> Dispatch Ephemeral GitHub Actions Runner
        if active_count < max_runners:
            await redis.sadd(ACTIVE_RUNNERS_KEY, str(chat_id))
            await redis.hset(
                f"chat_progress_state:{chat_id}",
                mapping={
                    "status": "provisioning",
                    "percent": 0,
                    "queue_position": 0,
                    "dispatched_at": str(time.time()),
                },
            )
            # Publish initial provisioning event
            await redis.publish(
                f"chat_progress_{chat_id}",
                json.dumps({
                    "status": "provisioning",
                    "data": {
                        "percent": 0,
                        "status": "provisioning",
                        "message": "Allocating dedicated compute runner...",
                    },
                }),
            )

            # Trigger GitHub Actions REST API
            dispatch_ok = await _trigger_github_workflow(chat_id)
            if dispatch_ok:
                return {
                    "status": "dispatched",
                    "chat_id": chat_id,
                    "queue_position": 0,
                    "message": "Dedicated cloud compute runner allocated.",
                }
            else:
                log.warning("GitHub dispatch failed or unconfigured; falling back.")
                await redis.srem(ACTIVE_RUNNERS_KEY, str(chat_id))
                return {
                    "status": "error",
                    "chat_id": chat_id,
                    "message": "GitHub dispatch failed or unconfigured",
                }

        # 2. Capacity Full -> Enqueue in Redis Waiting Queue with Detailed Info
        await redis.rpush(PENDING_QUEUE_KEY, str(chat_id))
        queue_len = await redis.llen(PENDING_QUEUE_KEY)
        est_wait = queue_len * 30

        await redis.hset(
            f"chat_progress_state:{chat_id}",
            mapping={
                "status": "queued",
                "percent": 0,
                "queue_position": queue_len,
                "estimated_wait_seconds": est_wait,
                "queued_at": str(time.time()),
            },
        )
        await redis.publish(
            f"chat_progress_{chat_id}",
            json.dumps({
                "status": "queued",
                "data": {
                    "percent": 0,
                    "status": "queued",
                    "queue_position": queue_len,
                    "estimated_wait_seconds": est_wait,
                    "message": f"Server compute capacity reached ({active_count}/{max_runners} active). Your analysis is queued at position #{queue_len}.",
                },
            }),
        )

        log.info(
            "Chat %s queued at position #%d (estimated wait: %ds)",
            chat_id,
            queue_len,
            est_wait,
        )

        return {
            "status": "queued",
            "chat_id": chat_id,
            "queue_position": queue_len,
            "estimated_wait_seconds": est_wait,
            "message": f"Server compute capacity reached ({active_count}/{max_runners} active). Your analysis is queued at position #{queue_len}.",
        }

    except Exception as e:
        log.error("Error in dispatch_chat_worker for chat %s: %s", chat_id, e, exc_info=True)
        return {
            "status": "error",
            "chat_id": chat_id,
            "message": str(e),
        }
    finally:
        await redis.close()


async def _trigger_github_workflow(chat_id: int) -> bool:
    """Invokes GitHub Actions REST API repository_dispatch / workflow_dispatch."""
    token = getattr(settings, "GITHUB_TOKEN", None)
    repo = getattr(settings, "GITHUB_REPO", "JohnJodinho/whatsapp-sentiment-backend")
    workflow = getattr(settings, "GITHUB_WORKFLOW_FILE", "sentiment_worker_dispatch.yml")
    ref = getattr(settings, "GITHUB_REF", "main")

    if not token:
        log.warning("GITHUB_TOKEN is not set. Cannot trigger GitHub Actions runner.")
        return False

    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {
        "ref": ref,
        "inputs": {
            "chat_id": str(chat_id),
        },
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code in (200, 204):
                log.info("✅ GitHub Actions dispatch succeeded for chat %s (HTTP %d)", chat_id, resp.status_code)
                return True
            else:
                log.error("❌ GitHub Actions dispatch failed for chat %s: HTTP %d - %s", chat_id, resp.status_code, resp.text)
                return False
    except Exception as exc:
        log.error("Failed to connect to GitHub API: %s", exc)
        return False
