import redis
import logging
import json
from src.app.config import settings

log = logging.getLogger(__name__)

# Define priority: Upstash first, then local fallback
REDIS_URLS = [str(settings.CELERY_BROKER_URL), "redis://localhost:6379/0"]

def get_redis_client(use_async=False):
    """
    Returns a Redis client using the first available connection in the priority list.
    Includes keepalives to prevent Upstash 'Connection Reset' errors.
    """
    params = {
        "socket_timeout": 5,
        "socket_connect_timeout": 5,
        "socket_keepalive": True,
        "retry_on_timeout": True,
        "health_check_interval": 20,
        "decode_responses": True
    }

    for url in REDIS_URLS:
        try:
            if use_async:
                from redis import asyncio as aioredis
                client = aioredis.from_url(url, **params)
            else:
                client = redis.from_url(url, **params)
                client.ping() # Synchronous check
            return client
        except Exception as e:
            log.warning(f"⚠️ Redis URL {url} unreachable, trying next fallback... Error: {e}")
            continue
    
    raise ConnectionError("❌ All Redis brokers (Upstash & Local) are currently unavailable.")