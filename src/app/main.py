from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware import Middleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from fastapi import Request
from sqlalchemy.exc import SQLAlchemyError
import asyncio
import logging
import socket
import uuid
import time
from typing import Callable


from src.app.config import settings, setup_logging
from src.app.db.session import engine  
from src.app.routers import auth, uploads, chats, sentiment_sse, dashboard, sentiment_dashboard, rag
from src.app.limiter import limiter
setup_logging()
logger = logging.getLogger(__name__)
APP_VERSION = "v1"


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.time()
        try:
            response = await call_next(request)
        except Exception:

            logger.exception("Unhandled error during request %s %s", request.method, request.url)
            raise
        process_time = (time.time() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-ms"] = f"{process_time:.2f}"
        logger.info("%s %s %s -> %s (%.2fms) [rid=%s]",
                    request.client.host if request.client else "-",
                    request.method, request.url.path,
                    getattr(response, "status_code", "-"), process_time, request_id)
        return response


async def test_db_connection():
    
    async with engine.begin() as conn:
        await conn.run_sync(lambda sync_conn: None) 

@asynccontextmanager
async def lifespan(app: FastAPI):
   
    logger.info("Starting application: %s", settings.APP_NAME if hasattr(settings, "APP_NAME") else "whatsapp-sentiment")

    if settings.DEBUG and settings.ENVIRONMENT == "production":
        logger.critical("DEBUG mode is enabled while ENVIRONMENT==production")
        raise RuntimeError("Invalid configuration: DEBUG cannot be True in production")

    # Validate CORS config early
    if settings.CORS_ORIGINS == ["*"] and settings.CORS_ALLOW_CREDENTIALS:
        
        logger.critical("CORS misconfiguration: allow_origins='*' cannot be used with allow_credentials=True")
        raise RuntimeError("Invalid CORS configuration: '*' may not be used with credentials")


    max_attempts = getattr(settings, "DB_CONNECT_RETRIES", 3)
    backoff = getattr(settings, "DB_CONNECT_BACKOFF_SECONDS", 2)
    connected = False
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            await test_db_connection()
            connected = True
            logger.info("✅ Database connection successful")
            break
        except Exception as exc:
            last_exc = exc
            logger.warning("Attempt %d/%d - DB connection failed: %s", attempt, max_attempts, exc)
            await asyncio.sleep(backoff * attempt)

    if not connected:
        logger.critical("Could not connect to the database after %d attempts. Shutting down.", max_attempts)
    
        try:
            await engine.dispose()
        except Exception:
            logger.exception("Error while disposing engine after failed startup")

        raise RuntimeError("Database connection failed")
    yield


    logger.info("Shutting down application gracefully")
    try:
        await engine.dispose()
    except Exception:
        logger.exception("Failed to dispose DB engine during shutdown")


middleware = [
    Middleware(RequestIDMiddleware),
    
    Middleware(TrustedHostMiddleware, allowed_hosts=settings.TRUSTED_HOSTS or ["*"]),
]

app = FastAPI(title="WhatsApp Sentiment Analysis API", lifespan=lifespan, middleware=middleware)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=getattr(settings, "CORS_ALLOW_CREDENTIALS", True),
    expose_headers=["X-Request-ID", "X-Process-Time-ms"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):

    logger.warning("HTTPException (%s) - %s %s - %s", exc.status_code, request.method, request.url, exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation error for %s %s: %s", request.method, request.url, exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

@app.exception_handler(OSError)
async def os_exception_handler(request: Request, exc: OSError):

    if isinstance(exc, socket.gaierror):
        logger.exception("DNS/host resolution error while handling request %s %s: %s", request.method, request.url, exc)
        return JSONResponse(status_code=502, content={"detail": "Upstream host resolution failed"})
    logger.exception("OS error while handling request %s %s", request.method, request.url)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):

    logger.exception("Unhandled exception for request %s %s", request.method, request.url)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

@app.exception_handler(SQLAlchemyError)
async def db_exception_handler(request: Request, exc: SQLAlchemyError):
    """
    Globally catches Database errors (connection pool exhaustion, timeouts).
    Returns a 503 Service Unavailable instead of crashing the server.
    """
    # Log the full error for your debugging
    # log.error(f"Database error: {str(exc)}")
    
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Service is busy. Database connection limit reached. Please try again in a moment."
        },
    )

@app.get("/", tags=["health"])
async def root():
    return {"ok": True, "message": "Welcome to the WhatsApp Sentiment Analysis API"}

@app.get("/health/live", tags=["health"])
async def liveness_probe():
    return {"status": "alive"}

@app.get("/health/ready", tags=["health"])
async def readiness_probe():

    try:
        async with engine.connect() as conn:
            await conn.execute("SELECT 1")
    except Exception:
        logger.exception("Readiness check failed: DB not available")
        return JSONResponse(status_code=503, content={"status": "unavailable"})
    return {"status": "ready"}

# -----------------------------------------------------------------------------
# Include routers
# -----------------------------------------------------------------------------
app.include_router(auth.router, prefix=f"/api/{APP_VERSION}/auth", tags=["auth"])
app.include_router(uploads.router, prefix=f"/api/{APP_VERSION}/uploads", tags=["uploads"])
app.include_router(chats.router, prefix=f"/api/{APP_VERSION}/chats", tags=["chats"])
app.include_router(sentiment_sse.router, prefix=f"/api/{APP_VERSION}/sentiment", tags=["sentiment"])
app.include_router(dashboard.router, prefix=f"/api/{APP_VERSION}/dashboard", tags=["dashboard"])
app.include_router(sentiment_dashboard.router, prefix=f"/api/{APP_VERSION}/sentiment-dashboard", tags=["Sentiment Dashboard"])
app.include_router(rag.router, prefix=f"/api/{APP_VERSION}/rag", tags=["RAG Chat"])


def _log_startup_banner():
    masked_db = "(hidden)"
    try:
        masked_db = str(settings.DATABASE_URL)[:100] + "..."
    except Exception:
        pass

    logger.info("Application startup complete")
    logger.info("Environment: %s | Debug: %s | CORS origins: %s", settings.ENVIRONMENT, settings.DEBUG, settings.CORS_ORIGINS)
    logger.info("Database: %s", masked_db)

