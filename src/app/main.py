from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.app.config import settings
from src.app.db.session import engine
from contextlib import asynccontextmanager


async def test_db():
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: print("DB Connected!", c))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup logic
    await test_db()
    yield
    # shutdown logic (if any)
    # e.g. close connections, cleanup, etc
app = FastAPI(title="WhatsApp Sentiment Analysis API", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/")
async def root():
    return {"ok": True, "message": "Welcome to the WhatsApp Sentiment Analysis API"}
