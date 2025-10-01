from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.app.config import settings
from pydantic import BaseModel

app = FastAPI(title="WhatsApp Sentiment Analysis API")


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