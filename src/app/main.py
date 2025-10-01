from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="WhatsApp Sentiment Analysis API")

@app.get("/")
async def root():
    return {"ok": True, "message": "Welcome to the WhatsApp Sentiment Analysis API"}