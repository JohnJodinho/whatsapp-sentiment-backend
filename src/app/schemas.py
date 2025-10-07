from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class ParticipantBase(BaseModel):
    name: str

class ParticipantRead(ParticipantBase):
    id: int
    chat_id: int
    created_at: datetime

    class Config:
        orm_mode = True

class ChatBase(BaseModel):
    title: Optional[str] = None

class ChatCreate(ChatBase):
    pass

class ChatRead(ChatBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True


class MessageBase(BaseModel):
    timestamp: datetime
    content: str
    raw: Optional[str] = None

class MessageCreate(MessageBase):
    participant_id: Optional[int] = None
    chat_id: int

class MessageRead(MessageBase):
    id: int
    participant: Optional[ParticipantRead] = None

    class Config:
        orm_mode = True

class SentimentBase(BaseModel):
    overall_label: str
    score_positive: float
    score_negative: float
    score_neutral: float
    sentences_summary: Optional[Dict] = None
    opinions: Optional[Dict] = None
    api_version: Optional[str] = None
    analysis_timestamp: Optional[datetime] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None

class SentimentCreate(SentimentBase):
    message_id: int


class SentimentRead(SentimentBase):
    id: int
    message_id: int

    class Config:
        orm_mode = True

