from datetime import datetime, timezone
from typing import List, Optional, Dict
from sqlalchemy import (
    JSON, Integer, String, DateTime, Text, ForeignKey, Float,
    UniqueConstraint, Boolean, BigInteger, func, Index
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
import uuid
from sqlalchemy.dialects.postgresql import UUID
from src.app.db.base import Base
from src.app.schemas import SentimentStatusEnum, EmbeddingStatusEnum
from pgvector.sqlalchemy import Vector


class User(Base):
    __tablename__ = 'users'
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    chats: Mapped[List["Chat"]] = relationship(
        "Chat", 
        back_populates="owner", 
        lazy="selectin",
        passive_deletes=True
    )

class Chat(Base):
    __tablename__ = "chats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    sentiment_status: Mapped[SentimentStatusEnum] = mapped_column(
        String(30), nullable=False, default=SentimentStatusEnum.pending.value, index=True
    )
    # In your models.py

    embeddings_status: Mapped[EmbeddingStatusEnum] = mapped_column(
        String(30), 
        nullable=False, 
        default=EmbeddingStatusEnum.pending.value,       # Keeps Python logic happy
        server_default=EmbeddingStatusEnum.pending.value, # <--- ADDS SQL DEFAULT
        index=True
    )
    cancel_requested: Mapped[Optional[bool]] = mapped_column(Boolean, default=False)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="chat",
        lazy="selectin",
        passive_deletes=True
    )
    time_segments: Mapped[List["TimeSegment"]] = relationship(
        "TimeSegment",
        back_populates="chat",
        lazy="selectin",
        passive_deletes=True
    )

    history: Mapped[List["ConversationHistory"]] = relationship(
        "ConversationHistory",
        back_populates="chat",
        lazy="selectin",
        passive_deletes=True
    )
    owner: Mapped["User"] = relationship(
        "User", 
        back_populates="chats",
        lazy="selectin",
        passive_deletes=True
    )


class Participant(Base):
    __tablename__ = "participants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(
        ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )

    __table_args__ = (UniqueConstraint("chat_id", "name", name="uq_participant_chat_name"),)

    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="participant",
        lazy="selectin",
        passive_deletes=True
    )
    sender_segments: Mapped[List["SenderSegment"]] = relationship(
        "SenderSegment",
        back_populates="participant",
        lazy="selectin",
        passive_deletes=True
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(
        ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True
    )
    participant_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("participants.id", ondelete="SET NULL"), nullable=True, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    raw: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Analytics fields
    word_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    emojis_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    links_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_question: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_media: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    chat: Mapped["Chat"] = relationship(
        "Chat",
        back_populates="messages",
        lazy="selectin",
        passive_deletes=True
    )
    participant: Mapped[Optional["Participant"]] = relationship(
        "Participant",
        back_populates="messages",
        lazy="selectin"
    )
    sentiment: Mapped[Optional["MessageSentiment"]] = relationship(
        "MessageSentiment",
        back_populates="message",
        uselist=False,
        lazy="joined",
        passive_deletes=True
    )


class SenderSegment(Base):
    __tablename__ = "segments_sender"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time_segment_id: Mapped[int] = mapped_column(
        ForeignKey("segments_time.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sender_id: Mapped[int] = mapped_column(
        ForeignKey("participants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    combined_text: Mapped[str] = mapped_column(Text, nullable=False)

    time_segment: Mapped["TimeSegment"] = relationship(
        "TimeSegment",
        back_populates="sender_segments",
        lazy="selectin",
        passive_deletes=True
    )
    participant: Mapped["Participant"] = relationship(
        "Participant",
        back_populates="sender_segments",
        lazy="selectin",
        passive_deletes=True
    )
    sentiment: Mapped[Optional["SegmentSentiment"]] = relationship(
        "SegmentSentiment",
        back_populates="sender_segment",
        uselist=False,
        lazy="joined",
        passive_deletes=True
    )
class TimeSegment(Base):
    __tablename__ = "segments_time"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(
        ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True
    )
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    end_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    duration_minutes: Mapped[float] = mapped_column(Float, nullable=False)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    chat: Mapped["Chat"] = relationship(
        "Chat",
        back_populates="time_segments",
        lazy="selectin",
        passive_deletes=True
    )
    sender_segments: Mapped[List["SenderSegment"]] = relationship(
        "SenderSegment",
        back_populates="time_segment",
        lazy="selectin",
        passive_deletes=True
    )
    

class MessageSentiment(Base):
    __tablename__ = "message_sentiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True
    )
    overall_label: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    overall_label_score: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    score_positive: Mapped[Optional[float]] = mapped_column(Float)
    score_negative: Mapped[Optional[float]] = mapped_column(Float)
    score_neutral: Mapped[Optional[float]] = mapped_column(Float)
    sentences_summary: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    opinions: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    api_version: Mapped[Optional[str]] = mapped_column(String(50))
    analysis_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True
    )
    error_code: Mapped[Optional[str]] = mapped_column(String(100))
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    message: Mapped["Message"] = relationship(
        "Message",
        back_populates="sentiment",
        lazy="joined",
        passive_deletes=True
    )


class SegmentSentiment(Base):
    __tablename__ = "segment_sentiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sender_segment_id: Mapped[int] = mapped_column(
        ForeignKey("segments_sender.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True
    )
    overall_label: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    overall_label_score: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    score_positive: Mapped[Optional[float]] = mapped_column(Float)
    score_negative: Mapped[Optional[float]] = mapped_column(Float)
    score_neutral: Mapped[Optional[float]] = mapped_column(Float)
    sentences_summary: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    opinions: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    api_version: Mapped[Optional[str]] = mapped_column(String(50))
    analysis_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True
    )
    error_code: Mapped[Optional[str]] = mapped_column(String(100))
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    sender_segment: Mapped["SenderSegment"] = relationship(
        "SenderSegment",
        back_populates="sentiment",
        lazy="joined",
        passive_deletes=True
    )

# Stub table for reference purpose
class Embedding(Base):
    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(
        ForeignKey("chats.id", ondelete="CASCADE"), nullable=False
    )
    source_table: Mapped[str] = mapped_column(Text, nullable=False)
    source_id: Mapped[int] = mapped_column(Integer, nullable=False)
    
    embedding: Mapped[List[float]] = mapped_column(Vector(1536), nullable=False) 
    
    text_excerpt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    chunk_metadata: Mapped[Optional[dict]] = mapped_column(JSON, default={})
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        # Match the 'embeddings_chat_id_idx' from your manual migration
        Index('embeddings_chat_id_idx', 'chat_id'),

        # Match the composite index 'embeddings_chat_source_idx'
        Index('embeddings_chat_source_idx', 'chat_id', 'source_table', 'source_id'),

        # Match the IVFFlat vector index
        Index(
            'embeddings_embedding_ivfflat',
            'embedding',
            postgresql_using='ivfflat',
            postgresql_with={'lists': 100},
            postgresql_ops={'embedding': 'vector_l2_ops'}
        ),
    )

class ConversationHistory(Base):
    __tablename__ = "conversation_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(
        ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False) # "user" or "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[Optional[List[Dict]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    chat: Mapped["Chat"] = relationship(
        "Chat",
        back_populates="history",
        lazy="selectin",
        passive_deletes=True
    )
