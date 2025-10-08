from datetime import datetime, timezone
from sqlalchemy import JSON, Integer, String, DateTime, Text, ForeignKey, Float, UniqueConstraint
from sqlalchemy.orm import relationship, Mapped, mapped_column
from typing import List, Optional, Dict
from src.app.db.base import Base


class Chat(Base):
    __tablename__ = "chats"
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )
    title: Mapped[Optional[str]] = mapped_column(
        String(255), 
        nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    messages: Mapped[List["Message"]] = relationship(
        "Message", 
        back_populates="chat", 
        cascade="all, delete-orphan"
    )


class Participant(Base):
    __tablename__ = "participants"
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )
    chat_id: Mapped[int]= mapped_column(
        ForeignKey("chats.id", ondelete="CASCADE"), 
        nullable=False
    )
    name: Mapped[str] = mapped_column(
        String(255), 
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        UniqueConstraint(
            "chat_id", "name", name="uq_participant_chat_name"
        ),
    )
    messages: Mapped[List["Message"]] = relationship(
        "Message", 
        back_populates="participant", 
        # cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(
        Integer, 
        primary_key=True,
        autoincrement=True
    )
    chat_id: Mapped[int] = mapped_column(
        Integer, 
        ForeignKey(
            "chats.id", 
            ondelete="CASCADE"
        ), 
        nullable=False
    )
    participant_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("participants.id"), 
        nullable=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        index=True
    )
    content: Mapped[str] = mapped_column(
        Text, 
        nullable=False
    )
    raw: Mapped[Optional[str]] = mapped_column(
        Text, 
        nullable=False
    )
    
    chat: Mapped["Chat"] = relationship(
        "Chat", 
        back_populates="messages"
    )
    participant: Mapped[Optional["Participant"]] = relationship(
        "Participant", 
        back_populates="messages"
    )
    sentiment: Mapped[Optional["Sentiment"]] = relationship(
        "Sentiment", 
        back_populates="message", 
        uselist=False, 
        cascade="all, delete-orphan"
    )


class Sentiment(Base): 
    __tablename__ = "sentiments"
    
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )
    message_id: Mapped[int] = mapped_column(
        ForeignKey(
            "messages.id",
            ondelete="CASCADE"
        ),
        unique=True,
        nullable=False
    )
    overall_label: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )
    score_positive: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )
    score_negative: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )
    score_neutral: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    sentences_summary: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True
    )
    opinions: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True
    )

    api_version: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True
    )
    analysis_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )

    error_code: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )   
    message: Mapped["Message"] = relationship(
        "Message",
        back_populates="sentiment", 
        lazy="joined"
    )