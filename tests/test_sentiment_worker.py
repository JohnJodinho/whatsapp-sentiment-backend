# tests/test_sentiment_worker.py
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
from sqlalchemy import select
from unittest.mock import patch, AsyncMock

from src.app import crud, models, schemas
from app.services.sentiment_worker1 import queue_sentiment_analysis

@pytest.mark.asyncio
async def test_sentiment_worker_completes_successfully(db_session: AsyncSession):
    """
    Tests sentiment_worker
    """
    chat = await crud.create_chat(
        db_session, "Worker Test Chat", should_commit=False
    )
    
    participants = await crud.bulk_insert_participants(
        db_session, chat.id, ["Bob"], should_commit=False
    )
    participant = participants[0]

    message_to_score = {
        "chat_id": chat.id,
        "participant_id": participant.id,
        "timestamp": datetime.now(),
        "content": "I am so happy today, this is wonderful!",
        "raw": "raw text"
    }
    await crud.bulk_insert_messages(db_session, [message_to_score], should_commit=False)

    time_segment = await crud.create_time_segment(
        db=db_session,
        chat_id=chat.id,
        time_segment_details={
            "start_time": datetime.now(),
            "end_time": datetime.now(),
            "duration_minutes": 0,
            "message_count": 1,
        },
        should_commit=False
    )
    
    await crud.create_sender_segment(
        db=db_session,
        time_segment_id=time_segment.id,
        sender_id=participant.id,
        sender_details={
            "message_count": 1,
            "combined_text": "I am so happy today, this is wonderful!"
        },
        should_commit=False
    )

    await db_session.commit()
    
    # Start ACT.. when worker asks for a session, it gets test session
    mock_session_context = AsyncMock()
    mock_session_context.__aenter__.return_value = db_session

    with patch("src.app.services.sentiment_worker.AsyncSessionLocal", return_value=mock_session_context):
        await queue_sentiment_analysis(chat.id)

    
    message_sentiments_result = await db_session.execute(
        select(models.MessageSentiment)  
    )
    assert len(message_sentiments_result.scalars().all()) == 1

    segment_sentiments_result = await db_session.execute(
        select(models.SegmentSentiment) 
    )
    assert len(segment_sentiments_result.scalars().all()) == 1

    # Check that the chat status was updated to 'completed'
    updated_chat = await crud.get_chat(db_session, chat.id)
    assert updated_chat.sentiment_status == schemas.SentimentStatusEnum.completed