
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime, timedelta

from src.app import crud, models


@pytest.mark.asyncio
async def test_delete_existing_chat_and_cascades(db_session: AsyncSession):
    chat_to_delete = await crud.create_chat(
        db_session, "Chat to be Deleted", should_commit=False
    )

    participants = await crud.bulk_insert_participants(
        db_session, chat_to_delete.id, ["TempUser"], should_commit=False
    )

    message_to_delete = {
        "chat_id": chat_to_delete.id,
        "participant_id": participants[0].id,
        "timestamp": datetime.now(),
        "content": "This message should be deleted.",
        "raw": "raw text"
    }

    message_ids = await crud.bulk_insert_messages(
        db_session, [message_to_delete], should_commit=False
    )

    start_time = datetime.now()
    end_time = datetime.now()  + timedelta(minutes=30)
    duration_minutes = int((end_time - start_time).total_seconds() / 60)

    time_segment = await crud.create_time_segment(
        db=db_session,
        chat_id=chat_to_delete.id, 
        time_segment_details={
            "start_time": start_time,
            "end_time": end_time,
            "duration_minutes": duration_minutes,
            "message_count": 1
        },
        should_commit=False                             
    )

    sender_segment = await crud.create_sender_segment(
        db=db_session,
        time_segment_id=time_segment.id,
        sender_id=participants[0].id,
        sender_details={
            "message_count": 1,
            "combined_text": "This message should be deleted.",

        },
        should_commit=False
    )
    payload = {
        "overall_label": "negative",
        "overall_label_score": 0.894,
        "score_negative": None,
        "score_neutral": None,
        "score_positive": None,
        "sentences_summary": None,
        "opinions": None,
        "api_version": "onnx_local",
        "analysis_timestamp": None,
        "error_code": None,
        "error_message": None
    }
    msg_sentiment = await crud.create_message_sentiment(
        db=db_session,
        msg_id=message_ids[0],
        payload=payload,
        should_commit=False
    )

    segment_sentiment = await crud.create_segment_sentiment(
        db=db_session,
        sender_segment_id=sender_segment.id,
        payload=payload,
        should_commit=False
    )


    await db_session.commit()
    await db_session.refresh(chat_to_delete)


    # Oya let's check

    # Messages
    messages_count_before = await db_session.scalar(
        select(func.count()).select_from(
            models.Message
        )
        .where(models.Message.chat_id == chat_to_delete.id)
    )

    assert messages_count_before == 1

    # Sender segments
    sender_segments_count_before = await db_session.scalar(
        select(func.count()).select_from(
            models.SenderSegment
        )
        .where(models.SenderSegment.sender_id==participants[0].id)
    )

    assert sender_segments_count_before == 1

    # Time Segments
    time_segments_count_before = await db_session.scalar(
        select(func.count()).select_from(
            models.TimeSegment
        )
        .where(models.TimeSegment.chat_id==chat_to_delete.id)
    )

    assert time_segments_count_before == 1

    # Message Sentiments
    message_sentiments_count_before = await db_session.scalar(
        select(func.count()).select_from(
            models.MessageSentiment
        )
        .where(models.MessageSentiment.message_id==message_ids[0])
    )

    assert message_sentiments_count_before == 1

    segments_sentiment_count_before = await db_session.scalar(
        select(func.count()).select_from(
            models.SegmentSentiment
        )
        .where(models.SegmentSentiment.sender_segment_id == sender_segment.id)
    )

    assert segments_sentiment_count_before == 1


    del_result = await crud.delete_chat(db_session, chat_id=chat_to_delete.id, should_commit=True)
    assert del_result is True

    deleted_chat = await crud.get_chat(db_session, chat_id=chat_to_delete.id)
    assert deleted_chat is None

    messages_after = await crud.get_messages_by_chat(db_session, chat_id=chat_to_delete.id)
    messages_count_after = len(messages_after)

    assert messages_count_after == 0

    sender_segments_after = await crud.get_sender_segments_by_sender_id(db_session, participants[0].id)
    assert len(sender_segments_after) == 0

    time_segments_after = await crud.get_all_time_segments(db_session, chat_to_delete.id)
    assert len(time_segments_after) == 0

    messages_sentiment_after = await crud.get_chat_message_sentiments(db_session, chat_to_delete.id)
    assert len(messages_sentiment_after) == 0

    segments_sentiment_after = await crud.get_chat_segment_sentiments(
        db_session, chat_to_delete.id
    )
    assert len(segments_sentiment_after) == 0


