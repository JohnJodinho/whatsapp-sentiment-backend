from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.future import select
from sqlalchemy import insert, func, delete, select
from sqlalchemy.orm import selectinload, joinedload
from typing import List, Optional, Dict, Tuple, AsyncGenerator
from src.app import models, schemas
from src.app.utils.raw_txt_parser import CleanedMessage
from src.app.utils.message_analytics import compute_message_analytics
from datetime import date, datetime, time
import uuid

TIME_PERIOD_BUCKETS = {
    "Morning": [6, 7, 8, 9, 10, 11],
    "Afternoon": [12, 13, 14, 15, 16],
    "Evening": [17, 18, 19, 20],
    "Night": [21, 22, 23, 0, 1, 2, 3, 4, 5]
}


async def create_chat(db: AsyncSession, owner_id: uuid.UUID,  title: Optional[str] = None, should_commit: bool = True) -> models.Chat:
    new_chat = models.Chat(title=title, owner_id=owner_id)
    db.add(new_chat)
    await db.flush()

    if should_commit:
        await db.commit()
        await db.refresh(new_chat)
    return new_chat


async def get_chat(db: AsyncSession, chat_id: int) -> Optional[models.Chat]:
    result = await db.execute(select(models.Chat).where(models.Chat.id == chat_id))
    return result.scalars().first()

      

async def get_all_chats(db: AsyncSession) -> List[models.Chat]:
    result = await db.execute(select(models.Chat).order_by(models.Chat.created_at.desc()))
    return result.scalars().all()

async def get_chat_by_name(db: AsyncSession, chat_name: str) -> Optional[models.Chat]:
    chat_res = await db.execute(
        select(models.Chat).where(models.Chat.title == chat_name.strip())
    )
    return chat_res.scalars().first()




async def delete_chat(db: AsyncSession, chat_id: int, should_commit: bool = True) -> bool:
    chat = await get_chat(db, chat_id=chat_id)
    if chat:
        await db.execute(delete(models.Chat).where(models.Chat.id == chat_id))
        if should_commit:
            await db.commit()
        return True
    return False



async def request_chat_cancel(db: AsyncSession, chat_id: int) -> bool:
    result = await db.execute(select(models.Chat).filter(models.Chat.id == chat_id))
    chat = result.scalars().first()
    if chat:
        chat.cancel_requested = True
        await db.commit()
        return True
    return False


async def is_chat_cancelled(db: AsyncSession, chat_id: int) -> bool:
    result = await db.execute(select(models.Chat.cancel_requested).filter(models.Chat.id == chat_id))
    cancelled = result.scalar()
    return bool(cancelled)

async def update_chat_status(db: AsyncSession, chat_id: int, status: str):
    chat = await get_chat(db, chat_id)
    if chat:
        chat.sentiment_status = status
        db.add(chat)
        await db.commit()

async def update_chat_embedding_status(db: AsyncSession, chat_id: int, status: str, should_commit: bool = True):
    chat = await get_chat(db, chat_id)
    if chat:
        chat.embeddings_status = status
        db.add(chat)
        if should_commit:
            await db.commit()


async def get_chat_embedding_status(db: AsyncSession, chat_id: int) -> Optional[str]:
    """
    Quickly get just the embeddings_status of a chat.
    """
    result = await db.execute(
        select(models.Chat.embeddings_status)
        .where(models.Chat.id == chat_id)
    )
    return result.scalar_one_or_none()

async def get_chat_status(db: AsyncSession, chat_id: int) -> Optional[str]:
    """
    Quickly get just the sentiment_status of a chat.
    """
    result = await db.execute(
        select(models.Chat.sentiment_status)
        .where(models.Chat.id == chat_id)
    )
    return result.scalar_one_or_none()

async def get_conversation_history(
    db: AsyncSession, 
    chat_id: int, 
    limit: int = 6 # Fetch last 6 turns (3 pairs) to save tokens
) -> List[models.ConversationHistory]:
    stmt = (
        select(models.ConversationHistory)
        .where(models.ConversationHistory.chat_id == chat_id)
        .order_by(models.ConversationHistory.created_at.desc()) # Get newest first
        .limit(limit)
    )
    result = await db.execute(stmt)
    # Reverse to return chronological order (Oldest -> Newest)
    return result.scalars().all()[::-1] 

async def add_conversation_turn(
    db: AsyncSession, 
    chat_id: int, 
    user_q: str, 
    ai_a: str,
    sources: Optional[List[dict]] = None
):
    """Saves both user query and AI response in a transaction."""
    user_msg = models.ConversationHistory(chat_id=chat_id, role="user", content=user_q, sources=None)
    ai_msg = models.ConversationHistory(chat_id=chat_id, role="assistant", content=ai_a, sources=sources)
    db.add_all([user_msg, ai_msg])
    await db.commit()

async def get_messages_batch(
    db: AsyncSession,
    chat_id: int,
    limit: int,
    offset: int,
    min_date: Optional[datetime] = None
) -> List[models.Message]:
    """
    Fetches a batch of messages for a chat with optional filtering by minimum date.
    """
    stmt = (
        select(models.Message)
        .where(models.Message.chat_id == chat_id)
    )

    # Optional min timestamp
    if min_date:
        stmt = stmt.where(models.Message.timestamp >= min_date)

    # Pagination + ordering
    stmt = (
        stmt.order_by(models.Message.timestamp.desc())
            .limit(limit)
            .offset(offset)
    )

    result = await db.execute(stmt)
    return result.scalars().all()











async def clear_conversation_history(db: AsyncSession, chat_id: int):

    await db.execute(
        delete(models.ConversationHistory)
        .where(models.ConversationHistory.chat_id == chat_id)
    )
    await db.commit()

async def get_full_conversation_history(
    db: AsyncSession, 
    chat_id: int
) -> List[models.ConversationHistory]:
    """
    Fetches the complete history for a chat session to restore the UI state.
    Ordered purely chronologically (Oldest -> Newest).
    """
    stmt = (
        select(models.ConversationHistory)
        .where(models.ConversationHistory.chat_id == chat_id)
        .order_by(models.ConversationHistory.created_at.asc())
    )
    result = await db.execute(stmt)
    return result.scalars().all()


async def create_participant(db: AsyncSession, chat_id: int, name: str, should_commit: bool = True) -> models.Participant:
    new_participant = models.Participant(chat_id=chat_id, name=name)
    db.add(new_participant)
    await db.flush()
    if should_commit:
        await db.commit()
        await db.refresh(new_participant)
    return new_participant


async def bulk_insert_participants(db: AsyncSession, chat_id: int, names: List[str], should_commit: bool = True) -> List[models.Participant]:
    # Fetch existing participants to avoid duplicates
    existing_q = await db.execute(select(models.Participant).where(models.Participant.chat_id == chat_id))
    already_existing = {p.name: p for p in existing_q.scalars().all()}
    to_create = [models.Participant(chat_id=chat_id, name=name) for name in names if name not in already_existing]
    
    if to_create:
        db.add_all(to_create)
        await db.flush()

    if should_commit:
        await db.commit()
        for p in to_create:
            await db.refresh(p) 

    q = await db.execute(select(models.Participant).where(models.Participant.chat_id == chat_id))
    return q.scalars().all()

async def get_participants_by_chat(db: AsyncSession, chat_id: int) -> List[models.Participant]:
    result = await db.execute(select(models.Participant).where(models.Participant.chat_id == chat_id))
    return result.scalars().all()



async def create_message(db: AsyncSession, message: schemas.MessageCreate, chat_id: int, participant_id: Optional[int], should_commit: bool = True) -> models.Message:
    # Compute analytics for the message
    analytics = compute_message_analytics(message.content)
    
    new_msg = models.Message(
        chat_id=chat_id,
        participant_id=participant_id,
        timestamp=message.timestamp,
        content=message.content,
        raw=getattr(message, "raw", None) or message.content,  # Use content if raw is not provided
        word_count=analytics['word_count'],
        emojis_count=analytics['emojis_count'],
        links_count=analytics['links_count'],
        is_question=analytics['is_question'],
        is_media=analytics['is_media']
    )
    db.add(new_msg)
    await db.flush()
    if should_commit:  
        await db.commit()
        await db.refresh(new_msg)
    return new_msg


async def bulk_insert_messages(
    db: AsyncSession,
    messages: List[dict],
    should_commit: bool = True
) -> List[int]:
    if not messages:
        return []

    try:
        CHUNK_SIZE = 2500
        inserted_ids = []

        for i in range(0, len(messages), CHUNK_SIZE):
            chunk = messages[i:i + CHUNK_SIZE]
            stmt = insert(models.Message).values(chunk).returning(models.Message.id)
            result = await db.execute(stmt)
            inserted_ids.extend(result.scalars().all())

        if should_commit:
            await db.commit()

        return inserted_ids

    except SQLAlchemyError:
        await db.rollback()
        raise



async def get_messages_by_chat(db: AsyncSession, chat_id: int) -> List[models.Message]:
    result = await db.execute(select(models.Message).where(models.Message.chat_id == chat_id))
    return result.scalars().all()

async def get_all_sender_segments_for_chat(db: AsyncSession, chat_id: int) -> List[models.SenderSegment]:
    stmt = (
        select(models.SenderSegment)
        .join(models.TimeSegment)
        .where(models.TimeSegment.chat_id == chat_id)
    )
    result = await db.execute(stmt)
    return result.scalars().all()

async def get_segments_batch(
    db: AsyncSession,
    chat_id: int,
    limit: int,
    offset: int,
    min_date: Optional[datetime] = None
) -> List[models.SenderSegment]:

    stmt = (
        select(models.SenderSegment)
        .join(models.TimeSegment)
        .where(models.TimeSegment.chat_id == chat_id)
    )

    if min_date:
        stmt = stmt.where(models.TimeSegment.start_time >= min_date)

    stmt = (
        stmt.order_by(models.TimeSegment.start_time.desc())
            .limit(limit)
            .offset(offset)
    )

    result = await db.execute(stmt)
    return result.scalars().all()


# Issues here to handle later
async def delete_message(db: AsyncSession, message_id: int, should_commit: bool = True) -> bool:
    
    try:
        result = await db.execute(select(models.Message).where(models.Message.id == message_id))
        message = result.scalars().first()
        if not message:
            return False
        
        participant_id = message.participant_id
        chat_id = message.chat_id

        await db.delete(message)

        if participant_id:
            other_messages_query = await db.execute(
                select(models.Message).where(
                    models.Message.chat_id == chat_id,
                    models.Message.participant_id == participant_id
                )
            )

            rem_messages = other_messages_query.scalars().first()
            if not rem_messages:
                participant = await db.get(models.Participant, participant_id)
                if participant:
                    await db.delete(participant)
        if should_commit:
            await db.commit()
                    
        return True
    except Exception:
        if should_commit:
            await db.rollback()
        raise

async def create_sender_segment(
    db: AsyncSession,
    time_segment_id: int,
    sender_id: int,
    sender_details: dict,
    should_commit: bool = True
):
    seg = models.SenderSegment(
        time_segment_id=time_segment_id,
        sender_id=sender_id,
        message_count=sender_details.get("message_count"),
        combined_text=sender_details.get("combined_text", "")
    )
    db.add(seg)
    await db.flush()
    if should_commit:
        await db.commit()
        await db.refresh(seg)
    return seg

    

async def create_time_segment(
    db: AsyncSession,
    chat_id: int,
    time_segment_details: dict,
    should_commit: bool = True
) -> models.TimeSegment:
    seg = models.TimeSegment(
        chat_id=chat_id,
        start_time=time_segment_details.get("start_time"),
        end_time=time_segment_details.get("end_time"),
        duration_minutes=time_segment_details.get("duration_minutes"),
        message_count=time_segment_details.get("message_count"),
        summary=time_segment_details.get("summary", None)
    )
    db.add(seg)
    await db.flush()
    if should_commit:
        await db.commit()
        await db.refresh(seg)
    return seg


async def bulk_insert_segments(
    db: AsyncSession,
    chat_id: int,
    participants_map: dict,
    time_segments_list: List[dict]
) -> int:
    if not time_segments_list:
        return 0

    # Build all TimeSegment objects
    time_segments = [
        models.TimeSegment(
            chat_id=chat_id,
            start_time=seg.get("start_time"),
            end_time=seg.get("end_time"),
            duration_minutes=seg.get("duration_minutes"),
            message_count=seg.get("message_count"),
            summary=seg.get("summary")
        )
        for seg in time_segments_list
    ]

    # Bulk add time segments
    db.add_all(time_segments)
    await db.flush()  # Assign IDs

    # Build all SenderSegment objects linked to their TimeSegments
    sender_segments = []
    for seg_obj, seg_dict in zip(time_segments, time_segments_list):
        sender_groups = seg_dict.get("sender_groups", {})
        for sender, details in sender_groups.items():
            sender_id = participants_map.get(sender)
            if not sender_id:
                continue
            sender_segments.append(models.SenderSegment(
                time_segment_id=seg_obj.id,
                sender_id=sender_id,
                message_count=details.get("message_count"),
                combined_text=details.get("combined_text", "")
            ))

    if sender_segments:
        db.add_all(sender_segments)
        await db.flush()

    return len(time_segments)

async def get_sender_segments_by_sender_id(db: AsyncSession, sender_id: int) -> List[models.SenderSegment]:
    result = await db.execute(
        select(models.SenderSegment)
        .where(models.SenderSegment.sender_id == sender_id)
    )
    return result.scalars().all()

async def get_all_time_segments(db: AsyncSession, chat_id: int) -> List[models.TimeSegment]:
    result = await db.execute(
        select(models.TimeSegment)
        .where(models.TimeSegment.chat_id == chat_id)
    )
    return result.scalars().all()

async def create_message_sentiment(
        db: AsyncSession,
        msg_id: int,
        payload: dict,
        should_commit: bool = True
) -> models.MessageSentiment:
    s = models.MessageSentiment(message_id = msg_id, **payload)
    db.add(s)
    await db.flush()

    if should_commit:
        await db.commit()
        await db.refresh(s)
    return s

async def create_segment_sentiment(
        db: AsyncSession,
        sender_segment_id: int,
        payload: dict,
        should_commit: bool = True
) -> models.SegmentSentiment:
    s = models.SegmentSentiment(sender_segment_id=sender_segment_id, **payload)
    db.add(s)
    await db.flush()

    if should_commit:
        await db.commit()
        await db.refresh(s)
    return s
    

async def get_sentiment_progress(db: AsyncSession, chat_id: int):
    """
    Efficiently gets counts using SQL func.count.
    """
    
    # Use func.count for direct SQL COUNT(*), which is instant
    total_messages_q = await db.execute(
        select(func.count(models.Message.id))
        .where(models.Message.chat_id == chat_id)
    )
    total_messages = total_messages_q.scalar_one()

    scored_msgs_q = await db.execute(
        select(func.count(models.MessageSentiment.id))
        .join(models.Message)
        .where(models.Message.chat_id == chat_id)
    )
    total_messages_scored = scored_msgs_q.scalar_one()

    total_segments_q = await db.execute(
        select(func.count(models.SenderSegment.id))
        .join(models.TimeSegment)
        .where(models.TimeSegment.chat_id == chat_id)
    )
    total_segments = total_segments_q.scalar_one()

    scored_segments_q = await db.execute(
        select(func.count(models.SegmentSentiment.id))
        .join(models.SenderSegment)
        .join(models.TimeSegment)
        .where(models.TimeSegment.chat_id == chat_id)
    )
    total_segments_scored = scored_segments_q.scalar_one()

    return {
        "messages_total": total_messages,
        "messages_scored": total_messages_scored,
        "segments_total": total_segments,
        "segments_scored": total_segments_scored
    }

async def get_chat_message_sentiments(db: AsyncSession, chat_id: int) -> Optional[List[models.MessageSentiment]]:
    stmt = select(models.MessageSentiment).join(models.Message).where(
        models.Message.chat_id == chat_id
    )
    result = await db.execute(stmt)

    return result.scalars().all()

async def get_segments_for_summarization(
    db: AsyncSession, chat_id: int
) -> List[models.TimeSegment]:
    """
    Fetches all TimeSegments for a specific chat that do not 
    yet have a summary.
    
    This function is optimized to EAGERLY load the related
    'sender_segments' for each TimeSegment in a single batch, 
    preventing the N+1 query problem in the summary service.
    """
    
    # 1. Define the query
    stmt = (
        select(models.TimeSegment)
        .where(models.TimeSegment.chat_id == chat_id)
        .where(models.TimeSegment.summary == None)  # <-- Finds work to do
        .options(
            selectinload(models.TimeSegment.sender_segments) # <-- The EFFICIENT part
        )
        .order_by(models.TimeSegment.start_time) # Process segments in order
    )
    
    # 2. Execute the query
    result = await db.execute(stmt)
    
    # 3. Return the TimeSegment objects
    # .scalars() gets the TimeSegment objects
    # .all() executes and returns the list
    segments = result.scalars().all()
    
    return segments


async def get_chat_segment_sentiments(db: AsyncSession, chat_id: int) -> Optional[List[models.SegmentSentiment]]:
    stmt = select(models.SegmentSentiment).join(models.SenderSegment).join(
        models.TimeSegment
    ).where(
        models.TimeSegment.chat_id == chat_id
    )

    result = await db.execute(stmt)
    return result.scalars().all()

async def get_sentiment_by_message(db: AsyncSession, message_id: int) -> Optional[schemas.SentimentRead]:
    result = await db.execute(select(
        models.MessageSentiment
    )
    .where(models.MessageSentiment.message_id == message_id)
    )

    if result:
        return result.scalars().first()
    return None

async def get_sentiment_by_segment(db: AsyncSession, segment_id: int) -> Optional[schemas.SentimentRead]:
    result = await db.execute(
        select(
            models.SegmentSentiment
        )
        .where(models.SegmentSentiment.sender_segment_id == segment_id)
    )
    if result: return result.scalars().first()
    return None

# Ingest Cleaned Chat
async def ingest_cleaned_chat(
    db: AsyncSession,
    owner_id: uuid.UUID,
    chat_name: str,
    cleaned_messages: List["CleanedMessage"],
    segments_list: List[dict]
) -> models.Chat:
    try:
        # Create chat
        chat = await create_chat(db, owner_id=owner_id, title=chat_name, should_commit=False)

        # Insert participants
        all_chat_participants = {m.sender for m in cleaned_messages if m.sender}
        participants = await bulk_insert_participants(
            db=db,
            chat_id=chat.id,
            names=all_chat_participants,
            should_commit=False
        )

        participants_map = {p.name: p.id for p in participants}

        # Insert time and sender segments (batched)
        await bulk_insert_segments(
            db=db,
            chat_id=chat.id,
            participants_map=participants_map,
            time_segments_list=segments_list
        )
        # Insert messages in bulk (batched)
        # We must compute analytics for each message first
        message_dicts = []
        for msg in cleaned_messages:
            # Skip messages without a sender (e.g., system messages)
            if not msg.sender:
                continue
            
            # Compute analytics, just like in create_message
            analytics = compute_message_analytics(msg.text)
            
            message_dicts.append(
                {
                    "chat_id": chat.id,
                    "participant_id": participants_map.get(msg.sender),
                    "timestamp": msg.timestamp,
                    "content": msg.text,
                    # Match the logic from create_message: use content if raw is missing
                    "raw": getattr(msg, "raw", None) or msg.text,
                    
                    # Add the missing analytics fields
                    "word_count": analytics['word_count'],
                    "emojis_count": analytics['emojis_count'],
                    "links_count": analytics['links_count'],
                    "is_question": analytics['is_question'],
                    "is_media": analytics['is_media']
                }
            )

        await bulk_insert_messages(db, message_dicts, should_commit=False)

        # Mark chat status
        chat.sentiment_status = "pending"

        # Single final commit
        await db.commit()
        await db.refresh(chat)
        return chat

    except Exception:
        await db.rollback()
        raise
async def get_unscored_messages(
        db: AsyncSession,
        chat_id: int,
        limit: Optional[int] = None
):
    q = await db.execute(
        select(models.Message)
        .where(models.Message.chat_id==chat_id)
            .outerjoin(models.MessageSentiment)
            .where(models.MessageSentiment.id == None)
    )

    res = q.scalars().all()

    return res if limit is None else res[:limit]

async def get_unscored_sender_segments(
        db: AsyncSession,
        chat_id: int
):
    q = await db.execute(
        select(models.SenderSegment).join(models.TimeSegment)
        .where(models.TimeSegment.chat_id == chat_id)
        .outerjoin(models.SegmentSentiment)
        .where(models.SegmentSentiment.id == None)
    )

    res = q.scalars().all()

    return res


def _get_filtered_query(
    chat_id: int,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    participant_names: Optional[List[str]] = None,
    time_period: Optional[str] = None
):
    stmt = select(models.Message).where(models.Message.chat_id == chat_id)

    if date_from:
        stmt = stmt.where(func.date(models.Message.timestamp) >= date_from)
    if date_to:
        stmt = stmt.where(func.date(models.Message.timestamp) <= date_to)
    if participant_names:
        stmt = stmt.join(models.Participant).where(models.Participant.name.in_(participant_names))
    if time_period and time_period in TIME_PERIOD_BUCKETS:
        hours = TIME_PERIOD_BUCKETS[time_period]
        stmt = stmt.where(func.extract('hour', models.Message.timestamp).in_(hours))

    return stmt


# --- REFACTORED STREAMING FUNCTION ---
async def stream_unscored_messages(
    db: AsyncSession,
    chat_id: int
) -> AsyncGenerator[models.Message, None]:
    """
    Streams unscored messages one by one using a server-side cursor.
    This avoids loading all messages into memory.
    """
    stmt = (
        select(models.Message)
        .where(models.Message.chat_id == chat_id)
        .outerjoin(models.MessageSentiment)
        .where(models.MessageSentiment.id == None)
        .order_by(models.Message.timestamp) # Good practice to have an order
    )
    
    stream = await db.stream_scalars(stmt)
    async for msg in stream:
        yield msg

# --- REFACTORED STREAMING FUNCTION ---
async def stream_unscored_sender_segments(
    db: AsyncSession,
    chat_id: int
) -> AsyncGenerator[models.SenderSegment, None]:
    """
    Streams unscored sender segments one by one.
    """
    stmt = (
        select(models.SenderSegment)
        .join(models.TimeSegment)
        .where(models.TimeSegment.chat_id == chat_id)
        .outerjoin(models.SegmentSentiment)
        .where(models.SegmentSentiment.id == None)
    )
    
    stream = await db.stream_scalars(stmt)
    async for seg in stream:
        yield seg