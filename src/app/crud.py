from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import insert, func
from typing import List, Optional
from src.app import models, schemas
from src.app.utils.raw_txt_parser import CleanedMessage



async def create_chat(db: AsyncSession, title: Optional[str] = None) -> models.Chat:
    new_chat = models.Chat(title=title)
    db.add(new_chat)
    await db.commit()
    await db.refresh(new_chat)
    return new_chat


async def get_chat(db: AsyncSession, chat_id: int) -> Optional[models.Chat]:
    result = await db.execute(select(models.Chat).where(models.Chat.id == chat_id))
    return result.scalars().first()

      

async def get_all_chats(db: AsyncSession) -> List[models.Chat]:
    result = await db.execute(select(models.Chat).order_by(models.Chat.created_at.desc()))
    return result.scalars().all()


async def delete_chat(db: AsyncSession, chat_id: int) -> bool:
    chat = await get_chat(db, chat_id=chat_id)
    if chat:
        await db.delete(chat)
        await db.commit()
        return True
    return False


async def create_participant(db: AsyncSession, chat_id: int, name: str) -> models.Participant:
    new_participant = models.Participant(chat_id=chat_id, name=name)
    db.add(new_participant)
    await db.commit()
    await db.refresh(new_participant)
    return new_participant


async def bulk_insert_participants(db: AsyncSession, chat_id: int, names: List[str]) -> List[models.Participant]:
    # Fetch existing participants to avoid duplicates
    existing_q = await db.execute(select(models.Participant).where(models.Participant.chat_id == chat_id))
    already_existing = {p.name: p for p in existing_q.scalars().all()}

    to_create = [models.Participant(chat_id=chat_id, name=name) for name in names if name not in already_existing]
    db.add_all(to_create)
    await db.commit()

    q = await db.execute(select(models.Participant).where(models.Participant.chat_id == chat_id))
    return q.scalars().all()

async def get_participants_by_chat(db: AsyncSession, chat_id: int) -> List[models.Participant]:
    result = await db.execute(select(models.Participant).where(models.Participant.chat_id == chat_id))
    return result.scalars().all()



async def create_message(db: AsyncSession, message: schemas.MessageCreate, chat_id: int, participant_id: Optional[int]) -> models.Message:
    new_msg = models.Message(
        chat_id=chat_id,
        participant_id=participant_id,
        timestamp=message.timestamp,
        content=message.content,
        raw=getattr(message, "raw", message.content)  # Use content if raw is not provided
    )

    db.add(new_msg)
    await db.commit()
    await db.refresh(new_msg)
    return new_msg


async def bulk_insert_messages(db: AsyncSession, messages: List[dict]) -> List[models.Message]:
    if not messages:
        return []
    message_objects = [models.Message(**msg) for msg in messages]
    db.add_all(message_objects)
    await db.commit()
    await db.refresh(message_objects)
    return message_objects


async def get_messages_by_chat(db: AsyncSession, chat_id: int) -> List[models.Message]:
    result = await db.execute(select(models.Message).where(models.Message.chat_id == chat_id))
    return result.scalars().all()

async def delete_message(db: AsyncSession, message_id: int) -> bool:
    
    async with db.begin():
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
                    
    return True



async def create_sentiment(
        db: AsyncSession,
        sentiment: schemas.SentimentCreate
) -> models.Sentiment:
    new_sentiment = models.Sentiment(
        message_id = sentiment.message_id,
        overall_label = sentiment.overall_label,
        score_positive = sentiment.score_positive,
        score_negative = sentiment.score_negative,
        score_neutral = sentiment.score_neutral,
        sentences_summary = sentiment.sentences_summary,
        opinions = sentiment.opinions,
        api_version = sentiment.api_version,
        analysis_timestamp = sentiment.analysis_timestamp,
        error_code = sentiment.error_code,
        error_message = sentiment.error_message
    )

    db.add(new_sentiment)
    await db.commit()
    await db.refresh(new_sentiment)
    return new_sentiment

async def get_sentiment_by_message(db: AsyncSession, message_id: int) -> Optional[models.Sentiment]:
    result = await db.execute(select(
        models.Sentiment
    )
    .where(models.Sentiment.message_id == message_id)
    )

    if result:
        return result.scalars().first()

# Ingest Cleaned Chat
async def ingest_cleaned_chat(
    db: AsyncSession,
    chat_name: str,
    cleaned_messages: List[CleanedMessage]
):
    # Insert chat record
    chat = await create_chat(db, title=chat_name)

    # Create participants
    all_chat_participants = {m.sender for m in cleaned_messages if m.sender}
    participants = await bulk_insert_participants(
        db=db, 
        chat_id=chat.id, 
        names=all_chat_participants
    )

    participants_map = {p.name: p.id for p in participants}

    

    # Bulk insert messages 
    message_dicts = []
    for msg in cleaned_messages:
        if not msg.sender:
            continue
        message_dicts.append({
            "chat_id": chat.id,
            "participant_id": participants_map.get(msg.sender),
            "timestamp": msg.timestamp,
            "content": msg.text,
            "raw": getattr(msg, "raw", None)
        })

    messages = await bulk_insert_messages(db, message_dicts)


    return {
        "chat_id": chat.id,
        "participants_number": len(list(participants_map)),
        "messages_number": len(messages) 
    }

# Other Sentiment-based heavy analysis CRUDs here

