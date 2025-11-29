from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks, status, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import List

import shutil
import aiofiles
import os
import uuid

from src.app.config import settings
from src.app.db.session import get_db
from src.app import crud, schemas, models
from src.app.utils.raw_txt_parser import WhatsAppChatParser, CleanedMessage
from src.app.utils.pre_process import clean_messages
from src.app.utils.segment_chat import segment_by_time, group_by_sender
from src.app.utils.extract_file_name import extract_chat_title
from src.app.services.sentiment_worker import celery_app
import src.app.services.embedding_worker
from src.app.security import get_current_user
from src.app.limiter import limiter

router = APIRouter()



@router.post("/whatsapp", response_model=schemas.ChatRead, status_code=status.HTTP_201_CREATED)
@limiter.limit("1/minute")
async def upload_whatsapp_chat_file(
    request: Request,
    
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file. Please upload a .txt file.")
    
    file_id = str(uuid.uuid4())
    temp_filename =    f"upload_{file_id}.txt"

    try:
        
        async with aiofiles.open(temp_filename, 'wb') as out_file:
            while content := await file.read(1024 * 1024):  
                await out_file.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="File upload failed during write."
        )
    finally:
        await file.close()

    try:
        chat_title = extract_chat_title(file.filename)
        parser = WhatsAppChatParser(dayfirst=True)
 
        
        # Run blocking synchronous functions in a separate thread
        raw_messages = await run_in_threadpool(parser.parse_file, temp_filename)
        parsed: List[CleanedMessage] = await run_in_threadpool(clean_messages, raw_messages)

        if not parsed or all(msg.sender is None for msg in parsed):
             raise HTTPException(
                 status_code=400, 
                 detail="Parsed chat is empty or invalid format."
            )
        time_segments: List[dict] = segment_by_time(parsed)
        segments: List[dict] = group_by_sender(time_segments)
  
        chat_result = await crud.ingest_cleaned_chat(
            db,
            owner_id=current_user.id,
            chat_name=chat_title,
            cleaned_messages=parsed,
            segments_list= segments
        )
        celery_app.send_task("analyze_sentiment", args=[chat_result.id])
        celery_app.send_task("generate_embeddings", args=[chat_result.id])

        

        return chat_result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Processing failed: {str(e)}"
        )
    finally:
        
        if os.path.exists(temp_filename):
            os.remove(temp_filename)