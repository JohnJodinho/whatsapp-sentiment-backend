from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.app import schemas, crud, models
from src.app.db.session import get_db 
from src.app.services.sentiment_dashboard_service import get_sentiment_dashboard_data
from src.app.security import get_current_user
from src.app.limiter import limiter

router = APIRouter()

def get_sentiment_dashboard_filters(
    start_date: Optional[datetime] = Query(None, description="Start date for filtering (ISO 8601 format)"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering (ISO 8601 format)"),
    participants: Optional[List[str]] = Query(None, description="List of participant names to include"),
    time_period: Optional[str] = Query(None, description="time period to filter by", enum=["Morning", "Afternoon", "Evening", "Night"]),
    granularity: schemas.SentimentGranularity = Query(schemas.SentimentGranularity.message, description="Granularity of analysis (message or segment)"),
    sentiment_types: Optional[List[schemas.SentimentLabel]] = Query(None, description="List of sentiment types to include")
) -> schemas.SentimentDashboardFilters:
    """
    Dependency that parses GET query parameters into a
    SentimentDashboardFilters Pydantic model.
    """
    active_sentiment_types = sentiment_types or [
        schemas.SentimentLabel.Positive,
        schemas.SentimentLabel.Negative,
        schemas.SentimentLabel.Neutral
    ]
    return schemas.SentimentDashboardFilters(
        start_date=start_date,
        end_date=end_date,
        participants=participants,
        time_period=time_period,
        granularity=granularity,
        sentiment_types=active_sentiment_types # Default to all
    )


@router.get(
    "/chats/{chat_id}",
    response_model=schemas.SentimentDashboardData,
    summary="Get Chat Sentiment Analytics Dashboard"
)
@limiter.limit("15/minute")
async def read_chat_sentiment_dashboard(
    request: Request,
    chat_id: int,
    db: AsyncSession = Depends(get_db),
    filters: schemas.SentimentDashboardFilters = Depends(get_sentiment_dashboard_filters),
    current_user: models.User = Depends(get_current_user)
):
    """
    Retrieve a comprehensive sentiment analytics dashboard for a specific chat.
    
    This endpoint provides all data necessary to populate the sentiment dashboard,
    with optional filters for:
    
    - **Date Range**: `start_date` and `end_date`
    - **Participants**: `participants`
    - **Time of Day**: `time_period`
    - **Granularity**: `granularity` (message or segment)
    - **Sentiment Types**: `sentiment_types` (Positive, Negative, Neutral)
    """
    chat = await crud.get_chat(db, chat_id)
    if not chat or chat.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    dashboard_data = await get_sentiment_dashboard_data(
        chat_id=chat_id, db=db, filters=filters
    )
    
    if not dashboard_data:
        raise HTTPException(
            status_code=404, 
            detail="Chat not found or no sentiment data available for this query."
        )
        
    return dashboard_data