from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.schemas import DashboardData, DashboardFilters
from src.app.db.session import get_db 
from src.app import crud, models
from src.app.services.dashboard_service import get_dashboard_data
from src.app.security import get_current_user
from src.app.limiter import limiter

router = APIRouter()

def get_dashboard_filters(
    start_date: Optional[datetime] = Query(
        None, 
        description="Start date for filtering (ISO 8601 format)"
    ),
    end_date: Optional[datetime] = Query(
        None, 
        description="End date for filtering (ISO 8601 format)"
    ),
    participants: Optional[List[str]] = Query(
        None, 
        description="List of participant names to include (can be used multiple times)"
    ),
    time_period: Optional[str] = Query(
        None, 
        description="Time period to filter by",
        # This enum adds validation and OpenAPI docs for free
        enum=["Morning", "Afternoon", "Evening", "Night"]
    )
) -> DashboardFilters:
    """
    Dependency that parses GET query parameters into a
    DashboardFilters Pydantic model.
    """
    return DashboardFilters(
        start_date=start_date,
        end_date=end_date,
        participants=participants,
        time_period=time_period
    )


# --- Phase 1: Endpoint Definition ---

@router.get(
    "/chats/{chat_id}",
    response_model=DashboardData,
    summary="Get Chat Dashboard Analytics"
)
@limiter.limit("20/minute")
async def read_chat_dashboard(
    request: Request,
    chat_id: int,
    db: AsyncSession = Depends(get_db),
    filters: DashboardFilters = Depends(get_dashboard_filters),
    current_user: models.User = Depends(get_current_user)
):
    """
    Retrieve a comprehensive analytics dashboard for a specific chat.
    
    This endpoint provides all data necessary to populate the dashboard,
    with optional filters for:
    
    - **Date Range**: `start_date` and `end_date`
    - **Participants**: `participants` (e.g., ?participants=User+A&participants=User+B)
    - **Time of Day**: `time_period`
    """
    
    chat = await crud.get_chat(db, chat_id=chat_id)
    if not chat or chat.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    dashboard_data = await get_dashboard_data(chat_id=chat_id, db=db, filters=filters)
    
    if not dashboard_data:
        # The service will return None if the chat_id doesn't exist
        raise HTTPException(status_code=404, detail="Chat not found or no data available.")
        
    return dashboard_data