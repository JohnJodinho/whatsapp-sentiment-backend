import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case, Select, extract, and_, Integer, String
from sqlalchemy.dialects.postgresql import INTERVAL
import asyncio
from src.app import models, schemas, crud
from typing import List, Optional, Dict, Any, Coroutine
from sqlalchemy.orm import aliased
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta    
from collections import defaultdict

logger = logging.getLogger(__name__)

TIME_PERIOD_BUCKETS: Dict[str, List[int]] = {
    "Morning": [6, 7, 8, 9, 10, 11],
    "Afternoon": [12, 13, 14, 15, 16],
    "Evening": [17, 18, 19, 20],
    "Night": [21, 22, 23, 0, 1, 2, 3, 4, 5]
}

DAY_OF_WEEK_MAP: List[str] = [
    "sun", "mon", "tue", "wed", "thu", "fri", "sat"
]

async def _create_sentiment_base_query(
    chat_id: int, 
    db: AsyncSession, 
    filters: schemas.SentimentDashboardFilters
) -> Select:
    """
    Creates a dynamic, filtered SQLAlchemy Select statement
    that serves as the common base for all sentiment analytic queries.
    
    This is the core of our service. It dynamically joins either
    MessageSentiments or SegmentSentiments and applies all filters.
    """
    logger.debug(f"Creating sentiment base query for chat {chat_id} with granularity {filters.granularity}")
    
    # --- 1. Define Aliases ---
    participant = aliased(models.Participant)

    # --- 2. Build the query based on Granularity ---
    if filters.granularity == schemas.SentimentGranularity.message:
        sentiment = aliased(models.MessageSentiment)
        message = aliased(models.Message)
        
        # --- DEFINE SOURCE COLUMNS ---
        # These are the *original* columns we will filter on
        timestamp_col = message.timestamp
        sender_col = participant.name
        label_col = sentiment.overall_label
        
        stmt = (
            select(
                sentiment.id.label("sentiment_id"),
                label_col.label("label"), # Use source col
                sentiment.overall_label_score.label("score"),
                message.id.label("item_id"),
                timestamp_col.label("timestamp"), # Use source col
                message.content.label("text"),
                sender_col.label("sender") # Use source col
            )
            .join(message, sentiment.message_id == message.id)
            .join(participant, message.participant_id == participant.id)
            .where(message.chat_id == chat_id)
        )
        
    elif filters.granularity == schemas.SentimentGranularity.segment:
        sentiment = aliased(models.SegmentSentiment)
        sender_segment = aliased(models.SenderSegment)
        time_segment = aliased(models.TimeSegment)

        # --- DEFINE SOURCE COLUMNS ---
        timestamp_col = time_segment.start_time
        sender_col = participant.name
        label_col = sentiment.overall_label

        stmt = (
            select(
                sentiment.id.label("sentiment_id"),
                label_col.label("label"), # Use source col
                sentiment.overall_label_score.label("score"),
                sender_segment.id.label("item_id"),
                timestamp_col.label("timestamp"), # Use source col
                sender_segment.combined_text.label("text"),
                sender_col.label("sender") # Use source col
            )
            .join(sender_segment, sentiment.sender_segment_id == sender_segment.id)
            .join(participant, sender_segment.sender_id == participant.id)
            .join(time_segment, sender_segment.time_segment_id == time_segment.id)
            .where(time_segment.chat_id == chat_id)
        )
    else:
        raise ValueError(f"Invalid granularity: {filters.granularity}")

    # --- 3. Apply Common Filters (using source columns) ---
    
    # Date Range Filter
    if filters.start_date:
        stmt = stmt.where(
            func.date(timestamp_col) >= filters.start_date.date() # <-- FIXED
        )
    if filters.end_date:
        stmt = stmt.where(
            func.date(timestamp_col) <= filters.end_date.date() # <-- FIXED
        )
        
    # Participant Filter
    if filters.participants:
        stmt = stmt.where(sender_col.in_(filters.participants)) # <-- FIXED
        
    # Time Period Filter
    if filters.time_period and filters.time_period in TIME_PERIOD_BUCKETS:
        hour_bucket = TIME_PERIOD_BUCKETS[filters.time_period]
        stmt = stmt.where(extract('hour', timestamp_col).in_(hour_bucket)) # <-- FIXED

    # Sentiment Type Filter
    if filters.sentiment_types:
        lower_case_labels = [label.value.lower() for label in filters.sentiment_types]
        stmt = stmt.where(func.lower(label_col).in_(lower_case_labels)) # <-- FIXED

    logger.debug(f"Base query compiled (filters applied).")
    return stmt


async def _get_all_participants(
        chat_id: int,
        db: AsyncSession
) -> List[str]:
    """Retrive all participant names for the given chat."""
    stmt = (
        select(models.Participant.name)
        .where(models.Participant.chat_id == chat_id)
        .distinct()
        .order_by(models.Participant.name)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def _get_sentiment_kpis(
    base_query_sub: Select,
    db: AsyncSession
) -> schemas.SentimentKpiData | None:
    """Calcs the primary KPI metrics from the filtered base query."""
    logger.debug("Calculating sentiment KPIs.")

    positive_case = case((func.lower(base_query_sub.c.label) == "positive", 1), else_=0)
    negative_case = case((func.lower(base_query_sub.c.label) == "negative", 1), else_=0)
    neutral_case = case((func.lower(base_query_sub.c.label) == "neutral", 1), else_=0)

    stmt = (
        select(
            func.count().label("total_count"),
            func.sum(positive_case).label("positive_count"),
            func.sum(negative_case).label("negative_count"),
            func.sum(neutral_case).label("neutral_count")
        ).select_from(base_query_sub)
    )

    row = (await db.execute(stmt)).one_or_none()

    if not row or not row.total_count:
        logger.warning("No sentiment data found for KPIs.")
        return schemas.SentimentKpiData(
        overallScore=0,
        positivePercent=0,
        negativePercent=0,
        neutralPercent=0,
        totalMessagesOrSegments=0
    )
    
    total = row.total_count
    positive_count = row.positive_count or 0
    negative_count = row.negative_count or 0
    neutral_count = row.neutral_count or 0


    positive_percent = (positive_count / total) * 100 if total > 0 else 0
    negative_percent = (negative_count / total) * 100 if total > 0 else 0
    neutral_percent = (neutral_count / total) * 100 if total > 0 else 0

    overall_score = positive_percent - negative_percent

    logger.debug(f"Sentiment KPIs calculated: total={total}, score={overall_score:.2f}")

    return schemas.SentimentKpiData(
        overallScore=round(overall_score, 1),
        positivePercent=round(positive_percent, 1),
        negativePercent=round(negative_percent, 1),
        neutralPercent=round(neutral_percent, 1),
        totalMessagesOrSegments=total
    )

def _start_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)

def _start_of_week(dt: datetime) -> datetime:
    start_of_day = _start_of_day(dt)
    days_to_subtract = (start_of_day.weekday() + 1) % 7
    return start_of_day - timedelta(days=days_to_subtract)

def _start_of_month(dt: datetime) -> datetime:
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


async def _get_sentiment_trend(
    base_query_sub: Select, 
    db: AsyncSession
) -> List[schemas.TrendDataPoint]:
    """
    Fetches the time-series data for the sentiment trend chart.
    
    Dynamically groups by day, week, or month based on the
    total time range of the filtered data.
    """
    logger.debug("Calculating sentiment trend data.")
    
    # 1. Find the true min/max date range
    # ... (This logic is correct and remains unchanged) ...
    range_stmt = select(
        func.min(base_query_sub.c.timestamp).label("min_ts"),
        func.max(base_query_sub.c.timestamp).label("max_ts")
    )
    range_result = (await db.execute(range_stmt)).one_or_none()

    if not range_result or not range_result.min_ts:
        logger.debug("No data found for trend, returning empty list.")
        return []

    min_ts: datetime = range_result.min_ts
    max_ts: datetime = range_result.max_ts
    
    # 2. Choose aggregation level
    # ... (This logic is correct and remains unchanged) ...
    total_days = (max_ts - min_ts).days
    trunc_level: str
    interval: timedelta | relativedelta
    python_date_trunc: callable
    
    if total_days <= 90:
        trunc_level = "day"
        interval = timedelta(days=1)
        python_date_trunc = _start_of_day
    elif total_days <= 730:
        trunc_level = "week"
        interval = timedelta(weeks=1)
        python_date_trunc = _start_of_week
    else:
        trunc_level = "month"
        interval = relativedelta(months=1)
        python_date_trunc = _start_of_month
        
    logger.debug(f"Trend granularity set to '{trunc_level}' for {total_days} days.")

    # 3. Define case statements for aggregation
    # ... (This logic is correct and remains unchanged) ...
    positive_case = case((func.lower(base_query_sub.c.label) == 'positive', 1), else_=0)
    negative_case = case((func.lower(base_query_sub.c.label) == 'negative', 1), else_=0)
    neutral_case = case((func.lower(base_query_sub.c.label) == 'neutral', 1), else_=0)

    # 4. RUN SQL QUERY (REMEDIATION 4)
    
    # --- THIS IS THE FIX ---
    trunc_expression: Select
    
    if trunc_level == "week":
        # SQL to truncate to Sunday.
        # 'dow' in PostgreSQL is 0=Sunday.
        
        # 1. Get DOW (e.g., 2 for Tuesday)
        # 2. Cast to string (e.g., '2')
        dow_string = func.cast(
            func.extract('dow', base_query_sub.c.timestamp), 
            String
        )
        
        # 3. Concatenate with ' days' (e.g., '2 days')
        #    This is the correct SQLAlchemy string concatenation
        interval_string = dow_string + ' days'
        
        # 4. Cast the string to a PostgreSQL INTERVAL
        dow_interval = func.cast(interval_string, INTERVAL)
        
        # 5. Subtract the interval to get the preceding Sunday
        trunc_expression = func.date_trunc(
            'day', 
            base_query_sub.c.timestamp - dow_interval
        )
    else:
        # 'day' and 'month' truncations are aligned
        trunc_expression = func.date_trunc(
            trunc_level, 
            base_query_sub.c.timestamp
        )
    # --- END OF FIX ---
    
    stmt = (
        select(
            trunc_expression.label("period"),
            func.sum(positive_case).label("Positive"),
            func.sum(negative_case).label("Negative"),
            func.sum(neutral_case).label("Neutral")
        )
        .select_from(base_query_sub)
        .group_by(trunc_expression) # Group by the expression
        .order_by(trunc_expression) # Order by the expression
    )
    
    result = await db.execute(stmt)
    
    # 5. Put results into a map for fast, zero-filled lookup
    # ... (This logic is correct and remains unchanged) ...
    data_map = {
        row.period.date(): {
            "Positive": row.Positive or 0,
            "Negative": row.Negative or 0,
            "Neutral": row.Neutral or 0
        }
        for row in result
    }
    
    # 6. Zero-fill the gaps in Python
    # ... (This logic is correct and remains unchanged) ...
    trend_data: List[schemas.TrendDataPoint] = []
    
    current_ts = python_date_trunc(min_ts)
    series_end = python_date_trunc(max_ts)
    
    while current_ts <= series_end:
        period_key = current_ts.date()
        counts = data_map.get(
            period_key, 
            {"Positive": 0, "Negative": 0, "Neutral": 0}
        )
        
        trend_data.append(
            schemas.TrendDataPoint(
                date=period_key.strftime("%Y-%m-%d"),
                **counts
            )
        )
        current_ts += interval
        
    logger.debug(f"Trend data calculation complete. {len(trend_data)} periods.")
    return trend_data


async def _get_sentiment_breakdown(
    base_query_sub: Select,
    db: AsyncSession
) -> List[schemas.BreakdownDataPoint]:
    """
    Fetches sentiment counts grouped by participant name.
    """
    logger.debug("Calculating sentiment breakdown by participant.")

    # 1. Define case statements for aggregation
    positive_case = case((func.lower(base_query_sub.c.label) == 'positive', 1), else_=0)
    negative_case = case((func.lower(base_query_sub.c.label) == 'negative', 1), else_=0)
    neutral_case = case((func.lower(base_query_sub.c.label) == 'neutral', 1), else_=0)

    # 2. Build the query
    # We group by the 'sender' column from the subquery
    stmt = (
        select(
            base_query_sub.c.sender.label("name"),
            func.count().label("total"),
            func.sum(positive_case).label("Positive"),
            func.sum(negative_case).label("Negative"),
            func.sum(neutral_case).label("Neutral")
        )
        .select_from(base_query_sub)
        .group_by(base_query_sub.c.sender)
        .order_by(func.count().desc()) # Order by total messages
    )

    result = await db.execute(stmt)
    
    breakdown_data = [
        schemas.BreakdownDataPoint(
            name=row.name,
            total=row.total,
            Positive=row.Positive or 0,
            Negative=row.Negative or 0,
            Neutral=row.Neutral or 0
        )
        for row in result.all()
    ]
    
    logger.debug(f"Breakdown data calculation complete. {len(breakdown_data)} participants found.")
    return breakdown_data


async def _get_sentiment_by_day(
    base_query_sub: Select,
    db: AsyncSession
) -> schemas.SentimentByDayData:
    """
    Fetches sentiment counts grouped by day of the week (0=Sun, 1=Mon, ...).
    Zero-fills any missing days.
    """
    logger.debug("Calculating sentiment by day of week.")

    # 1. Define case statements for aggregation
    positive_case = case((func.lower(base_query_sub.c.label) == 'positive', 1), else_=0)
    negative_case = case((func.lower(base_query_sub.c.label) == 'negative', 1), else_=0)
    neutral_case = case((func.lower(base_query_sub.c.label) == 'neutral', 1), else_=0)

    # 2. Build the query
    # We group by the 'dow' (Day of Week) extract from 'timestamp'
    # PostgreSQL 'dow' is 0=Sunday, 1=Monday, ..., 6=Saturday,
    # which perfectly matches our DAY_OF_WEEK_MAP
    stmt = (
        select(
            func.cast(extract('dow', base_query_sub.c.timestamp), Integer).label("day_of_week"),
            func.count().label("total"),
            func.sum(positive_case).label("positive"),
            func.sum(negative_case).label("negative"),
            func.sum(neutral_case).label("neutral")
        )
        .select_from(base_query_sub)
        .group_by("day_of_week")
    )

    result = await db.execute(stmt)
    
    # 3. Put results into a map for fast lookup
    day_map: Dict[int, Dict[str, int]] = {
        row.day_of_week: {
            "positive": row.positive or 0,
            "negative": row.negative or 0,
            "neutral": row.neutral or 0,
            "total": row.total
        }
        for row in result.all()
    }

    # 4. Zero-fill and calculate scores
    final_day_data: Dict[str, schemas.DailySentimentBreakdown] = {}
    for i, day_key in enumerate(DAY_OF_WEEK_MAP): # DAY_OF_WEEK_MAP = ["sun", "mon", ...]
        counts = day_map.get(i, {"positive": 0, "negative": 0, "neutral": 0, "total": 0})
        
        total = counts["total"]
        pos_percent = (counts["positive"] / total) * 100 if total > 0 else 0
        neg_percent = (counts["negative"] / total) * 100 if total > 0 else 0
        score = round(pos_percent - neg_percent, 1)
        
        final_day_data[day_key] = schemas.DailySentimentBreakdown(
            **counts,
            score=score
        )
    
    logger.debug("Sentiment by day calculation complete.")
    # Use Pydantic to cast the dict to the final schema
    return schemas.SentimentByDayData(**final_day_data)


async def _get_sentiment_by_hour(
    base_query_sub: Select,
    db: AsyncSession
) -> List[schemas.HourlySentimentBreakdown]:
    """
    Fetches sentiment counts grouped by hour of the day (0-23).
    Zero-fills any missing hours.
    """
    logger.debug("Calculating sentiment by hour of day.")

    # 1. Define case statements
    positive_case = case((func.lower(base_query_sub.c.label) == 'positive', 1), else_=0)
    negative_case = case((func.lower(base_query_sub.c.label) == 'negative', 1), else_=0)
    neutral_case = case((func.lower(base_query_sub.c.label) == 'neutral', 1), else_=0)

    # 2. Build the query
    # We group by the 'hour' extract from 'timestamp'
    stmt = (
        select(
            func.cast(extract('hour', base_query_sub.c.timestamp), Integer).label("hour_of_day"),
            func.count().label("total"),
            func.sum(positive_case).label("Positive"),
            func.sum(negative_case).label("Negative"),
            func.sum(neutral_case).label("Neutral")
        )
        .select_from(base_query_sub)
        .group_by("hour_of_day")
    )

    result = await db.execute(stmt)

    # 3. Put results into a map for fast lookup
    hour_map: Dict[int, Dict[str, int]] = {
        row.hour_of_day: {
            "Positive": row.Positive or 0,
            "Negative": row.Negative or 0,
            "Neutral": row.Neutral or 0,
            "total": row.total
        }
        for row in result.all()
    }

    # 4. Zero-fill the list
    final_hour_data: List[schemas.HourlySentimentBreakdown] = []
    for i in range(24): # 0 to 23
        counts = hour_map.get(i, {"Positive": 0, "Negative": 0, "Neutral": 0, "total": 0})
        final_hour_data.append(
            schemas.HourlySentimentBreakdown(
                hour=i,
                **counts
            )
        )
        
    logger.debug("Sentiment by hour calculation complete.")
    return final_hour_data


async def _get_top_highlights(
    base_query_sub: Select,
    db: AsyncSession,
    sentiment_label: str, # "positive" or "negative"
) -> List[schemas.HighlightMessage]:
    """
    A small, dedicated helper to fetch the top 5 items for a given sentiment.
    """
    
    # We query the common columns from the subquery
    stmt = (
        select(
            base_query_sub.c.item_id.label("id"),
            base_query_sub.c.sender.label("sender"),
            base_query_sub.c.text.label("text"),
            base_query_sub.c.timestamp.label("timestamp"),
            base_query_sub.c.score.label("score")
        )
        .select_from(base_query_sub)
        .where(func.lower(base_query_sub.c.label) == sentiment_label)
    )
    
 
    stmt = stmt.order_by(base_query_sub.c.score.desc())
        
    stmt = stmt.limit(5)
    
    result = await db.execute(stmt)
    
    # Use Pydantic's .from_attributes to map rows to schemas
    return [schemas.HighlightMessage.model_validate(row) for row in result.mappings()]
# 


async def _get_highlights_data(
    base_query_sub: Select,
    db: AsyncSession,
    kpi_data: schemas.SentimentKpiData | None # Pass in KPI data
) -> schemas.HighlightsData | None:
    """
    Fetches the top 5 positive and negative highlights concurrently.
    """
    
    # Optimization: If we know there are no messages, don't run queries.
    if not kpi_data or kpi_data.totalMessagesOrSegments == 0:
        logger.debug("Skipping highlights, no data.")
        return None
        
    logger.debug("Calculating highlights data.")

    # Run top positive and top negative queries in parallel
    task_positive = _get_top_highlights(base_query_sub, db, "positive")
    task_negative = _get_top_highlights(base_query_sub, db, "negative")
    
    (
        top_positive_list,
        top_negative_list
    ) = await asyncio.gather(
        task_positive,
        task_negative
    )
    
    logger.debug("Highlights data calculation complete.")
    
    return schemas.HighlightsData(
        topPositive=top_positive_list,
        topNegative=top_negative_list
    )


async def get_sentiment_dashboard_data(
        chat_id: int,
        db: AsyncSession,
        filters: schemas.SentimentDashboardFilters
) -> schemas.SentimentDashboardData | None:
    """
    Retrieves comprehensive sentiment dashboard data for a specific chat,
    """

    logger.info(
        f"Retrieving sentiment dashboard data for chat_id={chat_id} "
        f"with filters={filters.model_dump_json(indent=2)}"
    )


    chat = await crud.get_chat(db, chat_id=chat_id)
    if not chat:
        logger.warning(f"Chat with id {chat_id} not found")
        return None    
    
    base_query = await _create_sentiment_base_query(chat_id, db, filters)
    base_query_sub = base_query.subquery()

    kpi_data = await _get_sentiment_kpis(base_query_sub, db)

    task_all_participants = _get_all_participants(chat_id, db)
    task_sentiment_trend = _get_sentiment_trend(base_query_sub, db)
    task_sentiment_breakdown = _get_sentiment_breakdown(base_query_sub=base_query_sub, db=db)
    task_sentiment_by_hour = _get_sentiment_by_hour(base_query_sub, db)
    task_sentiment_by_day = _get_sentiment_by_day(base_query_sub, db)
    task_highlights = _get_highlights_data(base_query_sub, db, kpi_data)

    (
        all_participants,
        
        sentiment_trend_data,
        sentiment_breakdown_data,
        sentiment_by_day_dt,
        sentiment_by_hour_dt,
        highlights_data,
    ) = await asyncio.gather(
        task_all_participants,
        task_sentiment_trend,
        task_sentiment_breakdown,
        task_sentiment_by_day,
        task_sentiment_by_hour,
        task_highlights,
    )
    
    return schemas.SentimentDashboardData(
        participants=all_participants, 
        kpiData=kpi_data, 
        trendData=sentiment_trend_data,
        breakdownData=sentiment_breakdown_data,
        dayData=sentiment_by_day_dt,
        hourData=sentiment_by_hour_dt,
        highlightsData=highlights_data
    )