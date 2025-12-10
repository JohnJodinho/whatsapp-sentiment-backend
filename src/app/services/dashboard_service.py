import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, extract, Select, case, String
from sqlalchemy.dialects.postgresql import INTERVAL
from typing import Dict, List, Coroutine, Any, Tuple
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta

from src.app.schemas import (
    DashboardData, 
    DashboardFilters, 
    KpiMetric, 
    DayData, 
    HourData,
    MessagesOverTimeData,
    SparklineData,
    ContributionChartData,
    ContributionMulti,
    ContributionTwo,
    ContributionTwoData,
    ContributionSingle,
    ContributionSingleData,
    ContributionParticipant,
    
    ActivityChartData,
    ActivityParticipant,
    ChatSegment,
    ChatSegmentBase,
    MultiParticipantSegment,
    TwoParticipantSegment,
    TwoParticipantBalance,
    ConversationBalance
)

from collections import defaultdict
from src.app.models import Chat, Message, Participant



TIME_PERIOD_BUCKETS: Dict[str, List[int]] = {
    "Morning": [6, 7, 8, 9, 10, 11],
    "Afternoon": [12, 13, 14, 15, 16],
    "Evening": [17, 18, 19, 20],
    "Night": [21, 22, 23, 0, 1, 2, 3, 4, 5],
}


DAY_OF_WEEK_MAP: List[str] = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"]


def _get_day_suffix(day: int) -> str:
    """Formats a day number (1, 2, 31) into a string (1st, 2nd, 31st)."""
    if 11 <= day <= 13:
        return f"{day}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def create_base_query(filters: DashboardFilters, chat_id: int) -> Select:
    """
    Creates a reusable, un-executed SQLAlchemy Select statement.
    """
    
    query = (
        select(Message)
        .join(Participant, Message.participant_id == Participant.id)
        .where(Message.chat_id == chat_id)
    )

    if filters.start_date:
    
        start_date = filters.start_date
        if start_date.tzinfo:
            start_date = start_date.replace(tzinfo=None)
        query = query.where(Message.timestamp >= start_date)
            
    if filters.end_date:
        end_date = filters.end_date
        if end_date.tzinfo:
            end_date = end_date.replace(tzinfo=None)
        query = query.where(Message.timestamp <= end_date)
    if filters.participants:
        query = query.where(Participant.name.in_(filters.participants))
    if filters.time_period:
        hour_bucket = TIME_PERIOD_BUCKETS.get(filters.time_period)
        if hour_bucket:
            query = query.where(extract('hour', Message.timestamp).in_(hour_bucket))
            
    return query


def _start_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)

def _start_of_week(dt: datetime) -> datetime:
    """Returns the start of the week (Sunday), matching date-fns enUS locale."""
    start_of_day = _start_of_day(dt)

    days_to_subtract = (start_of_day.weekday() + 1) % 7
    return start_of_day - timedelta(days=days_to_subtract)

def _start_of_month(dt: datetime) -> datetime:
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


async def _get_all_participants(chat_id: int, db: AsyncSession) -> List[str]:
    """
    Fetches the full list of all participant names for the filter dropdown.
    This query IGNORES filters.
    """
    stmt = (
        select(Participant.name)
        .where(Participant.chat_id == chat_id)
        .distinct()
        .order_by(Participant.name)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def _get_kpi_aggregates(
    base_query: Select, db: AsyncSession
) -> Tuple[int, int, int]:

    sub = base_query.subquery()
    
    stmt = (
        select(
            func.count(sub.c.id).label("total_messages"),
            func.count(func.distinct(sub.c.participant_id)).label("participant_count"),
            func.count(
                func.distinct(func.date_trunc('day', sub.c.timestamp))
            ).label("active_days")
        )
        .select_from(sub)  
    )

    row = (await db.execute(stmt)).one()
    
    return (
        row.total_messages,
        row.participant_count,
        row.active_days
    )


async def _get_activity_by_day(base_query: Select, db: AsyncSession) -> List[DayData]:
    """Fetches message counts grouped by day of the week.
    """

    sub = base_query.subquery()

    stmt = (
        select(
            extract('dow', sub.c.timestamp).label("day_of_week"),
            func.count(sub.c.id).label("message_count")
        )
        .select_from(sub)  
        .group_by("day_of_week")
    )
    
    
    result = await db.execute(stmt)
    
    counts_map = {int(row.day_of_week): row.message_count for row in result.all()}
    
    return [
        DayData(
            day=DAY_OF_WEEK_MAP[i],
            messages=counts_map.get(i, 0)
        ) for i in range(7)
    ]


async def _get_hourly_activity(base_query: Select, db: AsyncSession) -> List[HourData]:
    """
    Fetches message counts grouped by hour of the day.
    (0=12am, 1=1am, ..., 23=11pm)
    """
    sub = base_query.subquery()
    stmt = (
        select(
            extract('hour', sub.c.timestamp).label("hour_of_day"),
            func.count(sub.c.id).label("message_count")
        )
        .select_from(sub) 
        .group_by("hour_of_day")
    )
    
    result = await db.execute(stmt)
    
    counts_map = {int(row.hour_of_day): row.message_count for row in result.all()}
    
    return [
        HourData(
            hour=i,
            messages=counts_map.get(i, 0)
        ) for i in range(24)
    ]

async def _get_time_series_data(
    base_query: Select, db: AsyncSession
) -> Tuple[List[MessagesOverTimeData], List[SparklineData], List[SparklineData]]:
    """
    Fetches all time-series data:
    1. Messages over time (main chart)
    2. Message count sparkline
    3. Participant count sparkline

    Fully aligned with sentiment trend time-series logic.
    """

    # ------------------------------
    # 1. Get real min/max timestamps
    # ------------------------------
    sub = base_query.subquery()
    range_stmt = select(
        func.min(sub.c.timestamp).label("min_ts"),
        func.max(sub.c.timestamp).label("max_ts")
    )
    range_result = (await db.execute(range_stmt)).one_or_none()

    if not range_result or not range_result.min_ts:
        return ([], [], [])

    min_ts: datetime = range_result.min_ts
    max_ts: datetime = range_result.max_ts

    # ------------------------------
    # 2. Detect range and granularity
    # ------------------------------
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

    # ------------------------------
    # 3. SQL truncation expression
    # ------------------------------
    if trunc_level == "week":
        # Match EXACT logic of sentiment trend
        dow_string = func.cast(func.extract('dow', sub.c.timestamp), String)
        interval_string = dow_string + ' days'
        dow_interval = func.cast(interval_string, INTERVAL)

        trunc_expression = func.date_trunc(
            'day',
            sub.c.timestamp - dow_interval
        )
    else:
        trunc_expression = func.date_trunc(
            trunc_level,
            sub.c.timestamp
        )

    # ------------------------------
    # 4. Build SQL query
    # ------------------------------
    stmt = (
        select(
            trunc_expression.label("period"),
            func.count(sub.c.id).label("message_count"),
            func.count(func.distinct(sub.c.participant_id)).label("participant_count")
        )
        .select_from(sub)
        .group_by(trunc_expression)
        .order_by(trunc_expression)
    )

    result = await db.execute(stmt)

    # ------------------------------
    # 5. Convert SQL output to map
    # ------------------------------
    data_map = {
        row.period.date(): (row.message_count, row.participant_count)
        for row in result
    }

    # ------------------------------
    # 6. Zero-fill missing periods
    # ------------------------------
    messages_over_time: List[MessagesOverTimeData] = []
    messages_sparkline: List[SparklineData] = []
    participants_sparkline: List[SparklineData] = []

    current_ts = python_date_trunc(min_ts)
    series_end = python_date_trunc(max_ts)

    while current_ts <= series_end:
        msg_count, part_count = data_map.get(current_ts.date(), (0, 0))

        # A. Main chart
        messages_over_time.append(
            MessagesOverTimeData(
                date=current_ts.strftime("%Y-%m-%d"),
                count=msg_count
            )
        )

        # B. Sparkline (messages)
        messages_sparkline.append(SparklineData(v=msg_count))

        # C. Sparkline (participants)
        participants_sparkline.append(SparklineData(v=part_count))

        current_ts += interval

    return (messages_over_time, messages_sparkline, participants_sparkline)


# In src/app/services/dashboard_service.py

# ... (other imports) ...

# MODIFICATION 1: Add 'chat_id: int' as the first argument
async def _get_contribution_data(
    chat_id: int,
    base_query: Select,
    db: AsyncSession,
    filters: DashboardFilters,
    participant_count: int,
    total_messages: int # This is the count AFTER all filters
) -> ContributionChartData:
    """
    Fetches the data for the Contribution chart, dynamically
    changing its shape based on the number of active participants.
    """
    
    # --- State 1: 0 Participants ---
    if participant_count == 0:
        return ContributionMulti(data=[])

    # --- State 2: 1 Participant ("Share of Voice") ---
    if participant_count == 1:
        # We need the total messages in this time range from *everyone*
        # to calculate the "share of voice".
        
        # Create a new filter set, but *remove* the participant filter
        total_filters = filters.model_copy(
            update={"participants": None}
        )
        
        # MODIFICATION 2: Use the 'chat_id' variable we passed in
        total_base_query = create_base_query(
            filters=total_filters, 
            chat_id=chat_id # Get chat_id from query
        )
        
        total_stmt = select(func.count()).select_from(total_base_query.subquery())
        total_in_period = (await db.execute(total_stmt)).scalar_one()
        
        participant_messages = total_messages # from the original filtered query
        others_messages = total_in_period - participant_messages
        
        # This logic is now safer
        participant_name = filters.participants[0] if filters.participants else "Unknown"
        if not filters.participants and participant_count == 1:
             # Fallback if participant filter was empty but count is 1 (e.g. no filter)
             contrib_data = await _get_contribution_data_for_names(base_query, db)
             if contrib_data:
                 participant_name = contrib_data[0].name

        return ContributionSingle(
            data=ContributionSingleData(
                name=participant_name,
                percentage=(participant_messages / total_in_period) * 100 if total_in_period > 0 else 0,
                othersPercentage=(others_messages / total_in_period) * 100 if total_in_period > 0 else 0
            )
        )
    
    # --- State 3: 2 or More Participants ---
    participant_data = await _get_contribution_data_for_names(base_query, db)

    # Case 3a: Exactly 2 Participants
    if participant_count == 2:
        return ContributionTwo(
            data=ContributionTwoData(
                participants=participant_data,
                totalMessages=total_messages
            )
        )
    
    # Case 3b: More than 2 Participants (Multi)
    # The data is already in the correct format
    return ContributionMulti(data=participant_data[:30]) # Max 30 participants


async def _get_contribution_data_for_names(
    base_query: Select, db: AsyncSession
) -> List[ContributionParticipant]:
    """
    Helper function to get the participant names and message counts
    for contribution charts (2 or more participants).
    """
    sub = base_query.subquery()
    # We must explicitly join Participant back in, as the subquery flattens it
    stmt = (
        select(Participant.name, func.count().label("message_count"))
        .join(sub, sub.c.participant_id == Participant.id)
        .group_by(Participant.name)
        .order_by(func.count().desc())
    )
    
    result = await db.execute(stmt)
    return [
        ContributionParticipant(name=row.name, messages=row.message_count)
        for row in result.all()
    ]

async def _get_activity_data(
    base_query: Select,
    db: AsyncSession,
    participant_count: int,
    contribution_data: ContributionChartData
) -> ActivityChartData | None:
    """
    Fetches the data for the Activity (Radar) Chart.
    
    It shows 1-3 participants. If more than 3 are active,
    it shows the Top 3 based on message count.
    """
    
    if participant_count < 1:
        return None

    participants_to_query: List[str] = []

    # --- THIS IS THE FIX ---
    # We must handle ContributionMulti and ContributionTwo differently
    # because their 'data' attribute has a different shape.
    if participant_count <= 3:
        if isinstance(contribution_data, ContributionMulti):
            # data is a List[ContributionParticipant]
            participants_to_query = [p.name for p in contribution_data.data]
        elif isinstance(contribution_data, ContributionTwo):
            # data is an object { participants: [...] }
            participants_to_query = [p.name for p in contribution_data.data.participants]
        elif isinstance(contribution_data, ContributionSingle):
            participants_to_query = [contribution_data.data.name]
            
    else:
        # More than 3 participants, get Top 3 from contribution data
        # This only happens on ContributionMulti, so this logic is correct.
        if isinstance(contribution_data, ContributionMulti):
            participants_to_query = [
                p.name for p in contribution_data.data[:3]
            ]
    # --- END OF FIX ---
            
    if not participants_to_query:
        return None
        
    labels = ["Text", "Media", "Links", "Questions", "Emojis"]
    
    sub = base_query.subquery()
    
    text_message_case = case(
        (sub.c.is_media == False, 1),
        else_=0
    )
    
    stmt = (
        select(
            Participant.name,
            func.sum(text_message_case).label("text_count"),
            func.sum(case((sub.c.is_media == True, 1), else_=0)).label("media_count"),
            func.sum(sub.c.links_count).label("links_count"),
            func.sum(case((sub.c.is_question == True, 1), else_=0)).label("questions_count"),
            func.sum(sub.c.emojis_count).label("emojis_count")
        )
        .join(sub, sub.c.participant_id == Participant.id)
        .where(Participant.name.in_(participants_to_query))
        .group_by(Participant.name)
    )

    result = await db.execute(stmt)
    
    participant_chart_data: List[ActivityParticipant] = []
    for row in result.all():
        participant_chart_data.append(
            ActivityParticipant(
                name=row.name,
                data=[
                    row.text_count,
                    row.media_count,
                    row.links_count,
                    row.questions_count,
                    row.emojis_count
                ]
            )
        )

    return ActivityChartData(
        labels=labels,
        participants=participant_chart_data
    )



async def _get_timeline_data(
    base_query: Select,
    db: AsyncSession,
    participant_count: int,
    contribution_data: ContributionChartData 
) -> List[ChatSegment]:
    """
    Fetches the data for the Timeline table.
    ...
    """
    sub = base_query.subquery()
    
    # Query 1 is correct (it uses a JOIN)
    participant_stmt = (
        select(
            func.date_trunc('month', sub.c.timestamp).label("month"),
            Participant.name,
            func.count().label("message_count")
        )
        .join(sub, sub.c.participant_id == Participant.id)
        .group_by("month", Participant.name)
        .order_by("month")
    )
    
    # --- THIS IS THE FIX (Query 2) ---
    daily_stmt = (
        select(
            func.date_trunc('month', sub.c.timestamp).label("month"),
            extract('day', sub.c.timestamp).label("day"),
            func.count(sub.c.id).label("message_count")
        )
        .select_from(sub)  # <-- ADD THIS LINE
        .group_by("month", "day")
    )
    # --- END OF FIX ---
    
    # --- Run both queries in parallel ---
    participant_task = db.execute(participant_stmt)
    daily_task = db.execute(daily_stmt)
    

    participant_results, daily_results = await asyncio.gather(
        participant_task, daily_task
    )
    
    # ... (Map processing is unchanged) ...
    month_participant_map = defaultdict(list)
    for row in participant_results.all():
        month_key = row.month.strftime("%B %Y")
        month_participant_map[month_key].append((row.name, row.message_count))
        
    peak_day_raw_map = defaultdict(lambda: (0, 0)) # Stores (day, count)
    for row in daily_results.all():
        month_key = row.month.strftime("%B %Y")
        day = int(row.day)
        count = row.message_count
        if count > peak_day_raw_map[month_key][1]:
            peak_day_raw_map[month_key] = (day, count)
            
    peak_day_map = {
        month: _get_day_suffix(day)
        for month, (day, count) in peak_day_raw_map.items()
        if day > 0
    }

    # --- Stitch data together ---
    
    final_timeline: List[ChatSegment] = []
    
    # --- MODIFICATION: Get definitive names for 2-participant view ---
    participant_names = []
    if participant_count == 2 and isinstance(contribution_data, ContributionTwo):
         participant_names = [p.name for p in contribution_data.data.participants]

    # Use the map from participant_results
    for month_key, participant_counts_list in month_participant_map.items():
        
        total_messages = sum(count for _, count in participant_counts_list)
        
        base_data = {
            "month": month_key,
            "totalMessages": total_messages,
            "peakDay": peak_day_map.get(month_key, "N/A")
        }

        # Case 1: Single Participant
        if participant_count == 1:
            final_timeline.append(ChatSegmentBase(**base_data))
            continue
        
        # --- MODIFICATION: This block is now robust ---
        # Case 2: Two Participants
        if participant_count == 2 and participant_names:
            # Create a simple lookup map for *this month's* counts
            counts_map = dict(participant_counts_list)
            
            # Get the names from the definitive list
            p1_name = participant_names[0]
            p2_name = participant_names[1]
            
            # Get count, defaulting to 0 if name not in this month's map
            p1_count = counts_map.get(p1_name, 0)
            p2_count = counts_map.get(p2_name, 0)
            
            p1_perc = (p1_count / total_messages) * 100 if total_messages > 0 else 0
            p2_perc = (p2_count / total_messages) * 100 if total_messages > 0 else 0
            
            final_timeline.append(
                TwoParticipantSegment(
                    **base_data,
                    conversationBalance=ConversationBalance(
                        participantA=TwoParticipantBalance(name=p1_name, percentage=p1_perc),
                        participantB=TwoParticipantBalance(name=p2_name, percentage=p2_perc)
                    )
                )
            )
            continue
        # --- END MODIFICATION ---
        
        # Case 3: Multi-Participant
        # (This block now also acts as a fallback if count=2 but data is weird)
        if participant_count > 2:
            most_active = max(participant_counts_list, key=lambda item: item[1])
            
            final_timeline.append(
                MultiParticipantSegment(
                    **base_data,
                    activeParticipants=len(participant_counts_list),
                    mostActive=most_active[0]
                )
            )

    # ... (Sorting is unchanged) ...
    try:
        final_timeline.sort(
            key=lambda x: datetime.strptime(x.month, "%B %Y"),
            reverse=True
        )
    except ValueError:
        pass # If data is empty

    return final_timeline

async def get_dashboard_data(
    chat_id: int, 
    db: AsyncSession, 
    filters: DashboardFilters
) -> DashboardData | None:
    
    # First, check if the chat actually exists
    chat = await db.get(Chat, chat_id)
    if not chat:
        return None # This will trigger the 404 in the router
    
    # --- Phase 2: Create the base query ---
    base_query = create_base_query(filters=filters, chat_id=chat_id)
    
# --- Phase 4: Run queries concurrently ---
    
    task_all_participants = _get_all_participants(chat_id, db)
    task_kpi_aggregates = _get_kpi_aggregates(base_query, db)
    task_activity_by_day = _get_activity_by_day(base_query, db)
    task_hourly_activity = _get_hourly_activity(base_query, db)
    task_time_series = _get_time_series_data(base_query, db)
    
    (
        all_participants,
        kpi_aggregates,
        activity_by_day_data,
        hourly_activity_data,
        time_series_data
    ) = await asyncio.gather(
        task_all_participants,
        task_kpi_aggregates,
        task_activity_by_day,
        task_hourly_activity,
        task_time_series
    )
    
    # Unpack Batch 1 results
    total_messages, participant_count, active_days = kpi_aggregates
    (
        messages_over_time_data, 
        messages_sparkline_data, 
        participants_sparkline_data
    ) = time_series_data
    
    # --- Batch 2 (Depends on Batch 1) ---
    # This task is now the only one in Batch 2
    # because Batch 3 depends on it.
    contribution_data = await _get_contribution_data(

        chat_id, base_query, db, filters, participant_count, total_messages
    )
    
    # --- Batch 3 (Depends on Batch 2) ---
    # These tasks depend on 'contribution_data'
    # and can run in parallel with each other.
    task_timeline = _get_timeline_data(
        base_query, db, participant_count, contribution_data
    )
    task_activity = _get_activity_data(
        base_query, db, participant_count, contribution_data
    )
    
    # Await all Batch 3 tasks
    (
        timeline_data,
        activity_data
    ) = await asyncio.gather(
        task_timeline,
        task_activity
    )
    # --- Build KPI Metrics ---
    avg_messages_per_day = (
        (total_messages / active_days) if active_days > 0 else 0
    )
    
    kpi_metrics = [
        KpiMetric(
            label="Total Messages",
            value=total_messages,
            definition="All messages sent in the filtered period.",
            sparkline=messages_sparkline_data # <-- MODIFIED
        ),
        KpiMetric(
            label="Active Participants",
            value=participant_count,
            definition="Unique participants who sent messages.",
            sparkline=participants_sparkline_data # <-- MODIFIED
        ),
        KpiMetric(
            label="Active Days",
            value=active_days,
            definition="The total number of unique days with at least one message."
        ),
        KpiMetric(
            label="Avg. Messages/Day",
            value=round(avg_messages_per_day, 1),
            definition="The average number of messages sent per active day."
        )
    ]
    
    
    # --- Assemble the final response ---
    
    return DashboardData(
        participants=all_participants,
        participantCount=participant_count,
        kpiMetrics=kpi_metrics,
        messagesOverTime=messages_over_time_data,
        contribution=contribution_data,
        activity=activity_data,
        timeline=timeline_data, 
        activityByDay=activity_by_day_data,
        hourlyActivity=hourly_activity_data
    )