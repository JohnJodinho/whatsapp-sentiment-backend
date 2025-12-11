from datetime import datetime
from typing import List, Optional, Dict, Union, Any, Literal
from pydantic import BaseModel, ConfigDict
import enum

QueryRoute = Literal["CHAT", "ANALYTICS", "BOTH"]

class SentimentStatusEnum(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled" # <-- ADD THIS

class EmbeddingStatusEnum(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"
class ParticipantBase(BaseModel):
    name: str

class ParticipantRead(ParticipantBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    chat_id: int
    created_at: datetime

class ChatBase(BaseModel):
    title: Optional[str] = None


class ChatCreate(ChatBase):
    pass

class ChatRead(ChatBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    sentiment_status: SentimentStatusEnum
    cancel_requested: Optional[bool] = False


class ChatReadExtended(ChatRead):
    participants_number: int
    messages_number: int

class MessageBase(BaseModel):
    timestamp: datetime
    content: str
    raw: Optional[str] = None
    word_count: Optional[int] = 0
    emojis_count: Optional[int] = 0
    links_count: Optional[int] = 0
    is_question: Optional[bool] = False
    is_media: Optional[bool] = False

class MessageCreate(MessageBase):
    participant_id: Optional[int] = None
    chat_id: int

class MessageRead(MessageBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    participant: Optional[ParticipantRead] = None




class SenderSegmentBase(BaseModel):
    time_segment_id: int
    sender_id: int
    message_count: int
    combined_text: str

class SenderSegmentCreate(SenderSegmentBase):
    pass
class SenderSegmentRead(SenderSegmentBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    


class TimeSegmentBase(BaseModel):
    chat_id: int
    start_time: datetime
    end_time: datetime
    duration_minutes: int
    message_count: int
    summary: Optional[str] = None

class TimeSegmentCreate(TimeSegmentBase):
    pass
class TimeSegmentRead(TimeSegmentBase):
    model_config = ConfigDict(from_attributes=True)

    id: int

   


class SentimentBase(BaseModel):
    overall_label: int
    overall_label_score: float
    score_positive: Optional[float]
    score_negative: Optional[float]
    score_neutral: Optional[float]
    
    sentences_summary: Optional[dict] = None
    opinions: Optional[dict] = None
    api_version: Optional[str] = None
    analysis_timestamp: Optional[datetime] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None

class SentimentCreate(SentimentBase):
    message_id: Optional[int] = None
    sender_segment_id: Optional[int] = None

class SentimentRead(SentimentBase):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    message_id: Optional[int] = None
    sender_segment_id: Optional[int] = None


class DashboardFilters(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    participants: Optional[List[str]] = None
    time_period: Optional[str] = None

class SparklineData(BaseModel):
    """Matches the TS contract: { v: number }"""
    v: Union[int, float]

class KpiMetric(BaseModel):
    """
    Corrected:
    - 'sparkline' is now correctly nested inside KpiMetric,
      matching the TS contract.
    - 'value' is Union[int, float] for flexibility.
    """
    label: str
    value: Union[int, float]
    definition: str
    sparkline: Optional[List[SparklineData]] = None

# --- B. MessagesOverTimeChart Props ---

class MessagesOverTimeData(BaseModel):
    """Matches the TS contract"""
    date: str  # "YYYY-MM-DD"
    count: int

# --- C. ParticipantCharts Props ---

# 1. For Contribution Chart (Discriminated Union)
# Refactored from 'Any' to a specific, type-safe Union.

class ContributionParticipant(BaseModel):
    name: str
    messages: int

class ContributionMulti(BaseModel):
    type: Literal["multi"] = "multi"
    data: List[ContributionParticipant]

class ContributionTwoData(BaseModel):
    participants: List[ContributionParticipant]
    totalMessages: int

class ContributionTwo(BaseModel):
    type: Literal["two"] = "two"
    data: ContributionTwoData

class ContributionSingleData(BaseModel):
    name: str
    percentage: float
    othersPercentage: float

class ContributionSingle(BaseModel):
    type: Literal["single"] = "single"
    data: ContributionSingleData

# This Union now perfectly matches the TS discriminated union
ContributionChartData = Union[ContributionMulti, ContributionTwo, ContributionSingle]


# 2. For Activity Chart (Radar)
# Refactored from 'Any' to a specific model.

class ActivityParticipant(BaseModel):
    """Matches the TS contract"""
    name: str
    data: List[Union[int, float]] # Use float/int for flexibility

class ActivityChartData(BaseModel):
    """
    Corrected:
    - This model, 'ActivityChartData', is what the frontend expects.
    - It replaces the old 'activityMetrics: List[ParticipantActivity]'.
    """
    labels: List[str]  # e.g., ["Text", "Media", "Links"]
    participants: List[ActivityParticipant]


# --- D. TimelineTable Props ---

# Refactored from 'Any' to a type-safe Union.
# Pydantic will try to match them in order (most specific to least specific).

class ChatSegmentBase(BaseModel):
    """The base model, matching 'SingleParticipantSegment' in TS"""
    month: str
    totalMessages: int
    peakDay: str

class MultiParticipantSegment(ChatSegmentBase):
    """Matches the 'MultiParticipantSegment' in TS"""
    activeParticipants: int
    mostActive: str

class TwoParticipantBalance(BaseModel):
    name: str
    percentage: float

class ConversationBalance(BaseModel):
    participantA: TwoParticipantBalance
    participantB: TwoParticipantBalance

class TwoParticipantSegment(ChatSegmentBase):
    """Matches the 'TwoParticipantSegment' in TS"""
    conversationBalance: ConversationBalance

# The Union: Pydantic tries Multi, then Two, then falls back to Base
ChatSegment = Union[MultiParticipantSegment, TwoParticipantSegment, ChatSegmentBase]


# --- E. OptionalInsights Props ---

class DayData(BaseModel):
    """Matches the TS contract"""
    day: str  # "mon", "tue", etc.
    messages: int
    # 'fill' (color) is correctly left out; it's a frontend concern

class HourData(BaseModel):
    """Matches the TS contract"""
    hour: int  # 0-23
    messages: int


# --- The Main DashboardData Contract ---

class DashboardData(BaseModel):
    """
    This is the single, corrected response model that perfectly
    matches your TypeScript contract.
    """
    model_config = ConfigDict(from_attributes=True)

    # For FiltersCard (and other logic)
    participants: List[str]      # Full list of all participants
    participantCount: int      # Count *after* filtering

    # For KPICardsRow
    kpiMetrics: List[KpiMetric]  # Correct: sparkline is nested

    # For MessagesOverTimeChart
    messagesOverTime: List[MessagesOverTimeData]

    # For ParticipantCharts
    contribution: ContributionChartData  # Correct: Now a typed Union
    
    # Corrected: Renamed/replaced 'activityMetrics'
    activity: Optional[ActivityChartData]  

    # For TimelineTable
    timeline: List[ChatSegment]  # Correct: Now a typed Union

    # For OptionalInsights
    activityByDay: List[DayData]
    hourlyActivity: List[HourData]





# --- Additions for Sentiment Dashboard ---
class SentimentGranularity(str, enum.Enum):
    message = "message"
    segment = "segment"

class SentimentLabel(str, enum.Enum):
    Positive = "Positive"
    Negative = "Negative"
    Neutral = "Neutral"

# 1. Filter Dependency Model
# This model will be populated by the new router dependency
class SentimentDashboardFilters(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    participants: Optional[List[str]] = None
    time_period: Optional[str] = None
    granularity: SentimentGranularity = SentimentGranularity.message
    sentiment_types: Optional[List[SentimentLabel]] = None

# 2. KPI Row Data
class SentimentKpiData(BaseModel):
    overallScore: float
    positivePercent: float
    negativePercent: float
    neutralPercent: float
    totalMessagesOrSegments: int

# 3. Sentiment Trend Data
class TrendDataPoint(BaseModel):
    date: str  # "YYYY-MM-DD"
    Positive: int
    Negative: int
    Neutral: int

# 4. Sentiment Breakdown Data
class BreakdownDataPoint(BaseModel):
    name: str  # Participant name
    Positive: int
    Negative: int
    Neutral: int
    total: int

# 5. Sentiment by Day Data
class DailySentimentBreakdown(BaseModel):
    positive: int
    negative: int
    neutral: int
    total: int
    score: float # (%Pos - %Neg)

# This matches the TS type { sun: {...}, mon: {...}, ... }
class SentimentByDayData(BaseModel):
    sun: DailySentimentBreakdown
    mon: DailySentimentBreakdown
    tue: DailySentimentBreakdown
    wed: DailySentimentBreakdown
    thu: DailySentimentBreakdown
    fri: DailySentimentBreakdown
    sat: DailySentimentBreakdown

# 6. Sentiment by Hour Data
class HourlySentimentBreakdown(BaseModel):
    hour: int  # 0-23
    Positive: int
    Negative: int
    Neutral: int
    total: int

# 7. Highlights Data
class HighlightMessage(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int # Will be Message.id or SenderSegment.id
    sender: str
    text: str # Snippet
    timestamp: datetime # Use datetime for sorting, will be str on final JSON
    score: float

class HighlightsData(BaseModel):
    topPositive: List[HighlightMessage]
    topNegative: List[HighlightMessage]

# 8. Main Contract
class SentimentDashboardData(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    participants: List[str] # Full list for filter
    kpiData: Optional[SentimentKpiData] = None
    trendData: Optional[List[TrendDataPoint]] = None
    breakdownData: Optional[List[BreakdownDataPoint]] = None
    dayData: Optional[SentimentByDayData] = None
    hourData: Optional[List[HourlySentimentBreakdown]] = None
    highlightsData: Optional[HighlightsData] = None




class RagSource(BaseModel):
    source_table: str
    source_id: int
    sender_name: Optional[str] = None
    timestamp: Optional[datetime] = None
    distance: float
    text: Optional[str] = None


class RagQueryRequest(BaseModel):
    question: str
    analytics_json: Optional[Dict[str, Any]] = None

class RagQueryResponse(BaseModel):
    answer: str
    sources: List[RagSource]
    route: QueryRoute

class ConversationHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    role: str
    content: str
    sources: Optional[List[RagSource]] = None
    created_at: datetime