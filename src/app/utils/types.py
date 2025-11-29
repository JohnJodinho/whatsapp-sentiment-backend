from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class CleanedMessage:
    timestamp: Optional[datetime]
    sender: Optional[str]
    text: str
    raw: str
