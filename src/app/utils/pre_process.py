import re
from typing import Dict
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# ----------------------------------------
# Data structures
# ----------------------------------------
@dataclass
class Message:
    """Structured representation of a single chat message."""
    timestamp: Optional[datetime]
    sender: Optional[str]
    text: str
    raw: str = ""


# ----------------------------------------
# Dictionaries and regex patterns
# ----------------------------------------
SLANG_DICT: Dict[str, str] = {
    "u": "you",
    "btw": "by the way",
    "omg": "oh my god",
    "abi": "right?",        # Nigerian slang
    "wahala": "trouble",    # Nigerian slang
    "sef": "emphasis",      # Nigerian slang
    "howfar": "how are you",# Nigerian slang
    # keep adding...
}

PHONE_RE = re.compile(r'\+?\d[\d\-\s]{7,}\d', re.IGNORECASE)
EMAIL_RE = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', re.IGNORECASE)

# word-boundary slang regex for efficiency
SLANG_RE = re.compile(r'\b(' + '|'.join(map(re.escape, SLANG_DICT.keys())) + r')\b', re.IGNORECASE)


# ----------------------------------------
# Cleaning utilities
# ----------------------------------------
def anonymize_text(s: str) -> str:
    """Replace sensitive info like emails and phone numbers."""
    s = EMAIL_RE.sub('[EMAIL]', s)
    s = PHONE_RE.sub('[PHONE]', s)
    return s


def expand_slang(s: str, slang_map: Dict[str, str] = SLANG_DICT) -> str:
    """Replace slang with formal equivalents using regex word boundaries."""
    return SLANG_RE.sub(lambda m: slang_map.get(m.group(0).lower(), m.group(0)), s)


def normalize_message(s: str) -> str:
    """Apply anonymization, slang expansion, and strip whitespace."""
    s = anonymize_text(s)
    s = expand_slang(s)
    s = s.strip()
    return s


def clean_message(msg: Message) -> Message:
    """Return a new Message object with cleaned text."""
    cleaned = normalize_message(msg.text)
    return Message(
        timestamp=msg.timestamp,
        sender=msg.sender,
        text=cleaned,
        raw=msg.raw
    )


def clean_messages(messages: list[Message]) -> list[Message]:
    """Batch clean messages efficiently."""
    return [clean_message(m) for m in messages]





