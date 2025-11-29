import re
from typing import Dict
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from src.app.utils.raw_txt_parser import CleanedMessage

# ----------------------------------------
# Data structures
# ----------------------------------------



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
SYSTEM_MESSAGE_STRINGS = [
    "Messages and calls are end-to-end encrypted",
    "changed the group subject",
    "changed this group's icon",
    "changed the group description",
    "created group",
    "created this group",
    "You were added",
    "You joined",
    "joined using this group's invite link",
    "left",
    "added",
    "removed",
    "You're now an admin",
    "You are now an admin",
    "You're no longer an admin",
    "You are no longer an admin",
    "missed a call",
    "missed a video call",
    "deleted this message",
    "You deleted this message",
]
SYSTEM_RE = re.compile("|".join(SYSTEM_MESSAGE_STRINGS), re.IGNORECASE)
MEDIA_RE = re.compile(
    r"^\u200e?[<\[]?(image|video|audio|document|sticker|Media)\somitted[>\]]?$",
    re.IGNORECASE
)
CHARS_TO_STRIP = '\u200e \t\n\r'
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

def preprocess_messages(messages: list[CleanedMessage]) -> list[CleanedMessage]:
    """
    Remove system messages and mask media placeholders
    for both Android and iOS exports.
    """
    filtered: list[CleanedMessage] = []

    for msg in messages:
        # 1. Skip system messages (no sender)
        if msg.sender is None:
            continue

        # 2. Skip common system message text, even if sender was parsed
        if SYSTEM_RE.search(msg.text):
            continue

        # 3. Mask media messages
        # We strip standard whitespace AND the invisible LRM character
        stripped_text = msg.text.strip(CHARS_TO_STRIP)

        # Use the new robust regex
        if MEDIA_RE.fullmatch(stripped_text):
            masked = CleanedMessage(
                timestamp=msg.timestamp,
                sender=msg.sender,
                text="[MEDIA]",  # Standardize to [MEDIA]
                raw=msg.raw,
            )
            filtered.append(masked)
        else:
            # This is a regular text message
            filtered.append(msg)

    return filtered


def clean_message(msg: CleanedMessage) -> CleanedMessage:
    """Return a new Message object with cleaned text."""
    cleaned = normalize_message(msg.text)
    return CleanedMessage(
        timestamp=msg.timestamp,
        sender=msg.sender,
        text=cleaned,
        raw=msg.raw
    )


def clean_messages(messages: list[CleanedMessage]) -> list[CleanedMessage]:
    """Batch clean messages efficiently."""
    messages = preprocess_messages(messages)
    return [clean_message(m) for m in messages]





