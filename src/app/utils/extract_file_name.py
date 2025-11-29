import re
import os
import uuid

def extract_chat_title(filename: str) -> str:
    """
    Extracts a clean chat title from a WhatsApp chat export filename,
    handling multiple common formats and providing a unique fallback.
    """
    # Get just the filename (strip directories, fake paths, etc.)
    name = os.path.basename(filename or "")

    # Remove .txt extension
    if name.lower().endswith(".txt"):
        name = name[:-4]

    # Remove common WhatsApp export prefixes like "WhatsApp Chat with " or "WhatsApp Chat - "
    # This pattern looks for "whatsapp chat " followed by either "with " or "- ".
    pattern = re.compile(r"whatsapp\s+chat\s+(?:with|-)\s+", re.IGNORECASE)
    name = pattern.sub("", name)

    # Clean underscores, dashes, and multiple spaces that might remain
    name = re.sub(r"[_.-]+", " ", name)
    name = re.sub(r"\s{2,}", " ", name).strip()

    # Provide a unique fallback if the name is empty after cleaning
    if not name:
        unique_suffix = uuid.uuid4().hex[:8].upper()
        name = f"Unnamed Chat {unique_suffix}"

    return name
