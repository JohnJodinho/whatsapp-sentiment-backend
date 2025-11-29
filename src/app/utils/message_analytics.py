import re
from typing import Dict

def compute_message_analytics(content: str) -> Dict:
    """Compute analytics for a message during ingestion."""
    # Count words (split by whitespace and filter empty strings)
    word_count = len([w for w in content.split() if w.strip()])
    
    # Count emojis (simple regex for emoji range)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    emojis_count = len(emoji_pattern.findall(content))
    
    # Count links (simple URL pattern)
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    links_count = len(url_pattern.findall(content))
    
    # Detect questions (ends with ? or starts with question words)
    question_words = r'\b(who|what|when|where|why|how|which|whose|whom)\b'
    is_question = bool(re.search(r'\?$', content.strip()) or 
                      re.search(question_words, content.lower()))
    
    # Detect media messages (common WhatsApp media indicators)
    media_indicators = [
        'image omitted', 'video omitted', 'audio omitted', 
        'sticker omitted', 'document omitted', 'GIF omitted', 'media'
    ]
    is_media = any(indicator in content.lower() for indicator in media_indicators)
    
    return {
        'word_count': word_count,
        'emojis_count': emojis_count,
        'links_count': links_count,
        'is_question': is_question,
        'is_media': is_media
    }