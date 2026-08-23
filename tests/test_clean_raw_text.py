import json
from datetime import datetime
from src.app.utils.raw_txt_parser import CleanedMessage, WhatsAppChatParser
from src.app.utils.pre_process import clean_messages
from tests.test_vader import run_pidgin_vader_pass
from src.app.utils.segment_chat import segment_by_time, group_by_sender
from typing import List


def serialize_cleaned_message(msg: CleanedMessage) -> dict:
    """Convert a CleanedMessage object into a JSON-serializable dict."""
    return {
        "timestamp": msg.timestamp.isoformat() if isinstance(msg.timestamp, datetime) else msg.timestamp,
        "sender": msg.sender,
        "text": msg.text,
        "raw": msg.raw
    }


def serialize_segment(segment: dict) -> dict:
    """Convert each segment (with messages and sender_groups) into a serializable format."""
    return {
        "id": segment.get("id"),
        "start_time": segment["start_time"].isoformat() if isinstance(segment["start_time"], datetime) else segment["start_time"],
        "end_time": segment["end_time"].isoformat() if isinstance(segment["end_time"], datetime) else segment["end_time"],
        "duration_minutes": segment.get("duration_minutes"),
        "message_count": segment.get("message_count"),
        "messages": [serialize_cleaned_message(m) for m in segment.get("messages", [])],
        "sender_groups": segment.get("sender_groups", {})
    }


if __name__ == "__main__":
    parser_android = WhatsAppChatParser(dayfirst=True)
    file_path = "C:\\Users\\user\\Downloads\\WhatsApp Chat with SUPER EAGLES🤡🤡\\WhatsApp Chat with SUPER EAGLES🤡🤡.txt"
    file_name = "Chat with SUPER EAGLES"
    messages = parser_android.parse_file(file_path)
    print(f"Parsed {len(messages)} messages from {file_path}")

    cleaned: List[CleanedMessage] = clean_messages(messages)
    print(f"Cleaned chat has {len(cleaned)} messages")
    with open(f"sample_data/cleaned_messages {file_name}.json", "w", encoding="utf-8") as f:
        json.dump([serialize_cleaned_message(m) for m in cleaned], f, ensure_ascii=False, indent=2)

    time_segments = segment_by_time(cleaned)
    time_segments = group_by_sender(time_segments)
    print(f"Segmented chat by time has {len(time_segments)} segments")

    # --- Serialize and Save to JSON
    serializable_segments = [serialize_segment(seg) for seg in time_segments]

    output_file = f"sample_data/time_segments {file_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(serializable_segments, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(serializable_segments)} segments to {output_file}")
