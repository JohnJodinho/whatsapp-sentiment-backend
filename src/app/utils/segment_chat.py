from datetime import datetime, timedelta
from typing import List, Dict
import numpy as np
from src.app.utils.raw_txt_parser import CleanedMessage
from collections import defaultdict





def segment_by_time(messages: List["CleanedMessage"], gap_minutes: int = None) -> List[Dict]:
    """Group messages into conversation sessions based on dynamically derived inactivity gap."""
    if not messages:
        return []

    messages.sort(key=lambda m: m.timestamp or datetime.min)

    # --- Derive gap threshold ---
    if gap_minutes is None:
        timestamps = [m.timestamp for m in messages if m.timestamp]
        if len(timestamps) > 1:
            ts = np.array([t.timestamp() for t in timestamps])
            gaps = np.diff(ts) / 60.0
            gaps = gaps[gaps < 24 * 60]
            if len(gaps):
                median_gap = np.median(gaps)
                std_gap = np.std(gaps)
                gap_minutes = int(np.clip(median_gap + std_gap, 15, 180))
            else:
                gap_minutes = 30
        else:
            gap_minutes = 30

    threshold = timedelta(minutes=gap_minutes)
    segments = []
    segment_msgs = []
    last_time = None
    segment_id = 0

    for msg in messages:
        if not msg.timestamp:
            continue

        if last_time and (msg.timestamp - last_time) > threshold:
            # Commit the previous segment
            if segment_msgs:
                segments.append({
                    "id": segment_id,
                    "start_time": segment_msgs[0].timestamp,
                    "end_time": segment_msgs[-1].timestamp,
                    "duration_minutes": int((segment_msgs[-1].timestamp - segment_msgs[0].timestamp).total_seconds() / 60),
                    "message_count": len(segment_msgs),
                    "messages": list(segment_msgs),  # 👈 FIX
                })
                segment_id += 1
                segment_msgs = []  # fresh list

        segment_msgs.append(msg)
        last_time = msg.timestamp

    # Commit last segment
    if segment_msgs:
        segments.append({
            "id": segment_id,
            "start_time": segment_msgs[0].timestamp,
            "end_time": segment_msgs[-1].timestamp,
            "duration_minutes": int((segment_msgs[-1].timestamp - segment_msgs[0].timestamp).total_seconds() / 60),
            "message_count": len(segment_msgs),
            "messages": list(segment_msgs),
        })

    return segments


def group_by_sender(time_segments: List[Dict]):
    """Group messages by sender within each time segment."""
    for seg in time_segments:
        
        seg_messages: List[CleanedMessage] = seg["messages"]
        sender_groups = {}
        for msg in seg_messages:
            sender_groups.setdefault(msg.sender, []).append(msg)
    
        seg["sender_groups"] = {
            sender: {

                "message_count": len(msgs),
                "combined_text": "\n".join(m.text for m in msgs if m.text)
            }
            for sender, msgs in sender_groups.items()
        }
        
    return time_segments

