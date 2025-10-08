import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Iterable, Callable
from datetime import datetime
import re


@dataclass
class CleanedMessage:
    """Structured representation of a single chat message."""
    timestamp: Optional[datetime]
    sender: Optional[str]
    text: str
    raw: str = ""


class WhatsAppChatParser:
    """
    Minimal class-based template for parsing WhatsApp exported .txt files.

    Intended workflow:
      1. detect_format(lines) -> 'ios' | 'android' | 'unknown'
      2. parse_file(path) -> reads file, picks parser and returns List[CleanedMessage]
      3. _parse_ios / _parse_android -> implement platform-specific parsing
      4. helpers: _merge_continuations, _is_message_start, _parse_timestamp, ...
    """

    # how many lines to inspect when guessing day-first vs month-first
    _GUESS_SAMPLE_LINES = 200

    def __init__(self, tz: Optional[str] = None, dayfirst: Optional[bool] = None):
        """
        tz: reserved for future timezone handling (not used in this simple version).
        dayfirst: if None, will attempt to guess from data; else forces dayfirst or monthfirst parsing.
        """
        self.tz = tz
        self.dayfirst_override = dayfirst

        # Precompile regexes once for speed:
        # Android: e.g. "27/05/21, 7:28 PM - Alice: Hello"
        # allow optional seconds in time, optional AM/PM
        self._android_msg_re = re.compile(
            r"^(?P<date>\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4}),\s"
            r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APMapm]{2})?)\s-\s"
            r"(?:(?P<sender>[^:]+):\s)?(?P<message>.*)$"
        )

        # iOS: e.g. "[01/13/24, 12:24:48 AM] Alex: Hello"
        self._ios_msg_re = re.compile(
            r"^\[(?P<date>\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4}),\s"
            r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APMapm]{2})?)\]\s"
            r"(?:(?P<sender>[^:]+):\s)?(?P<message>.*)$"
        )

        # Simple detectors for message-start lines (used in merging)
        self._android_start_re = re.compile(
            r"^\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4},\s\d{1,2}:\d{2}"
        )
        self._ios_start_re = re.compile(
            r"^\[\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4},\s\d{1,2}:\d{2}"
        )

    def detect_format(self, lines: Iterable[str]) -> str:
        """
        Heuristic detection'
        """
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("[") and self._ios_start_re.match(line):
                return "ios"
            if self._android_start_re.match(line) and " - " in line:
                return "android"
        return "unknown"

    def parse_file(self, path: str) -> List[CleanedMessage]:
        """Open a .txt export, normalize into full messages, then parse."""
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            raw_lines = f.read().splitlines()

        format = self.detect_format(raw_lines[:50])

        if format == 'ios':
            normalized_lines = self._merge_continuations(raw_lines, self._is_ios_message_start)
            return self._parse_ios(normalized_lines)

        elif format == 'android':
            normalized_lines = self._merge_continuations(raw_lines, self._is_android_message_start)
            return self._parse_android(normalized_lines)

        else:
            #
            merged_android = self._merge_continuations(raw_lines, self._is_android_message_start)
            try: 
                return self._parse_android(merged_android)
            except Exception:
                merged_ios = self._merge_continuations(raw_lines, self._is_ios_message_start)
                return self._parse_ios(merged_ios)
    
    # --------------------
    # Platform parsers (stubs)
    # --------------------
    def _parse_android(self, lines: List[str]) -> List[CleanedMessage]:
        """
        Parse Android-style logical message lines (one entry == one message).
        Each line expected like:
            "27/05/21, 7:28 PM - Alice: Hello everyone!"
            or system lines without "Name:" like "27/05/21, 19:31 - Alice left the group"
        Returns: List[CleanedMessage]
        """
        messages: List[CleanedMessage] = []
        android_re = self._android_msg_re
       
        
        guessed = self._guess_dayfirst(lines, platform="android")
            

        for line in lines:
            if not line.strip():
                continue
            m = android_re.match(line)
            if not m:
                # fallback: try to salvage by splitting on " - " once
                if " - " in line:
                    try:
                        header, msg_txt = line.split(" - ", 1)
                        # header like "27/05/21, 7:28 PM"
                        # attempt to split header into date and time
                        if "," in header:
                            date_part, time_part = header.split(",", 1)
                            dt = self._parse_datetime_components(date_part.strip(), time_part.strip(), is_dayfirst=guessed)
                        else:
                            dt = None
                        # attempt to extract sender if colon present
                        sender = None
                        if ":" in msg_txt:
                            sender, msg_body = msg_txt.split(": ", 1)
                            msg_body = msg_body
                        else:
                            msg_body = msg_txt
                        messages.append(CleanedMessage(timestamp=dt, sender=(sender.strip() if sender else None),
                                                text=msg_body.strip(), raw=line))
                        continue
                    except Exception:
                        # ultimate fallback: put raw line with None timestamp
                        messages.append(CleanedMessage(timestamp=None, sender=None, text=line.strip(), raw=line))
                        continue
                else:
                    messages.append(CleanedMessage(timestamp=None, sender=None, text=line.strip(), raw=line))
                    continue

            date_part = m.group("date")
            time_part = m.group("time")
            sender = m.group("sender")
            message_text = m.group("message") or ""
            dt = self._parse_datetime_components(date_part, time_part, is_dayfirst=guessed)

            if sender is not None:
                sender = sender.strip()
            else:
                sender = None  # system message

            messages.append(CleanedMessage(timestamp=dt, sender=sender, text=message_text.strip(), raw=line))

        return messages

    def _parse_ios(self, lines: List[str]) -> List[CleanedMessage]:
        """
        Parse iOS-style logical message lines (one entry == one message).
        Each line expected like:
            "[01/13/24, 12:24:48 AM] Alex: Have you finished?"
        or system messages without "Name:"
        """
        messages: List[CleanedMessage] = []
        ios_re = self._ios_msg_re

        
        guessed = self._guess_dayfirst(lines, platform="ios")

        for line in lines:
            if not line.strip():
                continue
            m = ios_re.match(line)
            if not m:
                # fallback attempt: treat as raw (no timestamp)
                messages.append(CleanedMessage(timestamp=None, sender=None, text=line.strip(), raw=line))
                continue

            date_part = m.group("date")
            time_part = m.group("time")
            sender = m.group("sender")
            message_text = m.group("message") or ""
            dt = self._parse_datetime_components(date_part, time_part, is_dayfirst=guessed)

            if sender is not None:
                sender = sender.strip()
            else:
                sender = None

            messages.append(CleanedMessage(timestamp=dt, sender=sender, text=message_text.strip(), raw=line))

        return messages

    # --------------------
    # Helper stubs
    # --------------------
    def _normalize_year(self, y: int) -> int:
        return y + 2000 if y < 100 else y

    def _merge_continuations(self, lines: List[str], is_start_predicate: Callable[[str], bool]) -> List[str]:
        """
        Merge wrapped/continued lines into complete logical messages.
        - New messages always match the timestamp+sender format.
        - Continuation lines do not have a timestamp -> must be merged with last.
        """
        merged: List[str] = []
        buffer: Optional[str] = None 
        is_start = is_start_predicate

        for line in lines:
            if is_start(line):
                if buffer is not None:  
                    merged.append(buffer)
                buffer = line
            else:
                if  buffer is None: buffer = line
                else: buffer += "\n" + line
        if buffer is not None:
            merged.append(buffer)
        return merged


    def _is_android_message_start(self, line: str) -> bool:
        if not line:
            return False
        return bool(self._android_start_re.match(line))

    def _is_ios_message_start(self, line: str) -> bool:
        if not line:
            return False
        return bool(self._ios_start_re.match(line))

    def _guess_dayfirst(self, lines: List[str], platform: str = "android") -> Optional[bool]:
        """
        Inspect sample message-start lines to infer whether dates are day-first (DD/MM) or month-first (MM/DD).
        Returns True (dayfirst), False (monthfirst), or None (undetermined).
        """
        
        
        first_gt12 = 0
        second_gt12 = 0
        total_checked = 0
        pattern = self._android_msg_re if platform == "android" else self._ios_msg_re

        for i, line in enumerate(lines):
            if i >= self._GUESS_SAMPLE_LINES:
                break
            m = pattern.match(line)
            if not m:
                continue
            dt = m.group("date")
            parts = re.split(r"[\/\.\-]", dt)
            if len(parts) < 3:
                continue
            try:
                a = int(parts[0])
                b = int(parts[1])
            except ValueError:
                continue
            total_checked += 1
            if a > 12:
                first_gt12 += 1
            if b > 12:
                second_gt12 += 1

        # decide
        if first_gt12 > 0 and second_gt12 == 0:
            return True
        if second_gt12 > 0 and first_gt12 == 0:
            return False
        # indeterminate
        return True if platform == "android" else False
    
    def _parse_datetime_components(self, date_str: str, time_str: str, is_dayfirst: bool) -> Optional[datetime]:
        """Parse date and time components into a datetime object."""
        date_str = date_str.strip()
        time_str = time_str.strip()

        date_parts = re.split(r"[\/\.\-]", date_str)
        if len(date_parts) < 3:
            return None
        try:
            p0, p1, p2 = int(date_parts[0]), int(date_parts[1]), int(date_parts[2])
        except ValueError:
            return None
        
        year = self._normalize_year(p2)

        # decide day/month order
        if self.dayfirst_override is not None:
            dayfirst = self.dayfirst_override
        else:
            dayfirst = is_dayfirst
            
        if dayfirst:
            day, month = p0, p1
        else:
            month, day = p0, p1

        # validate:
        if month > 12 and day <= 12:
            month, day = day, month

        t = time_str.strip()
        ampm = None
    
        m_am = re.search(r"\b([APMapm]{2})\b", t)
        if m_am:
            ampm = m_am.group(1).upper()
            t = re.sub(r"\s*[APMapm]{2}\s*$", "", t).strip()

        time_parts = t.split(":")
        try:
            if len(time_parts) == 2:
                hour = int(time_parts[0])
                minute = int(time_parts[1])
                second = 0
            elif len(time_parts) >= 3:
                hour = int(time_parts[0])
                minute = int(time_parts[1])
                second = int(re.sub(r"\D.*$", "", time_parts[2]))  # remove trailing text if any
            else:
                return None
        except ValueError:
            return None

        # convert 12-hour to 24-hour if AM/PM present
        if ampm:
            if ampm == "AM":
                if hour == 12:
                    hour = 0
            else:  # PM
                if hour != 12:
                    hour = hour + 12

        # basic sanity check
        try:
            return datetime(year, month, day, hour, minute, second)
        except Exception:
            # try swapping day/month if that fixes it
            try:
                return datetime(year, day, month, hour, minute, second)
            except Exception:
                return None
    


    # Add more helpers as you need:
    # - _parse_android_line -> parse timestamp, sender, msg_text
    # - _parse_ios_line -> parse timestamp, sender, msg_text
    # - _parse_timestamp -> robust parsing with locale flexibility




