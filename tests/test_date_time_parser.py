
import re
from datetime import datetime
from typing import Optional

dayfirst_override = None  # Can be set to True/False to override dayfirst detection

def normalize_year(y: int) -> int:
        return y + 2000 if y < 100 else y

def parse_datetime_components(date_str: str, time_str: str, is_dayfirst: bool) -> Optional[datetime]:
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
        
        year = normalize_year(p2)
        dayfirst = None
        # decide day/month order
        if dayfirst is not None:
            dayfirst = dayfirst_override
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
            

if __name__ == "__main__":
    test_inputs = [
        ("27/05/21", "7:28 PM"),
        ("1/1/21", "12:00 AM"),
        ("12/11/2020", "11:59 PM"),
        ("05-09-19", "9:05 am"),
        ("9.7.2018", "10:15 Am"),

        ("27/05/21", "19:28"),
        ("01/01/2021", "00:00"),
        ("12/11/2020", "23:59"),
        ("5-9-19", "09:05"),
        ("9.7.2018", "22:15"),

        ("27/05/21", "19:28:45"),
        ("01/01/2021", "00:00"),
        ("12/11/2020", "23:59:59"),
        ("5/9/19", "09:05:01"),
        ("9/7/2018", "22:15"),

        ("05/27/21", "7:28 PM"),
        ("12/31/2020", "11:59 PM"),
        ("01-01-21", "12:00 AM"),
        ("7.4.19", "10:15 am"),

        ("27/05/21", "7:28:59 PM"),
        ("01/01/21", "0:0"),
        ("12-11-20", "23"),
        ("31/04/21", "10:30"),
        ("29/02/21", "08:00"),
    ]

    for date_time in test_inputs:

        dt = parse_datetime_components(date_time[0], date_time[1], is_dayfirst=True)
        print(f"Raw form: {date_time} Parsed datetime: {dt}")