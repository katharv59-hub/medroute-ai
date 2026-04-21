"""
MedRoute AI — CSV Logger
=========================
Saves every confirmed ambulance detection to a local CSV file.
Works independently of Firebase — no data is lost even if the
network or Firebase goes down.

Output: logs/detections.csv
"""

import csv
import os
import datetime

from config import LOG_DIR

CSV_PATH = os.path.join(LOG_DIR, "detections.csv")

_HEADERS = [
    "timestamp", "date", "time", "junction", "lane",
    "confidence", "detection_method", "ambulance_type",
    "ambulance_color", "text_detected"
]


def _ensure_csv():
    """Create logs/ dir and CSV with headers if they don't exist."""
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_HEADERS)


def log_detection(junction_id, lane, confidence, detection_method="YOLO",
                  ambulance_type="unknown", ambulance_color="white",
                  text_detected="none"):
    """
    Append one detection row to the CSV file.
    Safe to call from any thread — opens/closes the file each time.
    """
    _ensure_csv()
    now = datetime.datetime.now()
    row = [
        now.isoformat(),
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        junction_id,
        lane.lower(),
        round(confidence, 2),
        detection_method,
        ambulance_type,
        ambulance_color,
        text_detected,
    ]
    try:
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
    except Exception as e:
        print(f"[CSV ERROR] Failed to write: {e}")


def get_csv_path():
    """Return the absolute path to the CSV file."""
    return CSV_PATH
