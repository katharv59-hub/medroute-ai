"""
MedRoute AI — Frame Utilities
===============================
Lane detection, vehicle type classification, and priority helpers.
Extracted from detect.py — logic is identical.
"""


def detect_lane(x1, y1, x2, y2, fw, fh):
    """Determine which lane a bounding box falls in."""
    cx, cy = (x1 + x2) / 2 / fw, (y1 + y2) / 2 / fh
    if cy < 0.35:
        return "north"
    if cy > 0.65:
        return "south"
    if cx < 0.4:
        return "west"
    return "east"


def classify_type(color, has_red, text):
    """Classify ambulance type from color, red-cross presence, and OCR text."""
    t = text.upper()
    if "108" in t:
        return "108_service"
    if color == "white" and has_red:
        return "government"
    if color == "white":
        return "government"
    return "private"


def lane_priority(lane):
    """Return numeric priority for a lane direction."""
    return {"north": 4, "south": 3, "east": 2, "west": 1}.get(lane, 0)
