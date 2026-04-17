"""
MedRoute AI — Firebase Communication Module
======================================
Handles all Firebase Realtime Database operations:
- Sending ambulance detection alerts with full data schema
- Clearing detections and resetting signals
- Logging detection history for dashboard analytics
- Auto-reset timer for green corridor duration
"""

import firebase_admin
from firebase_admin import credentials, db
import datetime
import threading
import time
import os

from config import (
    FIREBASE_URL, SERVICE_ACCOUNT_KEY, GREEN_CORRIDOR_DURATION,
    AUTO_RESET_ENABLED, ALL_JUNCTION_IDS, VERSION, LOCATION
)

_firebase_initialized = False
_reset_timers = {}


def _init_firebase():
    """Initialize Firebase Admin SDK (singleton)."""
    global _firebase_initialized
    if _firebase_initialized:
        return True
    try:
        if not os.path.exists(SERVICE_ACCOUNT_KEY):
            print(f"[FIREBASE ERROR] Key not found: {SERVICE_ACCOUNT_KEY}")
            return False
        cred = credentials.Certificate(SERVICE_ACCOUNT_KEY)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_URL})
        _firebase_initialized = True
        print("[FIREBASE] OK - Connected to ev-priority-sys-001")
        return True
    except ValueError:
        _firebase_initialized = True
        return True
    except Exception as e:
        print(f"[FIREBASE ERROR] Init failed: {e}")
        return False


_init_firebase()


def _signal_for_lane(lane):
    """Green for ambulance lane, red for all others."""
    return {d: ("GREEN" if d == lane.lower() else "RED")
            for d in ["north", "south", "east", "west"]}


def _normal_signals():
    """Default Indian traffic pattern: N-S green, E-W red."""
    return {"north": "GREEN", "south": "GREEN", "east": "RED", "west": "RED"}


def send_detection(junction_id, lane, confidence, detection_method="YOLO",
                   ambulance_type="unknown", ambulance_color="white",
                   text_detected="none"):
    """
    Send ambulance detection alert to Firebase.
    Updates /junctions/{id}/ and logs to /history/{id}/.
    """
    if not _firebase_initialized:
        return False
    try:
        ts = datetime.datetime.now().isoformat()
        payload = {
            "ambulance_detected": True,
            "lane": lane.lower(),
            "confidence": round(confidence, 2),
            "priority": "HIGH",
            "detection_method": detection_method,
            "timestamp": ts,
            "signal_status": _signal_for_lane(lane),
            "ambulance_info": {
                "type": ambulance_type,
                "color": ambulance_color,
                "text_detected": text_detected
            }
        }
        db.reference(f"/junctions/{junction_id}").set(payload)

        # Log to history
        db.reference(f"/history/{junction_id}").push({
            "lane": lane.lower(), "confidence": round(confidence, 2),
            "detection_method": detection_method,
            "ambulance_type": ambulance_type,
            "ambulance_color": ambulance_color,
            "text_detected": text_detected,
            "timestamp": ts,
            "date": datetime.datetime.now().strftime("%Y-%m-%d")
        })

        print(f"[FIREBASE] OK - {junction_id} | {lane.upper()} | "
              f"{confidence:.2f} | {detection_method} | {ambulance_type}")

        if AUTO_RESET_ENABLED:
            _start_reset_timer(junction_id)
        return True
    except Exception as e:
        print(f"[FIREBASE ERROR] Send failed: {e}")
        return False


def clear_detection(junction_id):
    """Reset junction to normal signals."""
    if not _firebase_initialized:
        return False
    try:
        db.reference(f"/junctions/{junction_id}").set({
            "ambulance_detected": False, "lane": "none",
            "confidence": 0, "priority": "NORMAL",
            "detection_method": "none",
            "timestamp": datetime.datetime.now().isoformat(),
            "signal_status": _normal_signals(),
            "ambulance_info": {"type": "none", "color": "none", "text_detected": "none"}
        })
        print(f"[FIREBASE] OK - {junction_id} -> normal signals")
        return True
    except Exception as e:
        print(f"[FIREBASE ERROR] Clear failed: {e}")
        return False


def _start_reset_timer(junction_id):
    """Auto-reset junction after GREEN_CORRIDOR_DURATION seconds."""
    if junction_id in _reset_timers:
        _reset_timers[junction_id].cancel()
    def _reset():
        print(f"[FIREBASE] TIMER - Auto-reset: {junction_id} ({GREEN_CORRIDOR_DURATION}s elapsed)")
        clear_detection(junction_id)
        _reset_timers.pop(junction_id, None)
    t = threading.Timer(GREEN_CORRIDOR_DURATION, _reset)
    t.daemon = True
    t.start()
    _reset_timers[junction_id] = t


def initialize_junctions(junction_ids=None):
    """Initialize all junctions to normal state + set system metadata."""
    ids = junction_ids or ALL_JUNCTION_IDS
    if not _firebase_initialized:
        return
    for jid in ids:
        clear_detection(jid)
    try:
        db.reference("/system").set({
            "junctions": ids, "total_junctions": len(ids),
            "started_at": datetime.datetime.now().isoformat(),
            "version": VERSION, "location": LOCATION
        })
    except Exception:
        pass
    print(f"[FIREBASE] {len(ids)} junctions initialized")


if __name__ == "__main__":
    print("=" * 50)
    print("  ASEP — Firebase Connection Test")
    print("=" * 50)
    initialize_junctions()
    send_detection("J1", "west", 0.91, "YOLO+SYMBOL+COLOR",
                   "government", "white", "RED_CROSS")
    print(f"\nCheck Firebase console. Auto-reset in {GREEN_CORRIDOR_DURATION}s.")
    try:
        time.sleep(GREEN_CORRIDOR_DURATION + 2)
    except KeyboardInterrupt:
        print("Exiting...")
