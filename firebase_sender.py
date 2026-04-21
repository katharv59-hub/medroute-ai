"""
MedRoute AI — Firebase Communication Module  v3.0
==================================================
Smart Firebase writes with:
- State-diff throttling (skip identical writes)
- Confidence-delta gating
- Retry with exponential backoff
- Auto-reconnect on failure
- Full structured logging
"""

import firebase_admin
from firebase_admin import credentials, db
import datetime
import threading
import time
import os

from config import (
    FIREBASE_URL, SERVICE_ACCOUNT_KEY, GREEN_CORRIDOR_DURATION,
    AUTO_RESET_ENABLED, ALL_JUNCTION_IDS, VERSION, LOCATION,
    FIREBASE_MIN_INTERVAL_S, FIREBASE_RETRY_ATTEMPTS,
    FIREBASE_RETRY_BASE_DELAY, FIREBASE_CONF_CHANGE_MIN,
)
from logger import get_logger

log = get_logger("firebase")

_firebase_initialized = False
_reset_timers         = {}

# ── throttle state ─────────────────────────────────────────────────────────
# Per junction: {"lane", "confidence", "ambulance_detected", "last_sent_ts"}
_last_state: dict = {}


# ============================================================
#  INIT / RECONNECT
# ============================================================
def _init_firebase() -> bool:
    global _firebase_initialized
    if _firebase_initialized:
        return True
    try:
        if not SERVICE_ACCOUNT_KEY or not os.path.exists(SERVICE_ACCOUNT_KEY):
            log.warning("Service key not found — Firebase integration disabled.")
            return False
        cred = credentials.Certificate(SERVICE_ACCOUNT_KEY)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_URL})
        _firebase_initialized = True
        log.info("Firebase connected ✓ (ev-priority-sys-001)")
        return True
    except ValueError:
        # App already initialized
        _firebase_initialized = True
        return True
    except Exception as exc:
        log.error(f"Firebase init failed: {exc}")
        return False


_init_firebase()


# ============================================================
#  RETRY HELPER
# ============================================================
def _with_retry(fn, *args, **kwargs):
    """Call fn(*args) with exponential-backoff retry."""
    delay = FIREBASE_RETRY_BASE_DELAY
    for attempt in range(1, FIREBASE_RETRY_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == FIREBASE_RETRY_ATTEMPTS:
                log.error(f"Firebase write failed after {attempt} attempts: {exc}")
                # Flag for reconnect on next call
                global _firebase_initialized
                _firebase_initialized = False
                raise
            log.warning(f"Firebase write attempt {attempt} failed ({exc}). "
                        f"Retrying in {delay:.1f}s…")
            time.sleep(delay)
            delay *= 2
            # Attempt reconnect before next retry
            if not _firebase_initialized:
                _init_firebase()


# ============================================================
#  SIGNAL HELPERS
# ============================================================
def _signal_for_lane(lane: str) -> dict:
    return {d: ("GREEN" if d == lane.lower() else "RED")
            for d in ["north", "south", "east", "west"]}


def _normal_signals() -> dict:
    return {"north": "GREEN", "south": "GREEN", "east": "RED", "west": "RED"}


# ============================================================
#  THROTTLE CHECK
# ============================================================
def _should_skip_write(junction_id: str, lane: str,
                       confidence: float, ambulance_detected: bool) -> bool:
    """Return True if the write is redundant and within the throttle window."""
    now  = time.monotonic()
    prev = _last_state.get(junction_id)
    if prev is None:
        return False

    age = now - prev["last_sent_ts"]
    if age < FIREBASE_MIN_INTERVAL_S:
        # Within throttle window — only write if something significant changed
        same_lane  = prev["lane"] == lane.lower()
        same_state = prev["ambulance_detected"] == ambulance_detected
        conf_delta = abs(prev["confidence"] - confidence)
        if same_lane and same_state and conf_delta < FIREBASE_CONF_CHANGE_MIN:
            return True  # skip
    return False


def _record_state(junction_id: str, lane: str,
                  confidence: float, ambulance_detected: bool):
    _last_state[junction_id] = {
        "lane":               lane.lower(),
        "confidence":         round(confidence, 2),
        "ambulance_detected": ambulance_detected,
        "last_sent_ts":       time.monotonic(),
    }


# ============================================================
#  PUBLIC API
# ============================================================
def send_detection(junction_id: str, lane: str, confidence: float,
                   detection_method: str = "YOLO",
                   ambulance_type: str = "unknown",
                   ambulance_color: str = "white",
                   text_detected: str = "none") -> bool:
    """
    Send ambulance detection to Firebase.
    Skips write if state unchanged and within throttle window.
    """
    if not _firebase_initialized:
        if not _init_firebase():
            return False

    if _should_skip_write(junction_id, lane, confidence, True):
        log.debug(f"[{junction_id}] Throttled — no state change, skipping write.")
        return True

    ts = datetime.datetime.now().isoformat()
    payload = {
        "ambulance_detected": True,
        "lane":               lane.lower(),
        "confidence":         round(confidence, 2),
        "priority":           "HIGH",
        "detection_method":   detection_method,
        "timestamp":          ts,
        "signal_status":      _signal_for_lane(lane),
        "ambulance_info": {
            "type":          ambulance_type,
            "color":         ambulance_color,
            "text_detected": text_detected,
        }
    }

    try:
        _with_retry(db.reference(f"/junctions/{junction_id}").set, payload)
        _with_retry(db.reference(f"/history/{junction_id}").push, {
            "lane":            lane.lower(),
            "confidence":      round(confidence, 2),
            "detection_method": detection_method,
            "ambulance_type":  ambulance_type,
            "ambulance_color": ambulance_color,
            "text_detected":   text_detected,
            "timestamp":       ts,
            "date":            datetime.datetime.now().strftime("%Y-%m-%d"),
        })
        _record_state(junction_id, lane, confidence, True)
        log.info(f"[{junction_id}] SENT → {lane.upper()} | "
                 f"conf={confidence:.2f} | {detection_method} | {ambulance_type}")
        if AUTO_RESET_ENABLED:
            _start_reset_timer(junction_id)
        return True
    except Exception:
        return False


def clear_detection(junction_id: str) -> bool:
    """Reset junction to normal signals."""
    if not _firebase_initialized:
        if not _init_firebase():
            return False

    if _should_skip_write(junction_id, "none", 0.0, False):
        return True

    try:
        _with_retry(db.reference(f"/junctions/{junction_id}").set, {
            "ambulance_detected": False,
            "lane":               "none",
            "confidence":         0,
            "priority":           "NORMAL",
            "detection_method":   "none",
            "timestamp":          datetime.datetime.now().isoformat(),
            "signal_status":      _normal_signals(),
            "ambulance_info":     {"type": "none", "color": "none", "text_detected": "none"},
        })
        _record_state(junction_id, "none", 0.0, False)
        log.info(f"[{junction_id}] Cleared → normal signals")
        return True
    except Exception:
        return False


# ============================================================
#  AUTO-RESET TIMER
# ============================================================
def _start_reset_timer(junction_id: str):
    if junction_id in _reset_timers:
        _reset_timers[junction_id].cancel()

    def _reset():
        log.info(f"[{junction_id}] Auto-reset after {GREEN_CORRIDOR_DURATION}s.")
        clear_detection(junction_id)
        _reset_timers.pop(junction_id, None)

    t = threading.Timer(GREEN_CORRIDOR_DURATION, _reset)
    t.daemon = True
    t.start()
    _reset_timers[junction_id] = t


# ============================================================
#  JUNCTION INIT
# ============================================================
def initialize_junctions(junction_ids=None):
    ids = junction_ids or ALL_JUNCTION_IDS
    if not _firebase_initialized:
        return
    for jid in ids:
        clear_detection(jid)
    try:
        _with_retry(db.reference("/system").set, {
            "junctions":       ids,
            "total_junctions": len(ids),
            "started_at":      datetime.datetime.now().isoformat(),
            "version":         VERSION,
            "location":        LOCATION,
        })
    except Exception:
        pass
    log.info(f"{len(ids)} junction(s) initialized.")


# ============================================================
#  STANDALONE TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  ASEP — Firebase Connection Test")
    print("=" * 50)
    initialize_junctions()
    send_detection("J1", "west", 0.91, "YOLO+SYMBOL+COLOR", "government", "white", "RED_CROSS")
    print(f"\nCheck Firebase console. Auto-reset in {GREEN_CORRIDOR_DURATION}s.")
    try:
        time.sleep(GREEN_CORRIDOR_DURATION + 2)
    except KeyboardInterrupt:
        print("Exiting...")
