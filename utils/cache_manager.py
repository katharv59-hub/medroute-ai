"""
MedRoute AI — Thread-Safe Cache Manager
=========================================
Memory-safe OCR result cache with automatic expiry.
Prevents unbounded memory growth during long-running sessions.
"""

import threading
import time
from logger import get_logger

log = get_logger("cache")


class CacheManager:
    """Thread-safe cache with per-entry TTL and automatic cleanup."""

    def __init__(self, expiry_seconds=30):
        self._lock = threading.Lock()
        self._data = {}          # key -> {"value": ..., "last_seen": timestamp}
        self._expiry = expiry_seconds
        self._last_ocr_frame = {}  # key -> last frame number OCR was requested

    def get(self, key):
        """Return cached value or None if missing/expired."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            # Check expiry
            if time.monotonic() - entry["last_seen"] > self._expiry:
                del self._data[key]
                self._last_ocr_frame.pop(key, None)
                return None
            return entry["value"]

    def set(self, key, value):
        """Store a value, refreshing its timestamp."""
        with self._lock:
            self._data[key] = {
                "value": value,
                "last_seen": time.monotonic(),
            }

    def touch(self, key):
        """Refresh the timestamp of an existing entry without changing the value."""
        with self._lock:
            entry = self._data.get(key)
            if entry is not None:
                entry["last_seen"] = time.monotonic()

    def should_ocr(self, key, current_frame, interval):
        """Check if OCR should be requested for this object (per-object rate limit).

        Returns True if enough frames have passed since last OCR for this key.
        """
        with self._lock:
            # Already have a positive result — never re-OCR
            entry = self._data.get(key)
            if entry is not None and entry["value"][0]:
                return False
            last = self._last_ocr_frame.get(key, -interval - 1)
            if current_frame - last >= interval:
                self._last_ocr_frame[key] = current_frame
                return True
            return False

    def cleanup(self):
        """Remove all expired entries. Call periodically."""
        now = time.monotonic()
        removed = 0
        with self._lock:
            expired_keys = [
                k for k, v in self._data.items()
                if now - v["last_seen"] > self._expiry
            ]
            for k in expired_keys:
                del self._data[k]
                self._last_ocr_frame.pop(k, None)
                removed += 1
        if removed > 0:
            log.debug(f"Cache cleanup: removed {removed} expired entries")
        return removed

    @property
    def size(self):
        with self._lock:
            return len(self._data)
