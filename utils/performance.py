"""
MedRoute AI — Performance Monitoring
======================================
FPS counter, stats tracker, and latency recorder.
Extracted from detect.py with additions for per-subsystem latency tracking.
"""

import os
import time
import json
import datetime
from collections import deque
from logger import get_logger
from config import LOG_DIR

log = get_logger("performance")

os.makedirs(LOG_DIR, exist_ok=True)


class PerfMonitor:
    """Real-time FPS and frame-time monitor with HUD overlay."""

    def __init__(self, window=30):
        self._times = deque(maxlen=window)
        self._frame_start = 0
        self.current_fps = 0.0
        self.avg_fps = 0.0
        self.frame_ms = 0.0
        self._all_fps = []

    def tick_start(self):
        self._frame_start = time.perf_counter()

    def tick_end(self):
        now = time.perf_counter()
        self.frame_ms = (now - self._frame_start) * 1000
        self._times.append(now)
        if len(self._times) >= 2:
            elapsed = self._times[-1] - self._times[0]
            self.current_fps = (len(self._times) - 1) / elapsed if elapsed > 0 else 0
        self._all_fps.append(self.current_fps)
        self.avg_fps = sum(self._all_fps[-200:]) / min(len(self._all_fps), 200)

    def overlay(self, frame):
        import cv2
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (w - 220, 5), (w - 5, 75), (0, 0, 0), -1)
        cv2.rectangle(frame, (w - 220, 5), (w - 5, 75), (0, 255, 255), 1)
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (w - 210, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Avg: {self.avg_fps:.1f}", (w - 210, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)
        cv2.putText(frame, f"{self.frame_ms:.1f}ms", (w - 210, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 180), 1)


class StatsTracker:
    """Aggregate session statistics with periodic JSON save."""

    def __init__(self):
        self.start_time = time.time()
        self.total_frames = 0
        self.frames_dropped = 0
        self.total_detections = 0
        self.confidences = []
        self._path = os.path.join(LOG_DIR, "system_metrics.json")

    def record_detection(self, conf):
        self.total_detections += 1
        self.confidences.append(conf)

    def save(self):
        uptime = time.time() - self.start_time
        avg_conf = sum(self.confidences) / len(self.confidences) if self.confidences else 0
        drop_rate = self.frames_dropped / max(self.total_frames, 1) * 100
        data = {
            "uptime_seconds": round(uptime, 1),
            "total_frames": self.total_frames,
            "frames_dropped": self.frames_dropped,
            "frame_drop_rate_pct": round(drop_rate, 2),
            "total_detections": self.total_detections,
            "average_confidence": round(avg_conf, 3),
            "last_updated": datetime.datetime.now().isoformat(),
        }
        try:
            tmp = self._path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._path)
        except Exception as e:
            log.error(f"Metrics save failed: {e}")


class LatencyTracker:
    """Per-subsystem latency recorder. Saves to performance_metrics.json."""

    def __init__(self, save_interval=100):
        self._data = {
            "yolo_ms": [],
            "ocr_ms": [],
            "firebase_ms": [],
            "frame_capture_ms": [],
        }
        self._save_interval = save_interval
        self._call_count = 0
        self._path = os.path.join(LOG_DIR, "performance_metrics.json")

    def record(self, subsystem, latency_ms):
        """Record a latency sample for a subsystem."""
        if subsystem in self._data:
            self._data[subsystem].append(round(latency_ms, 2))
            # Keep only last 1000 samples per subsystem
            if len(self._data[subsystem]) > 1000:
                self._data[subsystem] = self._data[subsystem][-500:]
        self._call_count += 1
        if self._call_count % self._save_interval == 0:
            self.save()

    def save(self):
        """Write latency summary to JSON."""
        summary = {}
        for key, samples in self._data.items():
            if samples:
                summary[key] = {
                    "count": len(samples),
                    "mean": round(sum(samples) / len(samples), 2),
                    "min": round(min(samples), 2),
                    "max": round(max(samples), 2),
                    "last": round(samples[-1], 2),
                }
            else:
                summary[key] = {"count": 0}
        summary["last_updated"] = datetime.datetime.now().isoformat()
        try:
            tmp = self._path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(summary, f, indent=2)
            os.replace(tmp, self._path)
        except Exception as e:
            log.error(f"Performance metrics save failed: {e}")
