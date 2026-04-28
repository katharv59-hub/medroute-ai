"""
MedRoute AI — Firebase Worker Thread
=======================================
Async queue-based Firebase writer with retry and exponential backoff.
Decouples Firebase I/O from the detection pipeline — never blocks inference.
"""

import threading
import time
from queue import Queue, Empty
from logger import get_logger

log = get_logger("firebase_worker")


class FirebaseWorker(threading.Thread):
    """Background worker that processes Firebase write/clear events from a queue.

    Events are dicts with:
        {"action": "send"|"clear", "args": (...), "kwargs": {...}}
    """

    def __init__(self, queue, max_retries=3, base_delay=0.5, latency_tracker=None):
        super().__init__(daemon=True, name="FirebaseWorker")
        self._queue = queue
        self._shutdown = threading.Event()
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._latency_tracker = latency_tracker
        self._processed = 0
        self._failed = 0

    def run(self):
        log.info("Firebase worker started.")
        # Lazy import to avoid circular imports at module level
        from firebase_sender import send_detection, clear_detection

        while not self._shutdown.is_set():
            try:
                event = self._queue.get(timeout=0.2)
            except Empty:
                continue

            action = event.get("action")
            args = event.get("args", ())
            kwargs = event.get("kwargs", {})

            t0 = time.perf_counter()
            success = False
            delay = self._base_delay

            for attempt in range(1, self._max_retries + 1):
                try:
                    if action == "send":
                        send_detection(*args, **kwargs)
                    elif action == "clear":
                        clear_detection(*args, **kwargs)
                    success = True
                    break
                except Exception as e:
                    if attempt == self._max_retries:
                        log.error(f"Firebase {action} failed after {attempt} attempts: {e}")
                    else:
                        log.warning(f"Firebase {action} attempt {attempt} failed: {e}. "
                                    f"Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        delay *= 2

            elapsed_ms = (time.perf_counter() - t0) * 1000
            if self._latency_tracker:
                self._latency_tracker.record("firebase_ms", elapsed_ms)

            if success:
                self._processed += 1
            else:
                self._failed += 1

            self._queue.task_done()

        log.info(f"Firebase worker stopped. Processed: {self._processed}, Failed: {self._failed}")

    def stop(self):
        """Signal the worker to stop after draining the queue."""
        self._shutdown.set()

    @property
    def stats(self):
        return {
            "processed": self._processed,
            "failed": self._failed,
            "queue_depth": self._queue.qsize(),
        }


def enqueue_send(queue, junction_id, lane, confidence,
                 detection_method="YOLO", ambulance_type="unknown",
                 ambulance_color="white", text_detected="none"):
    """Non-blocking enqueue of a Firebase send event."""
    try:
        queue.put_nowait({
            "action": "send",
            "args": (junction_id, lane, confidence,
                     detection_method, ambulance_type,
                     ambulance_color, text_detected),
        })
    except Exception:
        log.warning("Firebase queue full — dropping event.")


def enqueue_clear(queue, junction_id):
    """Non-blocking enqueue of a Firebase clear event."""
    try:
        queue.put_nowait({
            "action": "clear",
            "args": (junction_id,),
        })
    except Exception:
        log.warning("Firebase queue full — dropping clear event.")
