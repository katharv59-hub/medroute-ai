"""
MedRoute AI — Frame Capture Thread
=====================================
Thread A: Continuously reads video frames into a thread-safe queue.
Handles camera failures, reconnection, and file-based video looping.
"""

import threading
import time
import os
import cv2
from logger import get_logger
from config import CAMERA_FAIL_THRESHOLD, CAMERA_RECONNECT_TRIES, CAMERA_RECONNECT_DELAY

log = get_logger("capture")


class FrameCapture(threading.Thread):
    """Dedicated frame capture thread that feeds frames into a queue.

    Puts (frame_number, frame) tuples into the output queue.
    Puts None when capture is exhausted or shutdown is signaled.
    """

    def __init__(self, video_source, frame_queue, latency_tracker=None):
        super().__init__(daemon=True, name="FrameCapture")
        self._source = video_source
        self._queue = frame_queue
        self._shutdown = threading.Event()
        self._latency_tracker = latency_tracker
        self._cap = None
        self._frame_count = 0
        self.fw = 0
        self.fh = 0
        self.src_fps = 30.0
        self._ready = threading.Event()

    def _open_camera(self):
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            log.error(f"Cannot open: {self._source}")
            return None
        self.fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        return cap

    def _reconnect(self):
        for attempt in range(1, CAMERA_RECONNECT_TRIES + 1):
            log.warning(f"Camera reconnect attempt {attempt}/{CAMERA_RECONNECT_TRIES}...")
            time.sleep(CAMERA_RECONNECT_DELAY)
            cap = self._open_camera()
            if cap:
                log.info("Camera reconnected ✓")
                return cap
        log.error("Camera reconnect failed — giving up.")
        return None

    def run(self):
        self._cap = self._open_camera()
        if not self._cap:
            self._queue.put(None)
            return

        self._ready.set()
        log.info(f"Capture thread started: {self.fw}x{self.fh} @ {self.src_fps:.0f}fps")
        fail_count = 0

        while not self._shutdown.is_set():
            t0 = time.perf_counter()
            ret, frame = self._cap.read()

            if not ret:
                fail_count += 1
                if fail_count >= CAMERA_FAIL_THRESHOLD:
                    if isinstance(self._source, str) and os.path.isfile(self._source):
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        fail_count = 0
                        continue
                    self._cap.release()
                    self._cap = self._reconnect()
                    if not self._cap:
                        break
                    fail_count = 0
                continue

            fail_count = 0
            self._frame_count += 1

            elapsed_ms = (time.perf_counter() - t0) * 1000
            if self._latency_tracker:
                self._latency_tracker.record("frame_capture_ms", elapsed_ms)

            # Drop old frames if queue is full (non-blocking)
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Exception:
                    pass

            self._queue.put((self._frame_count, frame))

        # Cleanup
        if self._cap:
            self._cap.release()
        self._queue.put(None)  # Sentinel for consumer
        log.info(f"Capture thread stopped. Total frames read: {self._frame_count}")

    def stop(self):
        self._shutdown.set()

    def wait_ready(self, timeout=10):
        """Block until the camera is opened and first frame is available."""
        return self._ready.wait(timeout=timeout)

    @property
    def frame_count(self):
        return self._frame_count
