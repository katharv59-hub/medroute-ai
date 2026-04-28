"""
MedRoute AI — OCR Worker Thread
==================================
Async OCR processing that runs EasyOCR off the main detection loop.
Results are stored in the shared CacheManager for the dispatcher to read.
"""

import threading
import time
from queue import Queue, Empty
from logger import get_logger

log = get_logger("ocr_worker")


class OCRWorker(threading.Thread):
    """Background worker that processes OCR requests from a queue.

    Requests are dicts with:
        {"det_id": str, "crop": ndarray, "frame_num": int}

    Results are stored in the shared CacheManager.
    """

    def __init__(self, queue, cache, gpu=True, keywords=None, latency_tracker=None):
        super().__init__(daemon=True, name="OCRWorker")
        self._queue = queue
        self._cache = cache
        self._gpu = gpu
        self._keywords = keywords or []
        self._latency_tracker = latency_tracker
        self._shutdown = threading.Event()
        self._reader = None
        self._processed = 0

    def _get_reader(self):
        """Lazy-load EasyOCR reader in the worker thread."""
        if self._reader is None:
            try:
                import easyocr
                log.info("OCR Worker: Loading EasyOCR...")
                self._reader = easyocr.Reader(['en'], gpu=self._gpu)
                log.info("OCR Worker: EasyOCR ready.")
            except Exception as e:
                log.error(f"OCR Worker: Failed to load EasyOCR: {e}")
        return self._reader

    def run(self):
        log.info("OCR worker started.")

        while not self._shutdown.is_set():
            try:
                request = self._queue.get(timeout=0.2)
            except Empty:
                continue

            det_id = request["det_id"]
            crop = request["crop"]

            # Skip if already have a positive result
            cached = self._cache.get(det_id)
            if cached is not None and cached[0]:
                self._queue.task_done()
                continue

            t0 = time.perf_counter()
            reader = self._get_reader()
            if reader is None:
                self._queue.task_done()
                continue

            try:
                results = reader.readtext(crop)
                found = False
                for (_, text, prob) in results:
                    if prob < 0.4:
                        continue
                    tu = text.upper().strip()
                    for kw in self._keywords:
                        if kw in tu or tu in kw or kw in tu[::-1]:
                            self._cache.set(det_id, (True, tu))
                            found = True
                            break
                    if found:
                        break
                if not found:
                    self._cache.set(det_id, (False, ""))
            except Exception as e:
                log.error(f"OCR Worker: processing error for {det_id}: {e}")
                self._cache.set(det_id, (False, ""))

            elapsed_ms = (time.perf_counter() - t0) * 1000
            if self._latency_tracker:
                self._latency_tracker.record("ocr_ms", elapsed_ms)

            self._processed += 1
            self._queue.task_done()

        log.info(f"OCR worker stopped. Processed: {self._processed}")

    def stop(self):
        """Signal the worker to stop."""
        self._shutdown.set()

    @property
    def stats(self):
        return {
            "processed": self._processed,
            "queue_depth": self._queue.qsize(),
        }


def enqueue_ocr(queue, det_id, crop, frame_num):
    """Non-blocking enqueue of an OCR request."""
    try:
        queue.put_nowait({
            "det_id": det_id,
            "crop": crop,
            "frame_num": frame_num,
        })
    except Exception:
        log.debug(f"OCR queue full — skipping {det_id}")
