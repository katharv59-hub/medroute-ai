"""
MedRoute AI — Detection Dispatcher
=====================================
Manages the voting system, confirmation tracker, and detection bookkeeping.
Extracted from detect.py — logic preserved exactly.
"""

from config import CONFIRM_FRAMES, VOTING_THRESHOLD
from logger import get_logger

log = get_logger("dispatcher")


class ConfirmTracker:
    """Temporal confirmation: requires N consecutive positive frames."""

    def __init__(self, required=CONFIRM_FRAMES):
        self.required = required
        self._streaks = {}
        self._confirmed = set()

    def update(self, track_id, detected):
        if detected:
            self._streaks[track_id] = self._streaks.get(track_id, 0) + 1
            if self._streaks[track_id] >= self.required:
                self._confirmed.add(track_id)
                return True
        else:
            self._streaks[track_id] = 0
            self._confirmed.discard(track_id)
        return track_id in self._confirmed

    def is_confirmed(self, track_id):
        return track_id in self._confirmed

    def reset(self, track_id):
        self._streaks.pop(track_id, None)
        self._confirmed.discard(track_id)


def evaluate_detection(tid, x1, y1, x2, y2, conf, label,
                       frame, fc, cache, color_detector,
                       ocr_queue, fps, ocr_every_n):
    """Run the 3-method voting system on a single tracked detection.

    Returns:
        dict with votes, methods, sym_ok, sym_text, col_ok, dom_col, has_red
    """
    import cv2
    from detectors.ocr_detector import detect_red_cross_strict, prepare_ocr_crop
    from workers.ocr_worker import enqueue_ocr

    det_id = f"T{tid}"
    h, w = frame.shape[:2]
    votes = 1
    methods = ["YOLO"]

    # ── Symbol / OCR check ──
    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    sym_ok = False
    sym_text = ""

    if crop.size > 0:
        # Fast red cross (inline — ~1ms)
        if detect_red_cross_strict(crop):
            cache.set(det_id, (True, "RED_CROSS"))
            sym_ok = True
            sym_text = "RED_CROSS"

    if not sym_ok:
        # Check cache for OCR result
        cached = cache.get(det_id)
        if cached is not None:
            sym_ok, sym_text = cached
        else:
            # Queue async OCR if rate limit allows
            if cache.should_ocr(det_id, fc, ocr_every_n):
                padded = prepare_ocr_crop(frame, x1, y1, x2, y2)
                if padded.size > 0:
                    enqueue_ocr(ocr_queue, det_id, padded.copy(), fc)

    if sym_ok:
        votes += 1
        methods.append("SYMBOL" if "CROSS" in sym_text else "OCR")

    # ── Color / Siren check ──
    col_ok, dom_col, has_red = color_detector.analyze(frame, x1, y1, x2, y2, det_id)
    if col_ok:
        votes += 1
        methods.append("COLOR")

    return {
        "votes": votes,
        "methods": methods,
        "sym_ok": sym_ok,
        "sym_text": sym_text,
        "col_ok": col_ok,
        "dom_col": dom_col,
        "has_red": has_red,
        "passed": votes >= VOTING_THRESHOLD,
    }
