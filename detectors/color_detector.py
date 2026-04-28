"""
MedRoute AI — Color & Siren Detector
=======================================
Analyzes vehicle color (white/yellow), red markings, and siren blink patterns.
Logic extracted from detect.py — detection algorithm unchanged.
"""

import cv2
import numpy as np
from collections import deque
from logger import get_logger

log = get_logger("color_detector")


class ColorSirenDetector:
    """Stateful detector that tracks siren blink patterns per vehicle."""

    def __init__(self):
        self._siren_buf = {}   # det_id -> deque of siren pixel counts

    def analyze(self, frame, x1, y1, x2, y2, det_id):
        """Analyze vehicle for ambulance color/siren characteristics.

        Returns:
            (is_ambulance_like, dominant_color, has_red)
        """
        h, w = frame.shape[:2]
        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if crop.size == 0:
            return False, "unknown", False

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        px = crop.shape[0] * crop.shape[1]
        if px == 0:
            return False, "unknown", False

        # Dominant color analysis
        white_r = cv2.countNonZero(cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))) / px
        yellow_r = cv2.countNonZero(cv2.inRange(hsv, (15, 80, 150), (35, 255, 255))) / px
        color = "white" if white_r > 0.35 else ("yellow" if yellow_r > 0.25 else "other")

        # Red marking detection
        rm1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        rm2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
        has_red = cv2.countNonZero(cv2.bitwise_or(rm1, rm2)) / px > 0.04

        # Siren blink analysis (top 20% of crop)
        sh = max(1, int(crop.shape[0] * 0.2))
        siren = crop[0:sh, :]
        if siren.size > 0:
            sv = cv2.cvtColor(siren, cv2.COLOR_BGR2HSV)
            rs = cv2.countNonZero(cv2.inRange(sv, (0, 150, 150), (10, 255, 255)))
            bs = cv2.countNonZero(cv2.inRange(sv, (100, 150, 150), (130, 255, 255)))
            if det_id not in self._siren_buf:
                self._siren_buf[det_id] = deque(maxlen=15)
            self._siren_buf[det_id].append(rs + bs)

        has_blink = False
        if det_id in self._siren_buf and len(self._siren_buf[det_id]) >= 8:
            vals = list(self._siren_buf[det_id])
            mean = sum(vals) / len(vals)
            if mean > 0:
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                has_blink = var > 1000

        return (color in ("white", "yellow")) and (has_red or has_blink), color, has_red

    def cleanup_id(self, det_id):
        """Remove siren buffer for a lost track."""
        self._siren_buf.pop(det_id, None)
