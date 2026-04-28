"""
MedRoute AI — OCR & Symbol Detector
======================================
Red-cross detection (fast, inline) and EasyOCR text matching.
Logic extracted from detect.py — detection algorithms unchanged.
"""

import cv2
import numpy as np
from config import (
    AMBULANCE_KEYWORDS, USE_GPU,
    RED_CROSS_MIN_SOLIDITY, RED_CROSS_MAX_SOLIDITY, RED_CROSS_MIN_AREA,
)
from logger import get_logger

log = get_logger("ocr_detector")

_ocr_reader = None


def get_ocr_reader():
    """Lazy-load EasyOCR reader (singleton)."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        log.info("Loading EasyOCR...")
        _ocr_reader = easyocr.Reader(['en'], gpu=USE_GPU)
        log.info("EasyOCR ready.")
    return _ocr_reader


def detect_red_cross_strict(crop):
    """Fast CV-based red cross symbol detection. Safe to call inline.

    Returns True if a red cross pattern is found in the crop.
    """
    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ch, cw = crop.shape[:2]
    crop_area = ch * cw

    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    if cv2.countNonZero(white_mask) / crop_area < 0.25:
        return False

    m1 = cv2.inRange(hsv, (0, 130, 120), (8, 255, 255))
    m2 = cv2.inRange(hsv, (165, 130, 120), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a < crop_area * RED_CROSS_MIN_AREA or a > crop_area * 0.10:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 8 or h < 8:
            continue
        ar2 = float(w) / float(h)
        if ar2 < 0.6 or ar2 > 1.7:
            continue
        hull_a = cv2.contourArea(cv2.convexHull(cnt))
        if hull_a == 0:
            continue
        sol = a / hull_a
        if sol < RED_CROSS_MIN_SOLIDITY or sol > RED_CROSS_MAX_SOLIDITY:
            continue
        roi = white_mask[y:y + h, x:x + w]
        if roi.size > 0 and cv2.countNonZero(roi) / roi.size < 0.15:
            continue
        return True
    return False


def prepare_ocr_crop(frame, x1, y1, x2, y2, pad=10):
    """Return a padded crop suitable for OCR processing."""
    h, w = frame.shape[:2]
    return frame[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]


def run_ocr_on_crop(crop):
    """Run EasyOCR on a crop and return matched keyword or None.

    Returns:
        (True, matched_text) if ambulance keyword found, else (False, "")
    """
    reader = get_ocr_reader()
    try:
        results = reader.readtext(crop)
        for (_, text, prob) in results:
            if prob < 0.4:
                continue
            tu = text.upper().strip()
            for kw in AMBULANCE_KEYWORDS:
                if kw in tu or tu in kw or kw in tu[::-1]:
                    return True, tu
    except Exception as e:
        log.error(f"OCR processing error: {e}")
    return False, ""


def method_symbol_ocr(frame, x1, y1, x2, y2, frame_count, cache, det_id, fps, ocr_every_n):
    """Combined red-cross + OCR detection with caching (synchronous fallback).

    Used when OCR worker is not available. Identical to original logic.
    """
    h, w = frame.shape[:2]
    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0:
        return False, ""

    # Fast red cross check
    if detect_red_cross_strict(crop):
        cache.set(det_id, (True, "RED_CROSS"))
        return True, "RED_CROSS"

    # Check cache
    cached = cache.get(det_id)
    if cached is not None and cached[0]:
        return cached

    # Dynamic OCR interval
    ocr_interval = ocr_every_n
    if fps > 0 and fps < 10:
        ocr_interval = ocr_every_n * 2

    if frame_count % ocr_interval == 0:
        padded = prepare_ocr_crop(frame, x1, y1, x2, y2)
        found, text = run_ocr_on_crop(padded)
        cache.set(det_id, (found, text))
        return found, text

    result = cache.get(det_id)
    return result if result is not None else (False, "")
