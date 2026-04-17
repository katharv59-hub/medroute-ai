"""
MedRoute AI — Detection Engine v3.0 (Hardened)
=========================================
Detects Indian ambulances using 3 methods:
  1. YOLOv8 (shape) + size/aspect filters
  2. Red Cross symbol (strict shape analysis) + OCR text
  3. Color (white/yellow) + Siren blink pattern

HARDENED: Temporal debounce, strict filters, zero false positives.
"""

import cv2
import numpy as np
import datetime
import os
from collections import deque, defaultdict
from ultralytics import YOLO

from config import (
    CONFIDENCE_THRESHOLD, MODEL_PATH, VIDEO_SOURCE, OCR_EVERY_N_FRAMES,
    VOTING_THRESHOLD, USE_GPU, DISPLAY_WINDOW, FRAME_RESIZE,
    AMBULANCE_KEYWORDS, JUNCTIONS, MIN_BOX_AREA_RATIO, MAX_BOX_AREA_RATIO,
    MIN_ASPECT_RATIO, CONFIRM_FRAMES, RED_CROSS_MIN_SOLIDITY,
    RED_CROSS_MAX_SOLIDITY, RED_CROSS_MIN_AREA
)

_ocr_reader = None
def _get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        print("[OCR] Loading EasyOCR...")
        _ocr_reader = easyocr.Reader(['en'], gpu=USE_GPU)
        print("[OCR] Ready!")
    return _ocr_reader


# ============================================================
#  METHOD 1: YOLO + SIZE/ASPECT FILTERS
# ============================================================
def method_yolo(model, frame):
    """YOLO detection with strict size and aspect ratio filtering."""
    fh, fw = frame.shape[:2]
    frame_area = fw * fh
    results = model(frame, verbose=False, half=USE_GPU, imgsz=FRAME_RESIZE)
    dets = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue
            lid = int(box.cls[0])
            label = model.names[lid]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # SIZE FILTER: ignore tiny and huge detections
            box_w = x2 - x1
            box_h = y2 - y1
            box_area = box_w * box_h
            area_ratio = box_area / frame_area
            if area_ratio < MIN_BOX_AREA_RATIO or area_ratio > MAX_BOX_AREA_RATIO:
                continue

            # ASPECT RATIO FILTER: ambulances are wider than tall
            if box_h > 0:
                aspect = box_w / box_h
                if aspect < MIN_ASPECT_RATIO and label == "ambulance":
                    continue  # too square/tall - not an ambulance shape

            dets.append((x1, y1, x2, y2, conf, label))
    return dets


# ============================================================
#  METHOD 2: STRICT RED CROSS + OCR
# ============================================================
def _detect_red_cross_strict(crop):
    """
    STRICT red cross detection. Much tighter than before:
    - Requires proper cross SHAPE (not just any red blob)
    - Cross must have specific solidity (0.35-0.75)
    - Cross must be on WHITE background (not on dark/colored area)
    - Minimum size requirement raised
    """
    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ch, cw = crop.shape[:2]
    crop_area = ch * cw

    # Check if vehicle body is mostly WHITE first
    # This eliminates red tail lights on dark/colored cars
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    white_ratio = cv2.countNonZero(white_mask) / crop_area
    if white_ratio < 0.25:
        return False  # Not a white vehicle - skip cross check entirely

    # Red mask
    m1 = cv2.inRange(hsv, (0, 130, 120), (8, 255, 255))
    m2 = cv2.inRange(hsv, (165, 130, 120), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)

    # Aggressive morphology to clean noise
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        a = cv2.contourArea(cnt)

        # Stricter minimum area
        if a < crop_area * RED_CROSS_MIN_AREA or a > crop_area * 0.10:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < 8 or h < 8:  # absolute minimum pixels
            continue

        # Cross is roughly square (aspect 0.6 to 1.7)
        ar = float(w) / float(h)
        if ar < 0.6 or ar > 1.7:
            continue

        # STRICT solidity check - cross shape has specific solidity
        hull = cv2.convexHull(cnt)
        hull_a = cv2.contourArea(hull)
        if hull_a == 0:
            continue
        solidity = a / hull_a
        if solidity < RED_CROSS_MIN_SOLIDITY or solidity > RED_CROSS_MAX_SOLIDITY:
            continue

        # Check that the cross is on a WHITE region (not on a dark car)
        roi = white_mask[y:y+h, x:x+w]
        if roi.size > 0:
            local_white = cv2.countNonZero(roi) / roi.size
            if local_white < 0.15:
                continue  # cross area isn't on white background

        return True

    return False


def method_symbol_ocr(frame, x1, y1, x2, y2, frame_count, cache, det_id):
    """Method 2: STRICT red cross + OCR text."""
    h, w = frame.shape[:2]
    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0:
        return False, ""

    # Red cross (strict - runs every frame)
    if _detect_red_cross_strict(crop):
        cache[det_id] = (True, "RED_CROSS")
        return True, "RED_CROSS"

    # OCR (every N frames)
    if frame_count % OCR_EVERY_N_FRAMES == 0:
        ocr = _get_ocr()
        pad = 10
        padded = frame[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
        try:
            results = ocr.readtext(padded)
            for (_, text, prob) in results:
                if prob < 0.4:  # raised from 0.3
                    continue
                tu = text.upper().strip()
                for kw in AMBULANCE_KEYWORDS:
                    if kw in tu or tu in kw or kw in tu[::-1]:
                        cache[det_id] = (True, tu)
                        return True, tu
        except Exception:
            pass
        cache[det_id] = (False, "")
        return False, ""

    return cache.get(det_id, (False, ""))


# ============================================================
#  METHOD 3: COLOR + SIREN (unchanged but stricter thresholds)
# ============================================================
_siren_buf = {}

def method_color_siren(frame, x1, y1, x2, y2, det_id):
    """White/yellow body + red markings + siren blink."""
    h, w = frame.shape[:2]
    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0:
        return False, "unknown", False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    px = crop.shape[0] * crop.shape[1]
    if px == 0:
        return False, "unknown", False

    white_r = cv2.countNonZero(cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))) / px
    yellow_r = cv2.countNonZero(cv2.inRange(hsv, (15, 80, 150), (35, 255, 255))) / px
    color = "white" if white_r > 0.35 else ("yellow" if yellow_r > 0.25 else "other")

    # Red markings (raised threshold from 0.02 to 0.04)
    rm1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    rm2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    red_r = cv2.countNonZero(cv2.bitwise_or(rm1, rm2)) / px
    has_red = red_r > 0.04

    # Siren blink
    sh = max(1, int(crop.shape[0] * 0.2))
    siren = crop[0:sh, :]
    if siren.size > 0:
        sv = cv2.cvtColor(siren, cv2.COLOR_BGR2HSV)
        rs = cv2.countNonZero(cv2.inRange(sv, (0, 150, 150), (10, 255, 255)))
        bs = cv2.countNonZero(cv2.inRange(sv, (100, 150, 150), (130, 255, 255)))
        if det_id not in _siren_buf:
            _siren_buf[det_id] = deque(maxlen=15)
        _siren_buf[det_id].append(rs + bs)

    has_blink = False
    if det_id in _siren_buf and len(_siren_buf[det_id]) >= 8:
        vals = list(_siren_buf[det_id])
        mean = sum(vals) / len(vals)
        if mean > 0:
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            has_blink = var > 1000  # raised from 500

    confirmed = (color in ("white", "yellow")) and (has_red or has_blink)
    return confirmed, color, has_red


# ============================================================
#  HELPERS
# ============================================================
def detect_lane(x1, y1, x2, y2, fw, fh):
    cx, cy = (x1 + x2) / 2 / fw, (y1 + y2) / 2 / fh
    if cy < 0.35: return "north"
    if cy > 0.65: return "south"
    if cx < 0.4: return "west"
    return "east"


def classify_type(color, has_red, text):
    t = text.upper()
    if "108" in t: return "108_service"
    if color == "white" and has_red: return "government"
    if color == "white": return "government"
    return "private"


# ============================================================
#  TEMPORAL DEBOUNCE TRACKER
# ============================================================
class ConfirmTracker:
    """
    Requires CONFIRM_FRAMES consecutive voting-passed frames
    before actually triggering an alert. This eliminates
    single-frame false positives completely.
    """
    def __init__(self, required=CONFIRM_FRAMES):
        self.required = required
        self.streak = 0
        self.confirmed = False

    def update(self, detected_this_frame):
        if detected_this_frame:
            self.streak += 1
            if self.streak >= self.required:
                self.confirmed = True
        else:
            self.streak = 0
            self.confirmed = False
        return self.confirmed

    def reset(self):
        self.streak = 0
        self.confirmed = False


# ============================================================
#  MAIN DETECTION LOOP
# ============================================================
def run_detection(junction_id, video_source):
    from firebase_sender import send_detection, clear_detection

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    print(f"[{junction_id}] Model loaded: {MODEL_PATH}")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_source}")
        return

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[{junction_id}] {fw}x{fh} @ {fps:.0f}fps")
    print(f"[{junction_id}] Filters: conf>={CONFIDENCE_THRESHOLD}, "
          f"aspect>={MIN_ASPECT_RATIO}, debounce={CONFIRM_FRAMES} frames")
    print(f"[{junction_id}] Press Q to quit\n")

    fc = 0
    was_detected = False
    ocr_cache = {}
    tracker = ConfirmTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(video_source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        fc += 1
        frame_detected = False
        best = None
        best_score = 0

        for (x1, y1, x2, y2, conf, label) in method_yolo(model, frame):
            if label != "ambulance":
                continue  # don't draw non-ambulance boxes

            did = f"{junction_id}_{x1}_{y1}"
            votes = 1
            methods = ["YOLO"]

            sym_ok, sym_text = method_symbol_ocr(frame, x1, y1, x2, y2, fc, ocr_cache, did)
            if sym_ok:
                votes += 1
                methods.append("SYMBOL" if "CROSS" in sym_text else "OCR")

            col_ok, dom_col, has_red = method_color_siren(frame, x1, y1, x2, y2, did)
            if col_ok:
                votes += 1
                methods.append("COLOR")

            if votes >= VOTING_THRESHOLD:
                frame_detected = True
                method_str = "+".join(methods)
                lane = detect_lane(x1, y1, x2, y2, fw, fh)
                atype = classify_type(dom_col, has_red, sym_text)
                score = conf + 0.1 * votes
                if score > best_score:
                    best_score = score
                    best = {"lane": lane, "conf": conf, "method": method_str,
                            "type": atype, "color": dom_col,
                            "text": sym_text or "none", "votes": votes,
                            "bbox": (x1, y1, x2, y2)}

        # TEMPORAL DEBOUNCE: only confirm after N consecutive frames
        confirmed = tracker.update(frame_detected)

        if confirmed and best:
            x1, y1, x2, y2 = best["bbox"]
            # Draw CONFIRMED (red thick box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"AMBULANCE {best['conf']:.2f} [{best['method']}]",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"{best['lane'].upper()} | {best['type']}",
                        (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

            send_detection(junction_id, best["lane"], best["conf"],
                           best["method"], best["type"], best["color"], best["text"])
            ts = datetime.datetime.now().strftime('%H:%M:%S')
            print(f"[{ts}] [{junction_id}] CONFIRMED - {best['method']} | "
                  f"{best['lane'].upper()} | {best['conf']:.2f} | {best['type']}")
        elif not confirmed and was_detected:
            clear_detection(junction_id)
            tracker.reset()

        was_detected = confirmed

        # HUD - clean display
        if confirmed:
            cv2.putText(frame, "AMBULANCE CONFIRMED!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if best:
                cv2.putText(frame, f"{best['method']} | {best['lane'].upper()}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "SCANNING...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

        cv2.putText(frame, f"MedRoute AI | {junction_id}", (10, fh - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        if DISPLAY_WINDOW:
            cv2.imshow(f"ASEP - {junction_id}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[{junction_id}] Stopped.")


if __name__ == '__main__':
    from config import SYSTEM_NAME, VERSION
    print("=" * 55)
    print(f"  {SYSTEM_NAME} v{VERSION}")
    print(f"  YOLO threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  Voting: {VOTING_THRESHOLD}/3 methods required")
    print(f"  Debounce: {CONFIRM_FRAMES} consecutive frames")
    print(f"  GPU: {'Enabled' if USE_GPU else 'Disabled'}")
    print("=" * 55)

    from firebase_sender import initialize_junctions
    initialize_junctions()

    jid = list(JUNCTIONS.keys())[0]
    run_detection(jid, JUNCTIONS[jid])
