"""
MedRoute AI — YOLO Detector
==============================
Wraps YOLO inference with size/aspect-ratio filtering.
Logic is identical to the original method_yolo in detect.py.
"""

import time
from config import (
    CONFIDENCE_THRESHOLD, INFERENCE_RESOLUTION,
    MIN_BOX_AREA_RATIO, MAX_BOX_AREA_RATIO, MIN_ASPECT_RATIO,
)
from logger import get_logger

log = get_logger("yolo_detector")


def detect(model, frame, gpu, latency_tracker=None):
    """Run YOLO inference and return filtered detections.

    Returns:
        list of (x1, y1, x2, y2, conf, label)
    """
    fh, fw = frame.shape[:2]
    frame_area = fw * fh

    t0 = time.perf_counter()
    results = model(frame, verbose=False, half=gpu, imgsz=INFERENCE_RESOLUTION)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if latency_tracker is not None:
        latency_tracker.record("yolo_ms", elapsed_ms)

    dets = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue
            lid = int(box.cls[0])
            label = model.names[lid]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bw, bh = x2 - x1, y2 - y1
            ar = bw * bh / frame_area
            if ar < MIN_BOX_AREA_RATIO or ar > MAX_BOX_AREA_RATIO:
                continue
            if bh > 0 and bw / bh < MIN_ASPECT_RATIO and label == "ambulance":
                continue
            dets.append((x1, y1, x2, y2, conf, label))
    return dets
