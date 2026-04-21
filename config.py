"""
MedRoute AI — Emergency Vehicle Priority System
===================================================
Central Configuration File  •  v3.0 Production

All tunable parameters in one place.
Modify PERFORMANCE_MODE or set env var PERF_MODE=LOW|BALANCED|HIGH.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============ PATHS ============
BASE_DIR             = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH           = os.path.join(BASE_DIR, "runs", "detect", "ambulance_model_gpu", "weights", "best.pt")
BASE_MODEL           = os.path.join(BASE_DIR, "models", "yolov8n.pt")
DATASET_YAML         = os.path.join(BASE_DIR, "dataset", "data.yaml")
SERVICE_ACCOUNT_KEY  = os.getenv("FIREBASE_KEY_PATH")
LOG_DIR              = os.path.join(BASE_DIR, "logs")

if not SERVICE_ACCOUNT_KEY:
    print("[WARN] FIREBASE_KEY_PATH not set in .env — Firebase disabled.")

# ============ VIDEO SOURCE ============
VIDEO_SOURCE = os.path.join(BASE_DIR, "test_ambulance.mp4")

# ============================================================
#  PERFORMANCE MODES
#  Override by setting env var: PERF_MODE=LOW  or  PERF_MODE=HIGH
# ============================================================
_MODES = {
    "LOW": {
        "INFERENCE_RESOLUTION": 416,
        "FRAME_SKIP":           3,       # process 1 in every 3 frames
        "OCR_EVERY_N_FRAMES":   15,
        "FIREBASE_MIN_INTERVAL_S": 5.0,
        "DISPLAY_WINDOW":       False,
        "FPS_LOG_INTERVAL":     25,
    },
    "BALANCED": {
        "INFERENCE_RESOLUTION": 640,
        "FRAME_SKIP":           2,       # process every 2nd frame
        "OCR_EVERY_N_FRAMES":   8,
        "FIREBASE_MIN_INTERVAL_S": 2.0,
        "DISPLAY_WINDOW":       True,
        "FPS_LOG_INTERVAL":     50,
    },
    "HIGH": {
        "INFERENCE_RESOLUTION": 1280,
        "FRAME_SKIP":           1,       # process every frame
        "OCR_EVERY_N_FRAMES":   3,
        "FIREBASE_MIN_INTERVAL_S": 0.5,
        "DISPLAY_WINDOW":       True,
        "FPS_LOG_INTERVAL":     100,
    },
}

PERFORMANCE_MODE = os.getenv("PERF_MODE", "BALANCED").upper()
if PERFORMANCE_MODE not in _MODES:
    PERFORMANCE_MODE = "BALANCED"

_cfg = _MODES[PERFORMANCE_MODE]

INFERENCE_RESOLUTION     = _cfg["INFERENCE_RESOLUTION"]   # px — YOLO input size
FRAME_SKIP               = _cfg["FRAME_SKIP"]             # process 1 in N frames
OCR_EVERY_N_FRAMES       = _cfg["OCR_EVERY_N_FRAMES"]
FIREBASE_MIN_INTERVAL_S  = _cfg["FIREBASE_MIN_INTERVAL_S"]
DISPLAY_WINDOW           = _cfg["DISPLAY_WINDOW"]
FPS_LOG_INTERVAL         = _cfg["FPS_LOG_INTERVAL"]

# Legacy alias kept for backward compat
FRAME_RESIZE = INFERENCE_RESOLUTION

# ============ DETECTION ============
CONFIDENCE_THRESHOLD = 0.88
VOTING_THRESHOLD     = 2
USE_GPU              = True         # will be overridden by runtime GPU check
TRACKER_TYPE         = "CSRT"       # "CSRT" (default, no extra deps) or "KCF"

# Anti-false-positive filters
MIN_BOX_AREA_RATIO    = 0.005
MAX_BOX_AREA_RATIO    = 0.50
MIN_ASPECT_RATIO      = 1.2
CONFIRM_FRAMES        = 5
RED_CROSS_MIN_SOLIDITY = 0.35
RED_CROSS_MAX_SOLIDITY = 0.75
RED_CROSS_MIN_AREA    = 0.005

# Indian ambulance text patterns (normal + mirror)
AMBULANCE_KEYWORDS = [
    "AMBULANCE", "ECNALUBMA",
    "108",
    "EMERGENCY", "YCNEGRME",
    "HOSPITAL",
]

# ============ FIREBASE ============
FIREBASE_URL              = "https://ev-priority-sys-001-default-rtdb.asia-southeast1.firebasedatabase.app/"
GREEN_CORRIDOR_DURATION   = 60
AUTO_RESET_ENABLED        = True
FIREBASE_RETRY_ATTEMPTS   = 3        # retries on write failure
FIREBASE_RETRY_BASE_DELAY = 0.5      # seconds (doubles each retry)
FIREBASE_CONF_CHANGE_MIN  = 0.05     # min confidence delta to trigger re-write

# ============ JUNCTIONS ============
JUNCTIONS = {
    "J1": VIDEO_SOURCE,
    # "J2": "rtsp://camera2.local/stream",
}
ALL_JUNCTION_IDS = list(JUNCTIONS.keys())

# ============ FAULT TOLERANCE ============
CAMERA_FAIL_THRESHOLD = 5    # consecutive read failures before reconnect
CAMERA_RECONNECT_TRIES = 3   # reconnect attempts before giving up
CAMERA_RECONNECT_DELAY = 2.0 # seconds between reconnect attempts

# ============ STATISTICS ============
METRICS_SAVE_INTERVAL = 100  # frames between metrics.json saves

# ============ TRAINING ============
TRAIN_EPOCHS   = 100
TRAIN_BATCH    = 16
TRAIN_PATIENCE = 20
TRAIN_WORKERS  = 4
TRAIN_AMP      = True

AUGMENTATION = {
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
    "degrees": 10.0, "translate": 0.1, "scale": 0.5,
    "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.1,
    "copy_paste": 0.1,
}

# ============ SYSTEM ============
VERSION     = "3.0.0"
SYSTEM_NAME = "MedRoute AI"
LOCATION    = "India"
