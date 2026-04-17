"""
MedRoute AI — Emergency Vehicle Priority System
===================================================
Central Configuration File

All tunable parameters in one place.
Modify this file to adjust system behavior.
"""

import os

# ============ PATHS ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "ambulance_model_gpu", "weights", "best.pt")
BASE_MODEL = os.path.join(BASE_DIR, "models", "yolov8n.pt")
DATASET_YAML = os.path.join(BASE_DIR, "dataset", "data.yaml")
SERVICE_ACCOUNT_KEY = os.path.join(BASE_DIR, "serviceAccountKey.json")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# ============ VIDEO SOURCE ============
# Use 0 for webcam, or path to video file for testing
VIDEO_SOURCE = os.path.join(BASE_DIR, "test_ambulance.mp4")

# ============ DETECTION ============
CONFIDENCE_THRESHOLD = 0.88      # STRICT: Minimum YOLO confidence
VOTING_THRESHOLD = 2             # Minimum methods (out of 3) to confirm ambulance
OCR_EVERY_N_FRAMES = 5           # Run OCR every N frames (OCR is slow)
FRAME_RESIZE = 640               # Resize frames for YOLO inference
USE_GPU = True                   # Enable CUDA GPU acceleration
DISPLAY_WINDOW = True            # Show OpenCV detection window

# Anti-false-positive filters
MIN_BOX_AREA_RATIO = 0.005      # Minimum detection area as % of frame (ignore tiny boxes)
MAX_BOX_AREA_RATIO = 0.50       # Maximum detection area as % of frame (ignore huge boxes)
MIN_ASPECT_RATIO = 1.2          # Ambulances are wider than tall (w/h >= 1.2)
CONFIRM_FRAMES = 5              # Must detect in N consecutive frames before confirming
RED_CROSS_MIN_SOLIDITY = 0.35   # Stricter cross shape solidity minimum
RED_CROSS_MAX_SOLIDITY = 0.75   # Stricter cross shape solidity maximum
RED_CROSS_MIN_AREA = 0.005      # Minimum cross area as % of vehicle (bigger than before)

# Indian ambulance text patterns (normal + mirror)
AMBULANCE_KEYWORDS = [
    "AMBULANCE", "ECNALUBMA",    # English + mirror
    "108",                        # 108 emergency service number
    "EMERGENCY", "YCNEGRME",     # Emergency + mirror
    "HOSPITAL",                   # Hospital ambulance branding
]

# ============ FIREBASE ============
FIREBASE_URL = "https://ev-priority-sys-001-default-rtdb.asia-southeast1.firebasedatabase.app/"
GREEN_CORRIDOR_DURATION = 60     # Seconds to hold green for ambulance
AUTO_RESET_ENABLED = True        # Auto-reset signals after corridor duration

# ============ JUNCTIONS ============
# Map junction IDs to their camera source
# Add more junctions here for multi-junction support
JUNCTIONS = {
    "J1": VIDEO_SOURCE,
    # "J2": "rtsp://camera2.local/stream",
    # "J3": 0,  # second webcam
}
ALL_JUNCTION_IDS = list(JUNCTIONS.keys())

# ============ TRAINING ============
TRAIN_EPOCHS = 100
TRAIN_BATCH = 16                 # Fits in RTX 4050 6GB VRAM
TRAIN_PATIENCE = 20              # Early stopping patience
TRAIN_WORKERS = 4
TRAIN_AMP = True                 # Mixed precision for speed

# Data augmentation for Indian road conditions
AUGMENTATION = {
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
    "degrees": 10.0, "translate": 0.1, "scale": 0.5,
    "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.1,
    "copy_paste": 0.1,
}

# ============ SYSTEM ============
VERSION = "2.0.0"
SYSTEM_NAME = "MedRoute AI"
LOCATION = "India"
