# MedRoute AI — Emergency Vehicle Priority System v3.0

**Production-Grade Intelligent Traffic Signal Control**

AI-powered system that detects Indian ambulances using YOLOv8 + OCR + Color analysis with a 3-method voting architecture, then triggers green corridor signals via Firebase Realtime Database.

---

## 🚀 What's New in v3.0

| Feature | Description |
|---|---|
| **FPS Monitoring** | Real-time FPS overlay on video + terminal logging every 50 frames |
| **Performance Modes** | `LOW` / `BALANCED` / `HIGH` — controls resolution, frame skip, OCR frequency |
| **Frame Skipping** | Process every Nth frame while displaying smoothly (configurable) |
| **Object Tracking** | IoU-based multi-object tracker with stable IDs across frames |
| **Multi-Ambulance** | Handles multiple simultaneous detections with priority scoring |
| **Smart OCR** | Cached per track ID, dynamic interval based on FPS |
| **Firebase Throttling** | State-diff gating, retry with exponential backoff, auto-reconnect |
| **Fault Tolerance** | Camera watchdog, auto-reconnect, graceful shutdown (Ctrl+C) |
| **Logging Framework** | Python `logging` module with rotating file + colored console output |
| **GPU Validation** | Runtime CUDA/VRAM detection with automatic CPU fallback |
| **Statistics Dashboard** | `logs/system_metrics.json` — uptime, detections, drop rate |
| **Benchmark Tool** | `benchmark.py` — YOLO/OCR/Firebase/E2E profiler |

---

## 📁 Project Structure

```
ASEP/
├── detect.py              # Main detection engine (v3.0)
├── config.py              # Central config + performance modes
├── logger.py              # Logging framework
├── firebase_sender.py     # Firebase with throttle + retry
├── csv_logger.py          # CSV fallback logger
├── benchmark.py           # Performance profiler
├── run.bat                # One-click launcher
├── train.py               # Model training script
├── dashboard/
│   └── index.html         # Real-time web dashboard
├── logs/
│   ├── system.log         # Application logs (auto-created)
│   ├── system_metrics.json # Runtime statistics (auto-created)
│   ├── detections.csv     # Detection log (auto-created)
│   └── benchmark_report.txt # Benchmark results (auto-created)
├── runs/detect/           # Trained YOLO model weights
├── .env                   # Environment variables
└── requirements.txt       # Python dependencies
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env:
#   FIREBASE_KEY_PATH=serviceAccountKey.json
#   PERF_MODE=BALANCED    (LOW | BALANCED | HIGH)
```

### 3. Run
```bash
python detect.py
# or double-click run.bat
```

### 4. Benchmark (optional)
```bash
python benchmark.py
```

---

## 🎛️ Performance Modes

Set `PERF_MODE` in `.env` or as environment variable:

| Mode | Resolution | Frame Skip | OCR Interval | Firebase Throttle | Display |
|---|---|---|---|---|---|
| `LOW` | 416px | 1 in 3 | 15 frames | 5s | Off |
| `BALANCED` | 640px | 1 in 2 | 8 frames | 2s | On |
| `HIGH` | 1280px | Every frame | 3 frames | 0.5s | On |

---

## 🔍 Detection Pipeline

```
Frame → YOLO Detection → Object Tracker (IoU) → Per-Track Voting:
   ├─ Method 1: YOLO confidence + size/aspect filters
   ├─ Method 2: Red cross symbol + OCR text (cached per track)
   └─ Method 3: Color (white/yellow) + siren blink pattern

Votes ≥ 2/3 → Temporal Debounce (5 frames) → CONFIRMED
   ├─ Firebase alert (throttled, with retry)
   ├─ CSV log (always)
   └─ Dashboard update
```

---

## 🛡️ Fault Tolerance

- **Camera disconnect**: Auto-reconnect up to 3 attempts with 2s delay
- **Firebase failure**: Exponential backoff retry (3 attempts), auto-reconnect on next call
- **CSV fallback**: Always logs to CSV regardless of Firebase status
- **Graceful shutdown**: Ctrl+C saves stats and releases resources cleanly

---

## 📊 Monitoring

**Real-time on video window:**
- Current FPS, Average FPS, Frame processing time (ms)

**Terminal (every 50 frames):**
```
[detect] Frame 250 | FPS: 24.3 (avg 22.1) | 41ms | Detections: 3
```

**Log file:** `logs/system.log` (rotating, 5MB × 3 backups)

**Metrics JSON:** `logs/system_metrics.json`
```json
{
  "uptime_seconds": 342.1,
  "total_frames": 8500,
  "frames_dropped": 12,
  "frame_drop_rate_pct": 0.14,
  "total_detections": 47,
  "average_confidence": 0.923
}
```

---

## 🔧 Tech Stack

- **Detection**: YOLOv8 (Ultralytics)
- **OCR**: EasyOCR
- **Tracking**: IoU-based multi-object tracker
- **Backend**: Firebase Realtime Database
- **Video**: OpenCV
- **GPU**: CUDA (RTX-class, auto-fallback to CPU)
- **Logging**: Python `logging` with rotating file handler

---

## 👤 Author

**katharv59** — MedRoute AI Project

---

## 📜 License

For academic and research purposes.
