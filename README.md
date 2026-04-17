# MedRoute AI 🚑

> AI-powered dynamic emergency vehicle priority and green corridor system using real-time cloud computing — built for Indian roads.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)](https://ultralytics.com)
[![Firebase](https://img.shields.io/badge/Firebase-RTDB-orange?logo=firebase)](https://firebase.google.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green?logo=opencv)](https://opencv.org)

## Overview

MedRoute AI detects ambulances in real-time traffic camera feeds and dynamically manages traffic signals to create **green corridors** for emergency vehicles. The system uses a **3-method voting architecture** to ensure near-zero false positives while maintaining fast response times.

### Problem Statement
In India, ambulances lose an average of **15-20 minutes** per trip due to traffic congestion, especially at signalized junctions. This delay directly impacts patient survival rates, particularly for cardiac and trauma emergencies.

### Solution
MedRoute AI provides an automated, AI-driven traffic signal management system that:
- Detects ambulances using **multi-method computer vision** (not just one model)
- Dynamically switches traffic signals to create a green corridor
- Auto-resets signals after the ambulance passes
- Provides a real-time cloud dashboard for traffic control rooms

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐     ┌─────────────┐
│  CCTV/Camera │────▶│   detect.py      │────▶│   Firebase   │────▶│  Dashboard  │
│   Feed       │     │  3-Method Voting │     │   RTDB       │     │  (Browser)  │
└─────────────┘     └──────────────────┘     └──────────────┘     └─────────────┘
                     │                  │
                     ▼                  ▼
              ┌─────────────┐   ┌──────────────┐
              │ YOLOv8 Model│   │ Signal Logic  │
              │ (GPU/CUDA)  │   │ (Auto-Reset)  │
              └─────────────┘   └──────────────┘
```

## Detection Pipeline — 3-Method Voting

MedRoute AI does **not** rely on a single detection method. Instead, it runs **3 independent methods** and only triggers an alert when **≥2 agree** (2-of-3 voting):

| # | Method | What it Detects | How |
|---|--------|----------------|-----|
| 1 | **YOLOv8 Deep Learning** | Ambulance shape/body | Custom-trained model on Indian ambulance dataset |
| 2 | **Red Cross ✚ Symbol + OCR** | Plus symbol + mirror text | HSV color masking + contour analysis + EasyOCR |
| 3 | **Color + Siren Analysis** | White body + red markings + blink | HSV ratios + frame-differencing for siren blink |

### 5-Layer False Positive Prevention
```
YOLO (conf ≥ 0.88) → Aspect Ratio Filter → 2/3 Voting → Strict Symbol Check → 5-Frame Temporal Debounce
```

## Key Features

- **Multi-method voting** — Near-zero false positives (0 on test footage)
- **India-specific** — Trained on Indian ambulances (108, govt, private)
- **Real-time dashboard** — Glassmorphism UI with live junction map
- **Multi-junction support** — Manage J1, J2, J3 simultaneously
- **Auto-reset signals** — Green corridor reverts after 60 seconds
- **GPU-optimized** — RTX 4050 + CUDA 12.1 for real-time inference
- **Red Cross ✚ detection** — Detects the plus symbol, not just text
- **Mirror text OCR** — Reads "ECNALUBMA" on ambulance fronts

## Model Performance

| Metric | Score |
|--------|-------|
| mAP50 | **92.87%** |
| mAP50-95 | 64.6% |
| Precision | 95.4% |
| Recall | 92.9% |
| False Positives (with voting) | **0** |

## Quick Start

### Prerequisites
- Python 3.11+
- NVIDIA GPU with CUDA 12.1 (recommended)
- Firebase project with Realtime Database

### Installation

```bash
git clone https://github.com/katharv59-hub/medroute-ai.git
cd medroute-ai
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### Configuration
Edit `config.py` to set:
- `VIDEO_SOURCE` — camera feed or video file path
- `MODEL_PATH` — path to trained YOLOv8 weights
- Firebase credentials in `serviceAccountKey.json`

### Run Detection
```bash
python detect.py
```

### Run Dashboard
Open `dashboard/index.html` in a browser — it connects to Firebase automatically.

### Train Model
```bash
python train.py
```

## Project Structure

```
medroute-ai/
├── config.py              # Centralized configuration
├── detect.py              # Main detection engine (3 methods + voting)
├── firebase_sender.py     # Firebase RTDB communication + auto-reset
├── train.py               # YOLOv8 training script (GPU-optimized)
├── fix_labels.py          # Dataset label correction utility
├── requirements.txt       # Python dependencies
├── run.bat                # One-click Windows launcher
├── dashboard/
│   └── index.html         # Real-time monitoring dashboard
├── models/                # Base YOLO weights
├── runs/detect/           # Trained model weights
├── dataset/               # Training images + labels
└── logs/                  # Runtime logs
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Detection | YOLOv8 (Ultralytics), EasyOCR, OpenCV |
| Cloud | Firebase Realtime Database |
| Dashboard | HTML5, CSS3, JavaScript (Firebase SDK) |
| GPU | NVIDIA RTX 4050, CUDA 12.1 |
| Runtime | Python 3.11, Windows 11 |

## Dashboard Preview

The dashboard features a premium glassmorphism design with:
- Real-time junction map with color-coded signals
- Multi-junction tabs (J1, J2, J3)
- Detection method badges (YOLO, OCR, SYMBOL, COLOR)
- Alert history with confidence metrics
- System uptime and statistics

## How It Works

1. **Camera Feed** → Video frames are captured from CCTV/webcam
2. **YOLO Detection** → Custom model identifies ambulance-shaped vehicles
3. **Symbol + OCR** → Red cross symbol and "AMBULANCE"/"108" text are verified
4. **Color + Siren** → White/yellow body and siren blink pattern are checked
5. **2/3 Voting** → Alert triggers only when ≥2 methods confirm
6. **Temporal Debounce** → Must persist for 5 consecutive frames
7. **Firebase Update** → Signal change sent to cloud in real-time
8. **Dashboard** → Traffic control room sees live signal status
9. **Auto-Reset** → Signals revert to normal after 60 seconds

## License

This project is developed for academic purposes — AI-Based Dynamic Emergency Vehicle Priority System for Indian Traffic Management.

## Author

**katharv59**
