"""
MedRoute AI — Performance Benchmark Tool
==========================================
Standalone profiler. Tests YOLO, OCR, Firebase latency, and end-to-end FPS.
Usage:  python benchmark.py
Output: Terminal table + logs/benchmark_report.txt
"""
import time, os, sys, statistics, json
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_PATH, VIDEO_SOURCE, INFERENCE_RESOLUTION, USE_GPU,
    LOG_DIR, SERVICE_ACCOUNT_KEY
)
from logger import get_logger

log = get_logger("benchmark")
os.makedirs(LOG_DIR, exist_ok=True)
REPORT_PATH = os.path.join(LOG_DIR, "benchmark_report.txt")


def _header(title):
    log.info(f"\n{'─'*50}")
    log.info(f"  {title}")
    log.info(f"{'─'*50}")


def _stats(times_ms):
    if not times_ms:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
    return {
        "mean": round(statistics.mean(times_ms), 2),
        "std": round(statistics.stdev(times_ms) if len(times_ms) > 1 else 0, 2),
        "min": round(min(times_ms), 2),
        "max": round(max(times_ms), 2),
        "median": round(statistics.median(times_ms), 2),
    }


def bench_gpu():
    _header("GPU CHECK")
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            log.info(f"GPU: {name} | VRAM: {vram:.1f} GB | CUDA: {torch.version.cuda}")
            return {"gpu": name, "vram_gb": round(vram, 1), "cuda": torch.version.cuda}
        else:
            log.info("No CUDA GPU available — CPU mode.")
            return {"gpu": "CPU", "vram_gb": 0, "cuda": "N/A"}
    except ImportError:
        log.info("PyTorch not available.")
        return {"gpu": "unknown", "vram_gb": 0, "cuda": "N/A"}


def bench_yolo(n_frames=100):
    _header(f"YOLO INFERENCE ({n_frames} frames)")
    from ultralytics import YOLO
    if not os.path.exists(MODEL_PATH):
        log.error(f"Model not found: {MODEL_PATH}")
        return None

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        log.error(f"Cannot open video: {VIDEO_SOURCE}")
        return None

    gpu = USE_GPU
    times = []
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        t0 = time.perf_counter()
        model(frame, verbose=False, half=gpu, imgsz=INFERENCE_RESOLUTION)
        times.append((time.perf_counter() - t0) * 1000)

    cap.release()
    s = _stats(times)
    fps = 1000 / s["mean"] if s["mean"] > 0 else 0
    log.info(f"Mean: {s['mean']:.1f}ms | Max: {s['max']:.1f}ms | FPS: {fps:.1f}")
    s["fps"] = round(fps, 1)
    return s


def bench_ocr(n_crops=20):
    _header(f"OCR TIMING ({n_crops} crops)")
    try:
        import easyocr
    except ImportError:
        log.error("EasyOCR not installed.")
        return None

    reader = easyocr.Reader(['en'], gpu=USE_GPU)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        return None

    times = []
    for _ in range(n_crops):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        h, w = frame.shape[:2]
        crop = frame[h//4:3*h//4, w//4:3*w//4]
        t0 = time.perf_counter()
        reader.readtext(crop)
        times.append((time.perf_counter() - t0) * 1000)

    cap.release()
    s = _stats(times)
    log.info(f"Mean: {s['mean']:.1f}ms | Max: {s['max']:.1f}ms")
    return s


def bench_firebase(n_writes=10):
    _header(f"FIREBASE LATENCY ({n_writes} writes)")
    if not SERVICE_ACCOUNT_KEY or not os.path.exists(SERVICE_ACCOUNT_KEY):
        log.info("Firebase key not configured — skipping.")
        return None

    try:
        from firebase_sender import send_detection, clear_detection
    except Exception as e:
        log.warning(f"Firebase import failed: {e}")
        return None

    times = []
    for i in range(n_writes):
        t0 = time.perf_counter()
        try:
            send_detection("BENCH", "north", 0.95, "BENCHMARK", "test", "white", "TEST")
        except Exception:
            pass
        times.append((time.perf_counter() - t0) * 1000)

    try:
        clear_detection("BENCH")
    except Exception:
        pass

    s = _stats(times)
    log.info(f"Mean: {s['mean']:.1f}ms | Max: {s['max']:.1f}ms")
    return s


def bench_e2e(n_frames=200):
    _header(f"END-TO-END PIPELINE ({n_frames} frames)")
    from ultralytics import YOLO
    if not os.path.exists(MODEL_PATH):
        return None

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        return None

    gpu = USE_GPU
    times = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        t0 = time.perf_counter()
        results = model(frame, verbose=False, half=gpu, imgsz=INFERENCE_RESOLUTION)
        # simulate minimal post-processing
        for r in results:
            for box in r.boxes:
                _ = float(box.conf[0])
        times.append((time.perf_counter() - t0) * 1000)

    cap.release()
    s = _stats(times)
    fps = 1000 / s["mean"] if s["mean"] > 0 else 0
    log.info(f"Pipeline FPS: {fps:.1f} | Mean: {s['mean']:.1f}ms | Max: {s['max']:.1f}ms")
    s["fps"] = round(fps, 1)
    return s


def _find_bottleneck(results):
    slowest = None
    worst = 0
    for name, data in results.items():
        if data and data.get("mean", 0) > worst:
            worst = data["mean"]
            slowest = name
    return slowest


def main():
    log.info("=" * 55)
    log.info("  MedRoute AI — Performance Benchmark")
    log.info("=" * 55)

    report = {}
    report["gpu"] = bench_gpu()
    report["yolo"] = bench_yolo()
    report["ocr"] = bench_ocr()
    report["firebase"] = bench_firebase()
    report["e2e"] = bench_e2e()

    bottleneck = _find_bottleneck(report)

    log.info(f"\n{'='*55}")
    log.info("  RESULTS SUMMARY")
    log.info(f"{'='*55}")
    for name, data in report.items():
        if data:
            log.info(f"  {name:12s}: {json.dumps(data)}")
    log.info(f"  Bottleneck : {bottleneck or 'N/A'}")
    log.info(f"{'='*55}")

    # Save report
    with open(REPORT_PATH, "w") as f:
        f.write("MedRoute AI — Benchmark Report\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*55}\n")
        for name, data in report.items():
            f.write(f"{name}: {json.dumps(data, indent=2)}\n")
        f.write(f"\nBottleneck: {bottleneck or 'N/A'}\n")
    log.info(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
