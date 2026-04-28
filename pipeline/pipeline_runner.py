"""
MedRoute AI — Pipeline Runner
================================
Main orchestrator: starts all threads, runs the detection loop on the main
thread, coordinates shutdown.  Replaces the monolithic run_detection().

Thread Model:
  Thread A  — FrameCapture   (reads video)
  Main      — Detection loop (YOLO + tracking + voting + display)
  Thread C  — OCRWorker      (async OCR)
  Thread D  — FirebaseWorker (async Firebase writes)
"""

import cv2
import time
import signal
import os
from queue import Queue

from ultralytics import YOLO
from logger import get_logger
from config import (
    MODEL_PATH, USE_GPU, DISPLAY_WINDOW, FRAME_SKIP,
    INFERENCE_RESOLUTION, PERFORMANCE_MODE, FPS_LOG_INTERVAL,
    METRICS_SAVE_INTERVAL, OCR_EVERY_N_FRAMES, AMBULANCE_KEYWORDS,
    FRAME_QUEUE_SIZE, FIREBASE_QUEUE_SIZE, OCR_QUEUE_SIZE,
    CACHE_EXPIRY_SECONDS, SORT_MAX_AGE, SORT_MIN_HITS, SORT_IOU_THRESHOLD,
    FIREBASE_RETRY_ATTEMPTS, FIREBASE_RETRY_BASE_DELAY,
)
from csv_logger import log_detection

# Modular imports
from utils.performance import PerfMonitor, StatsTracker, LatencyTracker
from utils.cache_manager import CacheManager
from utils.frame_utils import detect_lane, classify_type, lane_priority
from detectors import yolo_detector
from detectors.color_detector import ColorSirenDetector
from tracking.sort_tracker import create_tracker
from pipeline.capture import FrameCapture
from pipeline.dispatcher import ConfirmTracker, evaluate_detection
from workers.firebase_worker import FirebaseWorker, enqueue_send, enqueue_clear
from workers.ocr_worker import OCRWorker

log = get_logger("pipeline")


# ── GPU VALIDATION ──────────────────────────────────────────
def validate_gpu():
    """Check GPU availability and return whether to use it."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            log.info(f"GPU: {name} | VRAM: {vram:.1f} GB | CUDA: {torch.version.cuda}")
            return True
        else:
            log.warning("CUDA not available — falling back to CPU.")
            return False
    except ImportError:
        log.warning("PyTorch not found — falling back to CPU.")
        return False


class PipelineRunner:
    """Orchestrates the entire multi-threaded detection pipeline."""

    def __init__(self, junction_id, video_source):
        self.junction_id = junction_id
        self.video_source = video_source
        self._shutdown_flag = False

    def run(self):
        """Entry point — sets up threads and runs detection on main thread."""
        jid = self.junction_id

        # ── Validate model ──
        if not os.path.exists(MODEL_PATH):
            log.error(f"Model not found: {MODEL_PATH}")
            return

        gpu = validate_gpu() and USE_GPU
        model = YOLO(MODEL_PATH)
        log.info(f"[{jid}] Model loaded | Mode: {PERFORMANCE_MODE} | GPU: {gpu}")

        # ── Shared queues ──
        frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
        firebase_queue = Queue(maxsize=FIREBASE_QUEUE_SIZE)
        ocr_queue = Queue(maxsize=OCR_QUEUE_SIZE)

        # ── Shared state ──
        ocr_cache = CacheManager(expiry_seconds=CACHE_EXPIRY_SECONDS)
        latency = LatencyTracker(save_interval=METRICS_SAVE_INTERVAL)
        color_detector = ColorSirenDetector()

        # ── Thread A: Frame Capture ──
        capture = FrameCapture(self.video_source, frame_queue,
                               latency_tracker=latency)
        capture.start()
        if not capture.wait_ready(timeout=10):
            log.error("Camera failed to open within timeout.")
            capture.stop()
            return

        fw, fh = capture.fw, capture.fh
        src_fps = capture.src_fps
        log.info(f"[{jid}] {fw}x{fh} @ {src_fps:.0f}fps | "
                 f"Skip: 1/{FRAME_SKIP} | Res: {INFERENCE_RESOLUTION}")

        # ── Thread C: OCR Worker ──
        ocr_worker = OCRWorker(ocr_queue, ocr_cache, gpu=gpu,
                               keywords=AMBULANCE_KEYWORDS,
                               latency_tracker=latency)
        ocr_worker.start()

        # ── Thread D: Firebase Worker ──
        fb_worker = FirebaseWorker(firebase_queue,
                                   max_retries=FIREBASE_RETRY_ATTEMPTS,
                                   base_delay=FIREBASE_RETRY_BASE_DELAY,
                                   latency_tracker=latency)
        fb_worker.start()

        # ── Detection state ──
        tracker = create_tracker(max_age=SORT_MAX_AGE,
                                 min_hits=SORT_MIN_HITS,
                                 iou_threshold=SORT_IOU_THRESHOLD)
        confirm = ConfirmTracker()
        perf = PerfMonitor()
        stats = StatsTracker()
        last_processed_frame = None
        prev_confirmed_ids = set()
        cache_cleanup_interval = 500  # frames between cache cleanups

        # ── Signal handling ──
        def _sighandler(sig, frame):
            self._shutdown_flag = True
            log.info("Shutdown signal received.")
        signal.signal(signal.SIGINT, _sighandler)

        # ══════════════════════════════════════════════════════
        #  MAIN DETECTION LOOP (runs on main thread)
        # ══════════════════════════════════════════════════════
        log.info(f"[{jid}] Detection pipeline running...")

        while not self._shutdown_flag:
            perf.tick_start()

            # ── Get frame from capture thread ──
            item = frame_queue.get()
            if item is None:
                log.info("Capture ended — shutting down pipeline.")
                break

            fc, frame = item
            stats.total_frames = fc

            # ── Frame skip ──
            if FRAME_SKIP > 1 and fc % FRAME_SKIP != 0:
                if last_processed_frame is not None and DISPLAY_WINDOW:
                    cv2.imshow(f"ASEP - {jid}", last_processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                perf.tick_end()
                continue

            # ── YOLO detection ──
            raw_dets = yolo_detector.detect(model, frame, gpu,
                                            latency_tracker=latency)
            ambulance_dets = [d for d in raw_dets if d[5] == "ambulance"]

            # ── Track across frames ──
            tracked = tracker.update(ambulance_dets)

            # ── Evaluate each tracked detection ──
            confirmed_this_frame = []
            for (tid, x1, y1, x2, y2, conf, label) in tracked:
                result = evaluate_detection(
                    tid, x1, y1, x2, y2, conf, label,
                    frame, fc, ocr_cache, color_detector,
                    ocr_queue, perf.current_fps, OCR_EVERY_N_FRAMES,
                )

                is_confirmed = confirm.update(tid, result["passed"])

                if is_confirmed:
                    lane = detect_lane(x1, y1, x2, y2, fw, fh)
                    atype = classify_type(result["dom_col"], result["has_red"],
                                          result["sym_text"])
                    method_str = "+".join(result["methods"])
                    priority = conf + 0.1 * result["votes"] + 0.05 * lane_priority(lane)
                    confirmed_this_frame.append({
                        "tid": tid, "lane": lane, "conf": conf,
                        "method": method_str, "type": atype,
                        "color": result["dom_col"],
                        "text": result["sym_text"] or "none",
                        "votes": result["votes"],
                        "bbox": (x1, y1, x2, y2),
                        "priority": priority,
                    })

            # Sort by priority (highest first)
            confirmed_this_frame.sort(key=lambda d: d["priority"], reverse=True)
            current_confirmed_ids = {d["tid"] for d in confirmed_this_frame}

            # ── Draw + log + queue Firebase for all confirmed ──
            for i, det in enumerate(confirmed_this_frame):
                x1, y1, x2, y2 = det["bbox"]
                stats.record_detection(det["conf"])

                # Draw bounding box
                color_bgr = (0, 0, 255) if i == 0 else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 3)
                cv2.putText(frame,
                            f"AMB T{det['tid']} {det['conf']:.2f} [{det['method']}]",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            color_bgr, 2)
                cv2.putText(frame,
                            f"{det['lane'].upper()} | {det['type']} | P{i+1}",
                            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 200, 255), 1)

                # CSV always (synchronous — fast)
                log_detection(jid, det["lane"], det["conf"],
                              det["method"], det["type"],
                              det["color"], det["text"])

                # Firebase: send highest priority only (async via queue)
                if i == 0:
                    enqueue_send(firebase_queue, jid, det["lane"], det["conf"],
                                 det["method"], det["type"],
                                 det["color"], det["text"])

            # ── Clear for tracks that lost confirmation ──
            lost_ids = prev_confirmed_ids - current_confirmed_ids
            if lost_ids and not current_confirmed_ids:
                enqueue_clear(firebase_queue, jid)
            prev_confirmed_ids = current_confirmed_ids

            # ── HUD ──
            n = len(confirmed_this_frame)
            if n > 0:
                cv2.putText(frame, f"AMBULANCE x{n} CONFIRMED!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "SCANNING...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

            # Mode badge
            cv2.putText(frame,
                        f"MedRoute AI | {jid} | {PERFORMANCE_MODE}",
                        (10, fh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (100, 100, 100), 1)

            perf.tick_end()
            perf.overlay(frame)
            last_processed_frame = frame.copy()

            # ── Periodic logging ──
            if fc % FPS_LOG_INTERVAL == 0:
                log.info(f"[{jid}] Frame {fc} | FPS: {perf.current_fps:.1f} "
                         f"(avg {perf.avg_fps:.1f}) | {perf.frame_ms:.0f}ms | "
                         f"Detections: {stats.total_detections}")
            if fc % METRICS_SAVE_INTERVAL == 0:
                stats.save()

            # ── Periodic cache cleanup ──
            if fc % cache_cleanup_interval == 0:
                ocr_cache.cleanup()

            if DISPLAY_WINDOW:
                cv2.imshow(f"ASEP - {jid}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # ══════════════════════════════════════════════════════
        #  SHUTDOWN
        # ══════════════════════════════════════════════════════
        log.info(f"[{jid}] Shutting down pipeline...")

        # Stop threads
        capture.stop()
        ocr_worker.stop()
        fb_worker.stop()

        # Wait for threads to finish
        capture.join(timeout=3)
        ocr_worker.join(timeout=3)
        fb_worker.join(timeout=5)

        # Final saves
        stats.save()
        latency.save()
        cv2.destroyAllWindows()

        log.info(f"[{jid}] Session complete | Frames: {fc} | "
                 f"Detections: {stats.total_detections}")
        log.info(f"[{jid}] OCR Worker: {ocr_worker.stats} | "
                 f"Firebase Worker: {fb_worker.stats}")
