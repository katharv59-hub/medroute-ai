"""
MedRoute AI — Detection Engine v3.0 (Production)
=================================================
Features: FPS monitor, frame skip, object tracking, multi-ambulance,
smart OCR, fault tolerance, GPU validation, stats dashboard.
"""
import cv2, numpy as np, datetime, os, sys, time, json, signal
from collections import deque, defaultdict
from ultralytics import YOLO
from logger import get_logger
from config import (
    CONFIDENCE_THRESHOLD, MODEL_PATH, VIDEO_SOURCE, OCR_EVERY_N_FRAMES,
    VOTING_THRESHOLD, USE_GPU, DISPLAY_WINDOW, FRAME_RESIZE,
    AMBULANCE_KEYWORDS, JUNCTIONS, MIN_BOX_AREA_RATIO, MAX_BOX_AREA_RATIO,
    MIN_ASPECT_RATIO, CONFIRM_FRAMES, RED_CROSS_MIN_SOLIDITY,
    RED_CROSS_MAX_SOLIDITY, RED_CROSS_MIN_AREA, FRAME_SKIP,
    INFERENCE_RESOLUTION, FPS_LOG_INTERVAL, PERFORMANCE_MODE,
    CAMERA_FAIL_THRESHOLD, CAMERA_RECONNECT_TRIES, CAMERA_RECONNECT_DELAY,
    METRICS_SAVE_INTERVAL, LOG_DIR, TRACKER_TYPE
)

log = get_logger("detect")
_ocr_reader = None

def _get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        log.info("Loading EasyOCR...")
        _ocr_reader = easyocr.Reader(['en'], gpu=USE_GPU)
        log.info("EasyOCR ready.")
    return _ocr_reader

# ── GPU VALIDATION ──────────────────────────────────────────
def validate_gpu():
    """Check GPU availability, print info, return whether to use GPU."""
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

# ── PERFORMANCE MONITOR ────────────────────────────────────
class PerfMonitor:
    def __init__(self, window=30):
        self._times = deque(maxlen=window)
        self._frame_start = 0
        self.current_fps = 0.0
        self.avg_fps = 0.0
        self.frame_ms = 0.0
        self._all_fps = []

    def tick_start(self):
        self._frame_start = time.perf_counter()

    def tick_end(self):
        now = time.perf_counter()
        self.frame_ms = (now - self._frame_start) * 1000
        self._times.append(now)
        if len(self._times) >= 2:
            elapsed = self._times[-1] - self._times[0]
            self.current_fps = (len(self._times) - 1) / elapsed if elapsed > 0 else 0
        self._all_fps.append(self.current_fps)
        self.avg_fps = sum(self._all_fps[-200:]) / min(len(self._all_fps), 200)

    def overlay(self, frame):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (w-220, 5), (w-5, 75), (0,0,0), -1)
        cv2.rectangle(frame, (w-220, 5), (w-5, 75), (0,255,255), 1)
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (w-210, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(frame, f"Avg: {self.avg_fps:.1f}", (w-210, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200), 1)
        cv2.putText(frame, f"{self.frame_ms:.1f}ms", (w-210, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,180,180), 1)

# ── STATS TRACKER ──────────────────────────────────────────
class StatsTracker:
    def __init__(self):
        self.start_time = time.time()
        self.total_frames = 0
        self.frames_dropped = 0
        self.total_detections = 0
        self.confidences = []
        self._path = os.path.join(LOG_DIR, "system_metrics.json")
        os.makedirs(LOG_DIR, exist_ok=True)

    def record_detection(self, conf):
        self.total_detections += 1
        self.confidences.append(conf)

    def save(self):
        uptime = time.time() - self.start_time
        avg_conf = sum(self.confidences)/len(self.confidences) if self.confidences else 0
        drop_rate = self.frames_dropped/max(self.total_frames,1)*100
        data = {
            "uptime_seconds": round(uptime, 1),
            "total_frames": self.total_frames,
            "frames_dropped": self.frames_dropped,
            "frame_drop_rate_pct": round(drop_rate, 2),
            "total_detections": self.total_detections,
            "average_confidence": round(avg_conf, 3),
            "last_updated": datetime.datetime.now().isoformat(),
        }
        try:
            tmp = self._path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._path)
        except Exception as e:
            log.error(f"Metrics save failed: {e}")

# ── SIMPLE IoU TRACKER ─────────────────────────────────────
class SimpleTracker:
    """Lightweight IoU-based multi-object tracker."""
    def __init__(self, iou_thresh=0.3, max_lost=10):
        self._next_id = 1
        self._tracks = {}  # id -> {"bbox", "lost", "data"}
        self._iou_thresh = iou_thresh
        self._max_lost = max_lost

    def _iou(self, a, b):
        xa = max(a[0], b[0]); ya = max(a[1], b[1])
        xb = min(a[2], b[2]); yb = min(a[3], b[3])
        inter = max(0, xb-xa) * max(0, yb-ya)
        aa = (a[2]-a[0])*(a[3]-a[1]); ab = (b[2]-b[0])*(b[3]-b[1])
        return inter / (aa + ab - inter + 1e-6)

    def update(self, detections):
        """detections: list of (x1,y1,x2,y2,conf,label). Returns list of (track_id, x1,y1,x2,y2,conf,label)."""
        matched = []
        used_det = set()
        used_trk = set()
        # Match existing tracks to detections by IoU
        for tid, trk in self._tracks.items():
            best_iou, best_idx = 0, -1
            for i, d in enumerate(detections):
                if i in used_det:
                    continue
                iou = self._iou(trk["bbox"], d[:4])
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_iou >= self._iou_thresh and best_idx >= 0:
                d = detections[best_idx]
                self._tracks[tid] = {"bbox": d[:4], "lost": 0}
                matched.append((tid, *d))
                used_det.add(best_idx)
                used_trk.add(tid)
        # New tracks for unmatched detections
        for i, d in enumerate(detections):
            if i not in used_det:
                tid = self._next_id; self._next_id += 1
                self._tracks[tid] = {"bbox": d[:4], "lost": 0}
                matched.append((tid, *d))
        # Age unmatched tracks
        for tid in list(self._tracks):
            if tid not in used_trk and tid in self._tracks:
                self._tracks[tid]["lost"] += 1
                if self._tracks[tid]["lost"] > self._max_lost:
                    del self._tracks[tid]
        return matched

    @property
    def active_ids(self):
        return set(self._tracks.keys())

# ── DETECTION METHODS (unchanged logic) ────────────────────
def method_yolo(model, frame, gpu):
    fh, fw = frame.shape[:2]
    frame_area = fw * fh
    results = model(frame, verbose=False, half=gpu, imgsz=INFERENCE_RESOLUTION)
    dets = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue
            lid = int(box.cls[0])
            label = model.names[lid]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bw, bh = x2-x1, y2-y1
            ar = bw*bh / frame_area
            if ar < MIN_BOX_AREA_RATIO or ar > MAX_BOX_AREA_RATIO:
                continue
            if bh > 0 and bw/bh < MIN_ASPECT_RATIO and label == "ambulance":
                continue
            dets.append((x1, y1, x2, y2, conf, label))
    return dets

def _detect_red_cross_strict(crop):
    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ch, cw = crop.shape[:2]; crop_area = ch * cw
    white_mask = cv2.inRange(hsv, (0,0,180), (180,40,255))
    if cv2.countNonZero(white_mask)/crop_area < 0.25:
        return False
    m1 = cv2.inRange(hsv, (0,130,120), (8,255,255))
    m2 = cv2.inRange(hsv, (165,130,120), (180,255,255))
    mask = cv2.bitwise_or(m1, m2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a < crop_area*RED_CROSS_MIN_AREA or a > crop_area*0.10:
            continue
        x,y,w,h = cv2.boundingRect(cnt)
        if w < 8 or h < 8: continue
        ar2 = float(w)/float(h)
        if ar2 < 0.6 or ar2 > 1.7: continue
        hull_a = cv2.contourArea(cv2.convexHull(cnt))
        if hull_a == 0: continue
        sol = a/hull_a
        if sol < RED_CROSS_MIN_SOLIDITY or sol > RED_CROSS_MAX_SOLIDITY: continue
        roi = white_mask[y:y+h, x:x+w]
        if roi.size > 0 and cv2.countNonZero(roi)/roi.size < 0.15: continue
        return True
    return False

def method_symbol_ocr(frame, x1, y1, x2, y2, frame_count, cache, det_id, fps):
    h, w = frame.shape[:2]
    crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
    if crop.size == 0: return False, ""
    if _detect_red_cross_strict(crop):
        cache[det_id] = (True, "RED_CROSS")
        return True, "RED_CROSS"
    # Smart OCR: skip if already cached for this track ID
    if det_id in cache:
        cached = cache[det_id]
        if cached[0]:  # already found text — don't re-OCR
            return cached
    # Dynamic OCR interval: slow down if FPS drops
    ocr_interval = OCR_EVERY_N_FRAMES
    if fps > 0 and fps < 10:
        ocr_interval = OCR_EVERY_N_FRAMES * 2
    if frame_count % ocr_interval == 0:
        ocr = _get_ocr()
        pad = 10
        padded = frame[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]
        try:
            results = ocr.readtext(padded)
            for (_, text, prob) in results:
                if prob < 0.4: continue
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

_siren_buf = {}
def method_color_siren(frame, x1, y1, x2, y2, det_id):
    h, w = frame.shape[:2]
    crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
    if crop.size == 0: return False, "unknown", False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    px = crop.shape[0]*crop.shape[1]
    if px == 0: return False, "unknown", False
    white_r = cv2.countNonZero(cv2.inRange(hsv,(0,0,180),(180,40,255)))/px
    yellow_r = cv2.countNonZero(cv2.inRange(hsv,(15,80,150),(35,255,255)))/px
    color = "white" if white_r > 0.35 else ("yellow" if yellow_r > 0.25 else "other")
    rm1 = cv2.inRange(hsv,(0,100,100),(10,255,255))
    rm2 = cv2.inRange(hsv,(160,100,100),(180,255,255))
    has_red = cv2.countNonZero(cv2.bitwise_or(rm1,rm2))/px > 0.04
    sh = max(1, int(crop.shape[0]*0.2))
    siren = crop[0:sh, :]
    if siren.size > 0:
        sv = cv2.cvtColor(siren, cv2.COLOR_BGR2HSV)
        rs = cv2.countNonZero(cv2.inRange(sv,(0,150,150),(10,255,255)))
        bs = cv2.countNonZero(cv2.inRange(sv,(100,150,150),(130,255,255)))
        if det_id not in _siren_buf: _siren_buf[det_id] = deque(maxlen=15)
        _siren_buf[det_id].append(rs+bs)
    has_blink = False
    if det_id in _siren_buf and len(_siren_buf[det_id]) >= 8:
        vals = list(_siren_buf[det_id])
        mean = sum(vals)/len(vals)
        if mean > 0:
            var = sum((v-mean)**2 for v in vals)/len(vals)
            has_blink = var > 1000
    return (color in ("white","yellow")) and (has_red or has_blink), color, has_red

# ── HELPERS ────────────────────────────────────────────────
def detect_lane(x1,y1,x2,y2,fw,fh):
    cx, cy = (x1+x2)/2/fw, (y1+y2)/2/fh
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

def _lane_priority(lane):
    return {"north": 4, "south": 3, "east": 2, "west": 1}.get(lane, 0)

class ConfirmTracker:
    def __init__(self, required=CONFIRM_FRAMES):
        self.required = required
        self._streaks = {}  # track_id -> streak count
        self._confirmed = set()

    def update(self, track_id, detected):
        if detected:
            self._streaks[track_id] = self._streaks.get(track_id, 0) + 1
            if self._streaks[track_id] >= self.required:
                self._confirmed.add(track_id)
                return True
        else:
            self._streaks[track_id] = 0
            self._confirmed.discard(track_id)
        return track_id in self._confirmed

    def is_confirmed(self, track_id):
        return track_id in self._confirmed

    def reset(self, track_id):
        self._streaks.pop(track_id, None)
        self._confirmed.discard(track_id)

# ── CAMERA WITH WATCHDOG ───────────────────────────────────
def _open_camera(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error(f"Cannot open: {source}")
        return None
    return cap

def _reconnect_camera(source):
    for attempt in range(1, CAMERA_RECONNECT_TRIES+1):
        log.warning(f"Camera reconnect attempt {attempt}/{CAMERA_RECONNECT_TRIES}...")
        time.sleep(CAMERA_RECONNECT_DELAY)
        cap = _open_camera(source)
        if cap:
            log.info("Camera reconnected ✓")
            return cap
    log.error("Camera reconnect failed — giving up.")
    return None

# ── MAIN LOOP ──────────────────────────────────────────────
def run_detection(junction_id, video_source):
    from firebase_sender import send_detection, clear_detection
    from csv_logger import log_detection, get_csv_path

    if not os.path.exists(MODEL_PATH):
        log.error(f"Model not found: {MODEL_PATH}")
        return

    gpu = validate_gpu() and USE_GPU
    model = YOLO(MODEL_PATH)
    log.info(f"[{junction_id}] Model loaded | Mode: {PERFORMANCE_MODE} | GPU: {gpu}")

    cap = _open_camera(video_source)
    if not cap: return

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    log.info(f"[{junction_id}] {fw}x{fh} @ {src_fps:.0f}fps | Skip: 1/{FRAME_SKIP} | Res: {INFERENCE_RESOLUTION}")

    fc = 0
    fail_count = 0
    ocr_cache = {}
    tracker = SimpleTracker(iou_thresh=0.3, max_lost=15)
    confirm = ConfirmTracker()
    perf = PerfMonitor()
    stats = StatsTracker()
    last_processed_frame = None
    prev_confirmed_ids = set()
    _shutdown = [False]

    def _sighandler(sig, frame):
        _shutdown[0] = True
        log.info("Shutdown signal received.")
    signal.signal(signal.SIGINT, _sighandler)

    while not _shutdown[0]:
        perf.tick_start()
        ret, frame = cap.read()

        if not ret:
            fail_count += 1
            if fail_count >= CAMERA_FAIL_THRESHOLD:
                if isinstance(video_source, str) and os.path.isfile(video_source):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    fail_count = 0
                    continue
                cap.release()
                cap = _reconnect_camera(video_source)
                if not cap:
                    break
                fail_count = 0
            continue

        fail_count = 0
        fc += 1
        stats.total_frames = fc

        # Frame skip — display last processed frame on skipped frames
        if FRAME_SKIP > 1 and fc % FRAME_SKIP != 0:
            if last_processed_frame is not None and DISPLAY_WINDOW:
                cv2.imshow(f"ASEP - {junction_id}", last_processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            perf.tick_end()
            continue

        # ── YOLO detection ──
        raw_dets = method_yolo(model, frame, gpu)
        ambulance_dets = [d for d in raw_dets if d[5] == "ambulance"]

        # ── Track across frames ──
        tracked = tracker.update(ambulance_dets)

        # ── Multi-ambulance: evaluate each tracked detection ──
        confirmed_this_frame = []
        for (tid, x1, y1, x2, y2, conf, label) in tracked:
            did = f"T{tid}"
            votes = 1
            methods = ["YOLO"]

            sym_ok, sym_text = method_symbol_ocr(
                frame, x1, y1, x2, y2, fc, ocr_cache, did, perf.current_fps)
            if sym_ok:
                votes += 1
                methods.append("SYMBOL" if "CROSS" in sym_text else "OCR")

            col_ok, dom_col, has_red = method_color_siren(frame, x1, y1, x2, y2, did)
            if col_ok:
                votes += 1
                methods.append("COLOR")

            passed = votes >= VOTING_THRESHOLD
            is_confirmed = confirm.update(tid, passed)

            if is_confirmed:
                lane = detect_lane(x1, y1, x2, y2, fw, fh)
                atype = classify_type(dom_col, has_red, sym_text)
                priority = conf + 0.1*votes + 0.05*_lane_priority(lane)
                confirmed_this_frame.append({
                    "tid": tid, "lane": lane, "conf": conf,
                    "method": "+".join(methods), "type": atype,
                    "color": dom_col, "text": sym_text or "none",
                    "votes": votes, "bbox": (x1,y1,x2,y2), "priority": priority,
                })

        # Sort by priority (highest first)
        confirmed_this_frame.sort(key=lambda d: d["priority"], reverse=True)
        current_confirmed_ids = {d["tid"] for d in confirmed_this_frame}

        # ── Draw + send for all confirmed ──
        for i, det in enumerate(confirmed_this_frame):
            x1,y1,x2,y2 = det["bbox"]
            stats.record_detection(det["conf"])
            # Draw box
            color_bgr = (0,0,255) if i == 0 else (0,165,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color_bgr, 3)
            cv2.putText(frame, f"AMB T{det['tid']} {det['conf']:.2f} [{det['method']}]",
                        (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2)
            cv2.putText(frame, f"{det['lane'].upper()} | {det['type']} | P{i+1}",
                        (x1,y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,255), 1)
            # CSV always
            log_detection(junction_id, det["lane"], det["conf"],
                          det["method"], det["type"], det["color"], det["text"])
            # Firebase: send highest priority only
            if i == 0:
                try:
                    send_detection(junction_id, det["lane"], det["conf"],
                                   det["method"], det["type"], det["color"], det["text"])
                except Exception as e:
                    log.warning(f"Firebase send failed (CSV saved): {e}")

        # ── Clear for tracks that lost confirmation ──
        lost_ids = prev_confirmed_ids - current_confirmed_ids
        if lost_ids and not current_confirmed_ids:
            try:
                clear_detection(junction_id)
            except Exception:
                pass
        prev_confirmed_ids = current_confirmed_ids

        # ── HUD ──
        n = len(confirmed_this_frame)
        if n > 0:
            cv2.putText(frame, f"AMBULANCE x{n} CONFIRMED!", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            cv2.putText(frame, "SCANNING...", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,0), 2)

        # Mode badge
        cv2.putText(frame, f"MedRoute AI | {junction_id} | {PERFORMANCE_MODE}",
                    (10, fh-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)

        perf.tick_end()
        perf.overlay(frame)
        last_processed_frame = frame.copy()

        # ── Periodic logs ──
        if fc % FPS_LOG_INTERVAL == 0:
            log.info(f"[{junction_id}] Frame {fc} | FPS: {perf.current_fps:.1f} "
                     f"(avg {perf.avg_fps:.1f}) | {perf.frame_ms:.0f}ms | "
                     f"Detections: {stats.total_detections}")
        if fc % METRICS_SAVE_INTERVAL == 0:
            stats.save()

        if DISPLAY_WINDOW:
            cv2.imshow(f"ASEP - {junction_id}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # ── Cleanup ──
    stats.save()
    if cap: cap.release()
    cv2.destroyAllWindows()
    log.info(f"[{junction_id}] Session complete | Frames: {fc} | Detections: {stats.total_detections}")


# ── ENTRY POINT ────────────────────────────────────────────
if __name__ == '__main__':
    import webbrowser
    from config import SYSTEM_NAME, VERSION, BASE_DIR, SERVICE_ACCOUNT_KEY
    from csv_logger import get_csv_path

    missing = []
    if not SERVICE_ACCOUNT_KEY or not os.path.exists(SERVICE_ACCOUNT_KEY):
        missing.append("Firebase key not found. Set FIREBASE_KEY_PATH in .env")
    if not os.path.exists(MODEL_PATH):
        missing.append(f"Model not found at MODEL_PATH")
    if str(VIDEO_SOURCE) != "0" and not os.path.exists(VIDEO_SOURCE):
        missing.append(f"Video source not found: {VIDEO_SOURCE}")
    if missing:
        for e in missing: log.error(e)
        sys.exit(1)

    log.info("=" * 55)
    log.info(f"  {SYSTEM_NAME} v{VERSION}")
    log.info(f"  Mode: {PERFORMANCE_MODE} | Skip: 1/{FRAME_SKIP} | Res: {INFERENCE_RESOLUTION}")
    log.info(f"  Voting: {VOTING_THRESHOLD}/3 | Debounce: {CONFIRM_FRAMES} frames")
    log.info(f"  CSV: {get_csv_path()}")
    log.info("=" * 55)

    try:
        from firebase_sender import initialize_junctions
        initialize_junctions()
    except Exception as e:
        log.warning(f"Firebase init failed: {e} — CSV-only mode.")

    dashboard = os.path.join(BASE_DIR, "dashboard", "index.html")
    if os.path.exists(dashboard):
        log.info("Opening dashboard...")
        webbrowser.open(f"file:///{dashboard}")

    jid = list(JUNCTIONS.keys())[0]
    run_detection(jid, JUNCTIONS[jid])
