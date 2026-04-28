"""
MedRoute AI — SORT Tracker
=============================
SORT (Simple Online and Realtime Tracking) implementation with Kalman filter.
Falls back to the original SimpleTracker if scipy is unavailable.

Based on: Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.
"""

import numpy as np
from logger import get_logger

log = get_logger("tracker")

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    log.warning("scipy not found — SORT tracker will fall back to SimpleTracker.")


# ── Coordinate Conversion ─────────────────────────────────────

def _bbox_to_z(bbox):
    """Convert [x1,y1,x2,y2] to [cx, cy, area, aspect_ratio]."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h) if h > 0 else 1.0
    return np.array([cx, cy, s, r]).reshape((4, 1))


def _z_to_bbox(z):
    """Convert [cx, cy, area, aspect_ratio] to [x1,y1,x2,y2]."""
    w = np.sqrt(z[2] * z[3]) if z[2] * z[3] > 0 else 0
    h = z[2] / w if w > 0 else 0
    return np.array([
        z[0] - w / 2.0,
        z[1] - h / 2.0,
        z[0] + w / 2.0,
        z[1] + h / 2.0,
    ]).flatten()


def _iou_batch(bb_det, bb_trk):
    """Compute IoU between two sets of bboxes. Returns NxM matrix."""
    nd = len(bb_det)
    nt = len(bb_trk)
    if nd == 0 or nt == 0:
        return np.empty((nd, nt))

    det = np.array(bb_det)
    trk = np.array(bb_trk)

    xx1 = np.maximum(det[:, 0:1], trk[:, 0:1].T)
    yy1 = np.maximum(det[:, 1:2], trk[:, 1:2].T)
    xx2 = np.minimum(det[:, 2:3], trk[:, 2:3].T)
    yy2 = np.minimum(det[:, 3:4], trk[:, 3:4].T)

    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_d = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
    area_t = (trk[:, 2] - trk[:, 0]) * (trk[:, 3] - trk[:, 1])
    union = area_d[:, None] + area_t[None, :] - inter + 1e-6
    return inter / union


# ── Kalman Box Tracker ──────────────────────────────────────

class _KalmanBoxTracker:
    """Single-object Kalman filter tracker.

    State: [cx, cy, area, aspect_ratio, v_cx, v_cy, v_area] (7-dim)
    Measurement: [cx, cy, area, aspect_ratio] (4-dim)
    """
    _count = 0

    def __init__(self, bbox):
        _KalmanBoxTracker._count += 1
        self.id = _KalmanBoxTracker._count

        # State and covariance
        self.x = np.zeros((7, 1))           # state
        self.x[:4] = _bbox_to_z(bbox)
        self.P = np.eye(7) * 10.0           # covariance
        self.P[4:, 4:] *= 1000.0            # high uncertainty on velocities

        # Transition matrix (constant velocity model)
        self.F = np.eye(7)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0

        # Measurement matrix
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)

        # Noise matrices
        self.Q = np.eye(7) * 1.0            # process noise
        self.Q[4:, 4:] *= 0.01
        self.R = np.eye(4) * 1.0            # measurement noise
        self.R[2, 2] *= 10.0                # area has more noise
        self.R[3, 3] *= 10.0

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def predict(self):
        """Advance state prediction by one step."""
        # Prevent negative area
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _z_to_bbox(self.x[:4].flatten())

    def update(self, bbox):
        """Update state with a matched detection."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        z = _bbox_to_z(bbox)
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        # Update
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def get_state(self):
        """Return current bbox estimate as [x1,y1,x2,y2]."""
        return _z_to_bbox(self.x[:4].flatten())


# ── SORT Tracker ─────────────────────────────────────────────

class SORTTracker:
    """SORT multi-object tracker using Kalman filter + Hungarian assignment.

    Interface compatible with the original SimpleTracker.
    """

    def __init__(self, max_age=15, min_hits=3, iou_threshold=0.3):
        self._max_age = max_age
        self._min_hits = min_hits
        self._iou_threshold = iou_threshold
        self._trackers = []

    def update(self, detections):
        """Update tracker with new detections.

        Args:
            detections: list of (x1, y1, x2, y2, conf, label)

        Returns:
            list of (track_id, x1, y1, x2, y2, conf, label)
        """
        # Predict existing trackers
        predicted_bboxes = []
        to_remove = []
        for i, trk in enumerate(self._trackers):
            pred = trk.predict()
            if np.any(np.isnan(pred)):
                to_remove.append(i)
            else:
                predicted_bboxes.append(pred)
        for i in reversed(to_remove):
            self._trackers.pop(i)

        # Build detection bbox array
        det_bboxes = [d[:4] for d in detections]

        # Associate detections to trackers using IoU + Hungarian
        matched, unmatched_dets, unmatched_trks = self._associate(
            det_bboxes, predicted_bboxes
        )

        # Update matched trackers
        for d_idx, t_idx in matched:
            self._trackers[t_idx].update(detections[d_idx][:4])

        # Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            self._trackers.append(_KalmanBoxTracker(detections[d_idx][:4]))

        # Remove dead trackers
        self._trackers = [
            t for t in self._trackers
            if t.time_since_update <= self._max_age
        ]

        # Build output — only return tracks with enough hits
        results = []
        for trk in self._trackers:
            if trk.time_since_update > 0:
                continue
            bbox = trk.get_state()
            x1, y1, x2, y2 = map(int, bbox)
            # Find the matched detection to get conf and label
            best_det = None
            best_iou = 0
            for d in detections:
                iou = self._single_iou(d[:4], (x1, y1, x2, y2))
                if iou > best_iou:
                    best_iou = iou
                    best_det = d
            if best_det is not None:
                conf, label = best_det[4], best_det[5]
            else:
                conf, label = 0.0, "ambulance"
            results.append((trk.id, x1, y1, x2, y2, conf, label))

        return results

    def _associate(self, det_bboxes, trk_bboxes):
        """Associate detections to trackers using IoU and Hungarian algorithm."""
        if len(trk_bboxes) == 0:
            return [], list(range(len(det_bboxes))), []
        if len(det_bboxes) == 0:
            return [], [], list(range(len(trk_bboxes)))

        iou_matrix = _iou_batch(det_bboxes, trk_bboxes)
        cost_matrix = 1.0 - iou_matrix

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_dets = list(range(len(det_bboxes)))
        unmatched_trks = list(range(len(trk_bboxes)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self._iou_threshold:
                matched.append((r, c))
                if r in unmatched_dets:
                    unmatched_dets.remove(r)
                if c in unmatched_trks:
                    unmatched_trks.remove(c)

        return matched, unmatched_dets, unmatched_trks

    @staticmethod
    def _single_iou(a, b):
        xa = max(a[0], b[0]); ya = max(a[1], b[1])
        xb = min(a[2], b[2]); yb = min(a[3], b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        aa = (a[2] - a[0]) * (a[3] - a[1])
        ab = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (aa + ab - inter + 1e-6)

    @property
    def active_ids(self):
        return {t.id for t in self._trackers if t.time_since_update <= self._max_age}


# ── Simple IoU Tracker (Fallback) ──────────────────────────────

class SimpleTracker:
    """Lightweight IoU-based multi-object tracker (original fallback)."""

    def __init__(self, iou_thresh=0.3, max_lost=10):
        self._next_id = 1
        self._tracks = {}
        self._iou_thresh = iou_thresh
        self._max_lost = max_lost

    def _iou(self, a, b):
        xa = max(a[0], b[0]); ya = max(a[1], b[1])
        xb = min(a[2], b[2]); yb = min(a[3], b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        aa = (a[2] - a[0]) * (a[3] - a[1])
        ab = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (aa + ab - inter + 1e-6)

    def update(self, detections):
        matched = []
        used_det = set()
        used_trk = set()
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
        for i, d in enumerate(detections):
            if i not in used_det:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"bbox": d[:4], "lost": 0}
                matched.append((tid, *d))
        for tid in list(self._tracks):
            if tid not in used_trk and tid in self._tracks:
                self._tracks[tid]["lost"] += 1
                if self._tracks[tid]["lost"] > self._max_lost:
                    del self._tracks[tid]
        return matched

    @property
    def active_ids(self):
        return set(self._tracks.keys())


def create_tracker(max_age=15, min_hits=3, iou_threshold=0.3):
    """Factory: returns SORT tracker if scipy available, else SimpleTracker."""
    if _HAS_SCIPY:
        log.info("Using SORT tracker (Kalman + Hungarian)")
        return SORTTracker(max_age=max_age, min_hits=min_hits,
                           iou_threshold=iou_threshold)
    else:
        log.warning("Using SimpleTracker fallback (no scipy)")
        return SimpleTracker(iou_thresh=iou_threshold, max_lost=max_age)
