"""
pipeline.py
===========
CPE Perception Stack — Stage 1 Orchestrator: RGB + Depth → CSV rows.

Ties together yolo_tracker, depth_loader, and physics into the main
perception loop. Returns a list of flat row dicts ready for csv_writer.

Public API:
    run_perception(rgb_dir, depth_dir, fps, source) → list[dict]
"""

from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from src.perception_stack.yolo_tracker import YoloTracker
from src.perception_stack.depth_loader import load_depth_map, median_depth_in_box
from src.perception_stack.physics      import compute_bearing, compute_velocity
from src.perception_stack.csv_writer   import CSV_FIELDS

VELOCITY_WINDOW = 5   # number of frames in rolling velocity buffer
PROXIMITY_M = 2.5     # alert threshold for unlabeled obstacles

def detect_unlabeled_obstacle(depth_map: np.ndarray, frame_w: int, frame_h: int) -> dict | None:
    """
    Scans the central path for unclassified physical obstacles (poles, bollards, etc).
    Returns a pseudo-detection det dict if an obstacle is < PROXIMITY_M.
    """
    if depth_map is None:
        return None

    # Define walk-path ROI: bottom half, central 40% of the frame
    x1 = int(frame_w * 0.3)
    x2 = int(frame_w * 0.7)
    y1 = int(frame_h * 0.5)
    y2 = frame_h
    
    # Clip to depth map bounds (in case it is smaller than RGB)
    dh, dw = depth_map.shape[:2]
    dx1, dx2 = min(x1, dw), min(x2, dw)
    dy1, dy2 = min(y1, dh), min(y2, dh)
    
    roi = depth_map[dy1:dy2, dx1:dx2]
    
    # Find valid pixels closer than PROXIMITY_M
    close = roi[(roi > 0.3) & (roi < PROXIMITY_M)]
    
    if close.size > 0:
        dist = float(close.min())
        # Return a pseudo YOLO detection covering the ROI
        return {
            "track_id":    9999,   # reserved ID for unlabeled
            "class_name":  "unlabeled_obstacle",
            "confidence":  1.0,
            "x1":          x1,
            "y1":          y1,
            "x2":          x2,
            "y2":          y2,
            "cx":          (x1 + x2) / 2.0,
            "min_depth":   dist
        }
    return None


def run_perception(
    rgb_dir:   Path,
    depth_dir: Path | None,
    fps:       float,
    source:    str = "sanpo",
) -> list[dict]:
    """
    Stage 1 perception loop: YOLO + ByteTrack → depth → physics → CSV rows.

    Args:
        rgb_dir:   Folder of sorted RGB frames (JPEG/PNG).
        depth_dir: Folder of co-registered 16-bit depth PNGs, or None to skip depth.
        fps:       Video framerate — used to convert frame-count deltas to seconds.
        source:    'sanpo' or 'uasol' — controls depth scale applied in depth_loader.

    Returns:
        List of flat dicts with keys matching CSV_FIELDS (csv_writer.py).
        One dict per tracked object per frame.
    """
    frame_paths = sorted([
        p for p in rgb_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found in {rgb_dir}")

    tracker = YoloTracker()

    # Per-track rolling depth history for velocity estimation
    # { track_id: deque[(frame_idx, distance_m)] }
    depth_history: dict = defaultdict(lambda: deque(maxlen=VELOCITY_WINDOW + 1))

    rows: list[dict] = []

    for frame_idx, rgb_path in enumerate(frame_paths):
        frame = cv2.imread(str(rgb_path))
        if frame is None:
            continue
        _, w = frame.shape[:2]

        # ── YOLO + ByteTrack ──
        detections = tracker.track(frame)

        # ── Load matched depth map ──
        # SANPO depth: <stem>.float16.gz  (gzip float16, already in metres)
        # UASOL depth: <stem>.png         (16-bit PNG, scale × 0.256)
        depth_map = None
        if depth_dir is not None:
            stem = rgb_path.stem   # e.g. "000000"
            for ext in (".float16.gz", ".png"):
                depth_path = depth_dir / (stem + ext)
                if depth_path.exists():
                    depth_map = load_depth_map(depth_path, source=source)
                    break
        # ── Find Unlabeled Obstacles (poles, walls) ──
        unlabeled = detect_unlabeled_obstacle(depth_map, w, frame.shape[0])
        if unlabeled:
            # Prevent double-counting if a YOLO object is already tracked at similar depth in that area
            closest_yolo = min(
                [median_depth_in_box(depth_map, d["x1"], d["y1"], d["x2"], d["y2"]) or 99.0 
                 for d in detections], 
                default=99.0
            )
            if unlabeled["min_depth"] < closest_yolo - 0.5:
                detections.append(unlabeled)

        for det in detections:
            tid = det["track_id"]

            # ── Depth ──
            if det.get("class_name") == "unlabeled_obstacle":
                distance_m = det["min_depth"]
                tid = f"9999_{frame_idx}"  # unique stateless ID per frame so it doesn't build fake velocity
            else:
                distance_m = None
                if depth_map is not None:
                    distance_m = median_depth_in_box(
                        depth_map, det["x1"], det["y1"], det["x2"], det["y2"]
                    )

            # ── Velocity ──
            if distance_m is not None and det.get("class_name") != "unlabeled_obstacle":
                depth_history[tid].append((frame_idx, distance_m))
            
            # Static obstacles have 0 velocity
            velocity_ms = 0.0
            if det.get("class_name") != "unlabeled_obstacle":
                velocity_ms = compute_velocity(list(depth_history[tid]), fps)

            # ── Bearing ──
            bearing = compute_bearing(det["cx"], w)

            rows.append({
                "frame_idx":   frame_idx,
                "source":      source,
                "track_id":    tid,
                "class":       det["class_name"],
                "confidence":  det["confidence"],
                "bbox_x1":     det["x1"],
                "bbox_y1":     det["y1"],
                "bbox_x2":     det["x2"],
                "bbox_y2":     det["y2"],
                "cx_px":       round(det["cx"], 1),
                "bearing_deg": round(bearing, 2),
                "distance_m":  round(distance_m, 2) if distance_m is not None else "",
                "velocity_ms": round(velocity_ms, 3),
            })

        if frame_idx % 50 == 0:
            n = sum(1 for r in rows if r["frame_idx"] == frame_idx)
            print(f"  [{frame_idx:05d}/{len(frame_paths)}] {n} objects tracked")

    return rows
