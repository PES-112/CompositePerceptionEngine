"""
pipeline.py
===========
CPE Perception Stack — Stage 1 Orchestrator: RGB + Depth → structured rows.

Ties together yolo_tracker, depth_loader, and physics into the main
perception loop.  Supports nth-frame skip and a streaming generator API.

Public API:
    run_perception_stream(...)  → Generator[list[dict]]  (yields rows per frame)
    run_perception(...)         → list[dict]              (backward compat wrapper)
"""

from collections import defaultdict, deque
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
from tqdm import tqdm

from src.perception_stack.yolo_tracker  import YoloTracker
from src.perception_stack.depth_loader  import load_depth_map, median_depth_in_box
from src.perception_stack.physics       import compute_bearing, compute_velocity

VELOCITY_WINDOW    = 5     # number of frames in rolling velocity buffer
PROXIMITY_M        = 8.0   # alert threshold for unlabeled obstacles
OBSTACLE_GRID_COLS = 5     # number of columns for grid sweep


# ── Grid-based unlabeled obstacle detection ───────────────────────────────────

def detect_unlabeled_obstacles(
    depth_map: np.ndarray,
    frame_w: int,
    frame_h: int,
    yolo_detections: list[dict],
) -> list[dict]:
    """
    Scan the lower 60% of the frame in a grid of vertical columns looking
    for depth-confirmed obstacles NOT already covered by YOLO detections.

    Returns a list of pseudo-detection dicts (may be empty).  Each covers
    one column where an unaccounted obstacle was found.
    """
    if depth_map is None:
        return []

    dh, dw = depth_map.shape[:2]
    y1_scan = int(frame_h * 0.4)
    y2_scan = frame_h
    dy1 = min(y1_scan, dh)
    dy2 = min(y2_scan, dh)

    # Build mask of pixels already claimed by YOLO boxes
    yolo_mask = np.zeros((dh, dw), dtype=bool)
    for d in yolo_detections:
        ex1 = max(0, min(d["x1"], dw))
        ey1 = max(0, min(d["y1"], dh))
        ex2 = max(0, min(d["x2"], dw))
        ey2 = max(0, min(d["y2"], dh))
        yolo_mask[ey1:ey2, ex1:ex2] = True

    obstacles = []
    col_w = frame_w // OBSTACLE_GRID_COLS

    for col in range(OBSTACLE_GRID_COLS):
        x1 = col * col_w
        x2 = x1 + col_w
        dx1 = min(x1, dw)
        dx2 = min(x2, dw)
        if dx1 >= dx2:
            continue

        roi_depth = depth_map[dy1:dy2, dx1:dx2]
        roi_mask  = yolo_mask[dy1:dy2, dx1:dx2]

        # Only look at pixels NOT already covered by YOLO
        unmasked = roi_depth[~roi_mask]
        close = unmasked[(unmasked > 0.3) & (unmasked < PROXIMITY_M)]

        if close.size > 0:
            dist = float(close.min())
            cx   = (x1 + x2) / 2.0
            obstacles.append({
                "track_id":   f"obs_{col}",
                "class_name": "unlabeled_obstacle",
                "confidence": 1.0,
                "x1": x1, "y1": y1_scan, "x2": x2, "y2": frame_h,
                "cx": cx,
                "min_depth": dist,
            })

    return obstacles


# ── Front-to-back depth sorting ───────────────────────────────────────────────

def _rough_depth(det: dict, depth_map: np.ndarray | None) -> float:
    """Quick center-pixel depth for sorting.  Unlabeled obstacles use min_depth."""
    if det.get("class_name") == "unlabeled_obstacle":
        return det.get("min_depth", 99.0)
    if depth_map is not None:
        h_dm, w_dm = depth_map.shape[:2]
        cx = min(int(det["cx"]), w_dm - 1)
        cy = min(int((det["y1"] + det["y2"]) / 2), h_dm - 1)
        v  = depth_map[cy, cx]
        return float(v) if v > 0 else 99.0
    return 99.0


# ── Streaming perception generator ───────────────────────────────────────────

def run_perception_stream(
    rgb_dir:    Path,
    depth_dir:  Path | None,
    fps:        float,
    source:     str = "sanpo",
    frame_step: int = 1,
) -> Generator[list[dict], None, None]:
    """
    Stage 1 streaming perception — yields a list of row dicts per processed frame.

    Args:
        rgb_dir:     Folder of sorted RGB frames (JPEG/PNG).
        depth_dir:   Folder of co-registered depth maps, or None.
        fps:         Video framerate for velocity calculation.
        source:      'sanpo' or 'uasol' — controls depth scale.
        frame_step:  Process every Nth frame (default 1 = all frames).
                     Set to 3 for SANPO preprocessing to cut compute by ~67%.

    Yields:
        List of flat row dicts for each processed frame.
    """
    frame_paths = sorted([
        p for p in rgb_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found in {rgb_dir}")

    tracker = YoloTracker()

    # Per-track rolling depth history for velocity estimation
    depth_history: dict = defaultdict(lambda: deque(maxlen=VELOCITY_WINDOW + 1))

    # Apply nth-frame skip
    selected_indices = range(0, len(frame_paths), frame_step)

    for frame_idx in tqdm(selected_indices, desc="Perception", unit="frame"):
        rgb_path = frame_paths[frame_idx]
        frame = cv2.imread(str(rgb_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        # ── Load matched depth map ──
        depth_map = None
        if depth_dir is not None:
            stem = rgb_path.stem
            for ext in (".float16.gz", ".png", ".npz"):
                depth_path = depth_dir / (stem + ext)
                if depth_path.exists():
                    depth_map = load_depth_map(depth_path, source=source)
                    break

        # ── YOLO + ByteTrack + depth-guided post-processing ──
        detections = tracker.track(frame, depth_map=depth_map)

        # ── Grid-based unlabeled obstacle sweep ──
        unlabeled_list = detect_unlabeled_obstacles(depth_map, w, h, detections)
        detections.extend(unlabeled_list)

        # ── Sort front-to-back for occlusion masking ──
        detections.sort(key=lambda d: _rough_depth(d, depth_map))

        # ── Per-detection depth with occlusion masking ──
        frame_rows: list[dict] = []

        for det in detections:
            tid = det["track_id"]

            if det.get("class_name") == "unlabeled_obstacle":
                distance_m = det["min_depth"]
                tid = f"obs_{frame_idx}_{det['cx']:.0f}"
            else:
                distance_m = None
                if depth_map is not None:
                    # Build occlusion mask from already-processed closer objects
                    occluders = [
                        (r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"])
                        for r in frame_rows
                        if r["distance_m"] is not None
                    ]
                    distance_m = median_depth_in_box(
                        depth_map,
                        det["x1"], det["y1"], det["x2"], det["y2"],
                        exclude_boxes=occluders,
                    )

            # ── Velocity ──
            if distance_m is not None and det.get("class_name") != "unlabeled_obstacle":
                depth_history[tid].append((frame_idx, distance_m))

            velocity_ms = 0.0
            if det.get("class_name") != "unlabeled_obstacle":
                velocity_ms = compute_velocity(list(depth_history[tid]), fps)

            # ── Bearing ──
            bearing = compute_bearing(det["cx"], w)

            frame_rows.append({
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
                "distance_m":  round(distance_m, 2) if distance_m is not None else None,
                "velocity_ms": round(velocity_ms, 3),
            })

        yield frame_rows


# ── Backward-compatible wrapper ───────────────────────────────────────────────

def run_perception(
    rgb_dir:    Path,
    depth_dir:  Path | None,
    fps:        float,
    source:     str = "sanpo",
    frame_step: int = 1,
) -> list[dict]:
    """
    Stage 1 perception — returns all rows as a flat list (backward compatible).

    Wraps run_perception_stream() and materialises the generator.
    See run_perception_stream() for argument docs.
    """
    rows: list[dict] = []
    for frame_rows in run_perception_stream(rgb_dir, depth_dir, fps, source, frame_step):
        rows.extend(frame_rows)
    return rows
