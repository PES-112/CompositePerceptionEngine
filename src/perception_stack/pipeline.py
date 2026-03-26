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

from src.perception_stack.yolo_tracker import YoloTracker
from src.perception_stack.depth_loader import load_depth_map, median_depth_in_box
from src.perception_stack.physics      import compute_bearing, compute_velocity
from src.perception_stack.csv_writer   import CSV_FIELDS

VELOCITY_WINDOW = 5   # number of frames in rolling velocity buffer


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
        # Both SANPO (lidar GT) and UASOL (stereo GT) store depth as 16-bit PNG
        # with the same stem name as the RGB frame.
        depth_map = None
        if depth_dir is not None:
            depth_path = depth_dir / (rgb_path.stem + ".png")
            depth_map  = load_depth_map(depth_path, source=source)

        for det in detections:
            tid = det["track_id"]

            # ── Depth ──
            distance_m = None
            if depth_map is not None:
                distance_m = median_depth_in_box(
                    depth_map, det["x1"], det["y1"], det["x2"], det["y2"]
                )

            # ── Velocity ──
            if distance_m is not None:
                depth_history[tid].append((frame_idx, distance_m))
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
