"""
extract_perception.py
=====================
CPE Phase 1 — Perception Pipeline

Runs YOLO + ByteTrack over a folder of video clips (SANPO or UASOL format).
For every tracked object it computes:
    - distance_m  : metric depth from the dataset's ground-truth depth map
    - velocity_ms : closing velocity estimated across a rolling window of frames
    - bearing_deg : approximate left/right bearing from image centre

Outputs a JSON file suitable for downstream SLM-1 fact-sheet generation.

Usage
-----
python tools/extract_perception.py \\
    --rgb_dir   data/sanpo/sample/rgb \\
    --depth_dir data/sanpo/sample/depth \\
    --out       data/processed/sanpo_tracks.json \\
    --fps       30

For UASOL (no ground-truth depth maps, monodepth fallback):
python tools/extract_perception.py \\
    --rgb_dir   data/uasol/sample/rgb \\
    --out       data/processed/uasol_tracks.json \\
    --fps       30 \\
    --no_depth
"""

import argparse
import json
import os
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ── Constants ────────────────────────────────────────────────────────────────
YOLO_MODEL      = "yolo11n.pt"   # swap to yolo12n.pt when ultralytics ships it
CONF_THRESHOLD  = 0.30
DEPTH_SCALE     = 0.001          # SANPO depth PNGs store millimetres → convert to metres
VELOCITY_WINDOW = 5              # frames used for rolling-average velocity
MIN_DEPTH_M     = 0.5            # discard objects closer than 50cm (sensor noise)
MAX_DEPTH_M     = 30.0           # discard objects farther than 30m (not relevant)

# COCO class → rough pedestrian-nav hazard weight
CLASS_SEVERITY = {
    "person":       1.0,
    "bicycle":      1.2,
    "car":          2.0,
    "motorcycle":   1.8,
    "bus":          2.5,
    "truck":        2.5,
    "dog":          0.8,
}
DEFAULT_SEVERITY = 1.0


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_depth_map(depth_path: Path) -> np.ndarray | None:
    """Load a 16-bit SANPO depth PNG and return metric depth in metres."""
    if not depth_path.exists():
        return None
    raw = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH).astype(np.float32)
    return raw * DEPTH_SCALE


def median_depth_in_box(depth_map: np.ndarray, x1: int, y1: int,
                         x2: int, y2: int) -> float | None:
    """Return the median metric depth (metres) inside a bounding box."""
    roi = depth_map[y1:y2, x1:x2]
    valid = roi[(roi > MIN_DEPTH_M) & (roi < MAX_DEPTH_M)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def compute_bearing(cx_pixel: float, frame_width: int, hfov_deg: float = 70.0) -> float:
    """
    Convert pixel x-coordinate to a bearing angle in degrees.
    Negative = left,  Positive = right,  0 = straight ahead.
    Assumes a horizontal field-of-view of 70° (typical phone camera).
    """
    normalised = (cx_pixel - frame_width / 2) / (frame_width / 2)  # [-1, 1]
    return normalised * (hfov_deg / 2)


def compute_ttc(distance_m: float, velocity_ms: float) -> float | None:
    """Time-to-collision estimate. None if object is moving away."""
    if velocity_ms <= 0:
        return None
    return distance_m / velocity_ms


def kinetic_score(distance_m: float, velocity_ms: float, class_name: str) -> float:
    """
    K = severity × v² / max(d, ε)
    Higher K = higher threat.
    """
    severity = CLASS_SEVERITY.get(class_name, DEFAULT_SEVERITY)
    eps = 0.5
    return severity * (velocity_ms ** 2) / max(distance_m, eps)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def process_video(rgb_dir: Path, depth_dir: Path | None, fps: float,
                  no_depth: bool) -> list[dict]:
    """
    Iterate through a sorted list of RGB frames (JPEG/PNG).
    Returns a list of frame records ready for JSON.
    """
    model = YOLO(YOLO_MODEL)
    model.overrides["conf"] = CONF_THRESHOLD
    model.overrides["tracker"] = "bytetrack.yaml"   # built into ultralytics ≥8.1

    frame_paths = sorted([
        p for p in rgb_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])

    if not frame_paths:
        raise FileNotFoundError(f"No images found in {rgb_dir}")

    # Rolling depth buffer per track_id for velocity estimation
    # deque of (frame_index, distance_m)
    depth_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=VELOCITY_WINDOW + 1))

    records = []

    for frame_idx, rgb_path in enumerate(frame_paths):
        frame = cv2.imread(str(rgb_path))
        h, w = frame.shape[:2]

        # ── YOLO inference + ByteTrack ──
        results = model.track(frame, persist=True, verbose=False)

        # ── Depth map for this frame ──
        depth_map = None
        if not no_depth and depth_dir is not None:
            # SANPO convention: same filename but in depth/ folder, extension .png
            depth_path = depth_dir / (rgb_path.stem + ".png")
            depth_map = load_depth_map(depth_path)

        frame_objects = []

        if results[0].boxes is None or results[0].boxes.id is None:
            # No detections this frame
            records.append({"frame_idx": frame_idx, "objects": []})
            continue

        boxes   = results[0].boxes.xyxy.cpu().numpy()
        ids     = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs   = results[0].boxes.conf.cpu().numpy()
        names   = results[0].names  # {int: str}

        for box, tid, cls_idx, conf in zip(boxes, ids, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            class_name = names[cls_idx]

            # ── Distance ──
            distance_m = None
            if depth_map is not None:
                distance_m = median_depth_in_box(depth_map, x1, y1, x2, y2)

            # ── Velocity (rolling window) ──
            velocity_ms = 0.0
            if distance_m is not None:
                depth_history[tid].append((frame_idx, distance_m))
                hist = list(depth_history[tid])
                if len(hist) >= 2:
                    # Take oldest and newest in the window
                    (f0, d0), (f1, d1) = hist[0], hist[-1]
                    dt = (f1 - f0) / fps
                    if dt > 0:
                        raw_v = (d0 - d1) / dt   # positive = closing
                        # Simple moving average: just clamp negatives to 0
                        velocity_ms = max(0.0, raw_v)

            # ── Bearing ──
            bearing = compute_bearing(cx, w)

            # ── TTC & Kinetic Score ──
            ttc = compute_ttc(distance_m, velocity_ms) if distance_m else None
            k   = kinetic_score(distance_m, velocity_ms, class_name) if distance_m else 0.0

            frame_objects.append({
                "track_id":    int(tid),
                "class":       class_name,
                "confidence":  round(float(conf), 3),
                "bbox":        [x1, y1, x2, y2],
                "cx_px":       round(cx, 1),
                "bearing_deg": round(bearing, 2),
                "distance_m":  round(distance_m, 2) if distance_m else None,
                "velocity_ms": round(velocity_ms, 3),
                "ttc_s":       round(ttc, 2) if ttc else None,
                "kinetic_score": round(k, 4),
            })

        # Sort by kinetic score descending so SLM-1 sees the hottest objects first
        frame_objects.sort(key=lambda o: o["kinetic_score"], reverse=True)

        records.append({
            "frame_idx": frame_idx,
            "source_file": rgb_path.name,
            "objects": frame_objects,
        })

        # ── progress log ──
        if frame_idx % 50 == 0:
            print(f"  [{frame_idx}/{len(frame_paths)}] processed — "
                  f"{len(frame_objects)} objects tracked")

    return records


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CPE Perception Pipeline — Phase 1")
    parser.add_argument("--rgb_dir",   required=True, help="Folder of RGB frames (JPG/PNG)")
    parser.add_argument("--depth_dir", default=None,  help="Folder of 16-bit depth PNGs (SANPO)")
    parser.add_argument("--out",       required=True, help="Output JSON file path")
    parser.add_argument("--fps",       type=float, default=30.0, help="Video framerate (default: 30)")
    parser.add_argument("--no_depth",  action="store_true",
                        help="Skip depth loading (for UASOL which has no depth GT)")
    args = parser.parse_args()

    rgb_dir   = Path(args.rgb_dir)
    depth_dir = Path(args.depth_dir) if args.depth_dir else None
    out_path  = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== CPE Perception Pipeline ===")
    print(f"  RGB frames : {rgb_dir}")
    print(f"  Depth maps : {depth_dir or 'None (no_depth mode)'}")
    print(f"  FPS        : {args.fps}")
    print(f"  Output     : {out_path}\n")

    records = process_video(rgb_dir, depth_dir, args.fps, args.no_depth)

    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    n_frames  = len(records)
    n_objects = sum(len(r["objects"]) for r in records)
    print(f"\n✅ Done — {n_frames} frames, {n_objects} object detections → {out_path}")


if __name__ == "__main__":
    main()
