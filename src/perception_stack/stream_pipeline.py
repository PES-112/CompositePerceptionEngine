"""
stream_pipeline.py
==================
CPE Perception Stack — End-to-End Streaming Pipeline.

Runs the perception stack (YOLO + Physics) entirely in memory,
processing frames directly from a generator to avoid disk writes.
Once a session is complete, it immediately builds and appends
its Fact Sheets to train.jsonl.
"""

from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np

from src.perception_stack.yolo_tracker import YoloTracker
from src.perception_stack.depth_loader import median_depth_in_box
from src.perception_stack.physics      import compute_bearing, compute_velocity
from src.perception_stack.pipeline     import VELOCITY_WINDOW, detect_unlabeled_obstacle
from src.perception_stack.fact_sheet_builder import build_fact_sheets


def run_stream_session(
    frame_generator: Iterable,
    fps: float,
    out_jsonl_path: Path,
    lookahead_s: float = 2.0
) -> Tuple[int, int]:
    """
    Runs the entire perception and scoring pipeline in memory for a single session.

    Args:
        frame_generator: Iterable yielding Frame data objects (e.g. from SANPOLoader)
        fps:             Video framerate.
        out_jsonl_path:  Destination Path for the output train.jsonl
        lookahead_s:     Lookahead time for K_future threat confirmation.

    Returns:
        (written, skipped) count of Fact Sheets generated for this session.
    """
    tracker = YoloTracker()
    depth_history: dict = defaultdict(lambda: deque(maxlen=VELOCITY_WINDOW + 1))
    
    # Store all tracking rows for the session grouped by frame_idx
    frames: dict = defaultdict(list)
    
    print("  -> Starting in-memory perception stream...")
    frame_count = 0
    
    for frame_idx, frame_data in enumerate(frame_generator):
        frame_count += 1
        
        # 1. Grab RGB and convert to BGR for standard cv2 processing
        frame = cv2.cvtColor(frame_data.rgb, cv2.COLOR_RGB2BGR)
        h, w = frame.shape[:2]

        # 2. YOLO + ByteTrack
        detections = tracker.track(frame)

        # 3. Depth (directly from the SANPO frame_data float16 array)
        depth_map = frame_data.depth
        
        # 4. Unlabeled Obstacle Detection (poles, walls in the central path)
        unlabeled = detect_unlabeled_obstacle(depth_map, w, h)
        if unlabeled:
            closest_yolo = min(
                [median_depth_in_box(depth_map, d["x1"], d["y1"], d["x2"], d["y2"]) or 99.0 
                 for d in detections], 
                default=99.0
            )
            # prevent double tracking if YOLO already caught something exactly there
            if unlabeled["min_depth"] < closest_yolo - 0.5:
                detections.append(unlabeled)

        # 5. Extract Features & Physics
        for det in detections:
            tid = det["track_id"]

            if det.get("class_name") == "unlabeled_obstacle":
                distance_m = det["min_depth"]
                # Give stateless ID to prevent fake velocity build-up
                tid = f"9999_{frame_idx}"
            else:
                distance_m = None
                if depth_map is not None:
                    distance_m = median_depth_in_box(
                        depth_map, det["x1"], det["y1"], det["x2"], det["y2"]
                    )

            if distance_m is not None and det.get("class_name") != "unlabeled_obstacle":
                depth_history[tid].append((frame_idx, distance_m))
            
            velocity_ms = 0.0
            if det.get("class_name") != "unlabeled_obstacle":
                velocity_ms = compute_velocity(list(depth_history[tid]), fps)

            bearing = compute_bearing(det["cx"], w)

            row = {
                "frame_idx":   frame_idx,
                "source":      "sanpo",
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
            }
            frames[frame_idx].append(row)

        if frame_idx > 0 and frame_idx % 100 == 0:
            print(f"      [Buffer] Processed and discarded {frame_idx} frames...")

    print(f"  -> Session Stream Complete! ({frame_count} frames)")
    print(f"  -> Scoring Fact Sheets & appending to JSONL...")
    
    # 6. Session complete! Immediately build Fact Sheets and flush to disk
    if frame_count > 0:
        written, skipped = build_fact_sheets(
            frames=frames, 
            fps=fps, 
            lookahead_s=lookahead_s, 
            out_path=out_jsonl_path,
            append=True
        )
        print(f"  -> Wrote {written} scenarios (Skipped {skipped})")
        return written, skipped
    else:
        print("  -> No frames processed.")
        return 0, 0
