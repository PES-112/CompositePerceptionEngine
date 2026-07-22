"""
Benchmark CPE perception latency under edge-style streaming conditions.

This measures the Stage 1 real-time path:
RGB frame -> optional depth load -> YOLO26n + ByteTrack -> depth post-processing
-> unlabeled obstacle sweep -> distance / velocity / bearing rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from statistics import mean, median
from time import perf_counter, sleep

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception_stack.depth_loader import load_depth_map, median_depth_in_box
from src.perception_stack.physics import compute_bearing, compute_velocity
from src.perception_stack.pipeline import (
    VELOCITY_WINDOW,
    _rough_depth,
    detect_unlabeled_obstacles,
)
from src.perception_stack.yolo_tracker import YoloTracker


DEFAULT_WEIGHTS = PROJECT_ROOT / "training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt"


EDGE_PROFILES = {
    "gb10_native": {
        "description": "Native GB10 measurement with no artificial slowdown.",
        "target_device": "NVIDIA GB10",
        "reflex_budget_ms": 50.0,
        "cognitive_budget_ms": 500.0,
        "latency_scale": 1.0,
        "sensor_overhead_ms": 0.0,
        "recommended_runtime": "pytorch_fp16_or_native",
    },
    "jetson_orin_nano_8gb": {
        "description": "Conservative Jetson Orin Nano 8GB proxy. Applies a slowdown to GB10 measured compute and adds small sensor/memory overhead.",
        "target_device": "Jetson Orin Nano 8GB",
        "reflex_budget_ms": 50.0,
        "cognitive_budget_ms": 500.0,
        "latency_scale": 4.0,
        "sensor_overhead_ms": 3.0,
        "recommended_runtime": "onnx_or_tensorrt_fp16_int8",
    },
    "cpu_only_low_power": {
        "description": "Very conservative CPU-only low-power proxy. Intended to show whether GPU/NPU acceleration is mandatory.",
        "target_device": "low-power CPU-only edge",
        "reflex_budget_ms": 50.0,
        "cognitive_budget_ms": 500.0,
        "latency_scale": 20.0,
        "sensor_overhead_ms": 5.0,
        "recommended_runtime": "not_recommended_for_yolo_realtime",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CPE edge-style perception latency.")
    parser.add_argument("--rgb-dir", type=Path, required=True, help="Directory of ordered RGB frames.")
    parser.add_argument("--depth-dir", type=Path, default=None, help="Optional directory of matching depth maps.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="YOLO checkpoint to benchmark.")
    parser.add_argument("--source", default="sanpo", choices=["sanpo", "uasol"], help="Depth source format.")
    parser.add_argument("--fps", type=float, default=30.0, help="Original stream FPS.")
    parser.add_argument("--frame-step", type=int, default=3, help="Process every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum processed frames to benchmark.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup frames before timing.")
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON summary output path.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional per-frame timing CSV output path.")
    parser.add_argument("--preload", action="store_true", help="Preload selected RGB/depth frames before timing to simulate live sensor buffers without dataset disk I/O.")
    parser.add_argument("--edge-profile", default="gb10_native", choices=sorted(EDGE_PROFILES), help="Budget/slowdown profile for simulated edge pass-fail reporting.")
    parser.add_argument("--latency-scale", type=float, default=None, help="Override profile latency scale for measured compute stages.")
    parser.add_argument("--sensor-overhead-ms", type=float, default=None, help="Override profile per-processed-frame sensor/memory overhead.")
    parser.add_argument("--simulate-wait", action="store_true", help="Actually sleep for simulated overhead/slowdown. Otherwise simulation is reported analytically.")
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def find_depth(depth_dir: Path | None, rgb_path: Path, source: str) -> np.ndarray | None:
    if depth_dir is None:
        return None
    for ext in (".float16.gz", ".png", ".npz"):
        depth_path = depth_dir / f"{rgb_path.stem}{ext}"
        if depth_path.exists():
            return load_depth_map(depth_path, source=source)
    return None


def frame_rows_from_detections(
    frame_idx: int,
    frame: np.ndarray,
    depth_map: np.ndarray | None,
    detections: list[dict],
    depth_history: dict,
    fps: float,
    source: str,
) -> list[dict]:
    h, w = frame.shape[:2]
    detections.extend(detect_unlabeled_obstacles(depth_map, w, h, detections))
    detections.sort(key=lambda det: _rough_depth(det, depth_map))

    rows = []
    for det in detections:
        tid = det["track_id"]
        if det.get("class_name") == "unlabeled_obstacle":
            distance_m = det["min_depth"]
            tid = f"obs_{frame_idx}_{det['cx']:.0f}"
        else:
            distance_m = None
            if depth_map is not None:
                occluders = [
                    (r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"])
                    for r in rows
                    if r["distance_m"] is not None
                ]
                distance_m = median_depth_in_box(
                    depth_map,
                    det["x1"],
                    det["y1"],
                    det["x2"],
                    det["y2"],
                    exclude_boxes=occluders,
                    frame_shape=frame.shape[:2],
                )

        if distance_m is not None and det.get("class_name") != "unlabeled_obstacle":
            depth_history[tid].append((frame_idx, distance_m))

        velocity_ms = 0.0
        if det.get("class_name") != "unlabeled_obstacle":
            velocity_ms = compute_velocity(list(depth_history[tid]), fps)

        rows.append(
            {
                "frame_idx": frame_idx,
                "source": source,
                "track_id": tid,
                "class": det["class_name"],
                "confidence": det["confidence"],
                "bbox_x1": det["x1"],
                "bbox_y1": det["y1"],
                "bbox_x2": det["x2"],
                "bbox_y2": det["y2"],
                "cx_px": round(det["cx"], 1),
                "bearing_deg": round(compute_bearing(det["cx"], w), 2),
                "distance_m": round(distance_m, 2) if distance_m is not None else None,
                "velocity_ms": round(velocity_ms, 3),
            }
        )
    return rows


def summarize(name: str, values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": mean(values) if values else 0.0,
        "median_ms": median(values) if values else 0.0,
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "max_ms": max(values) if values else 0.0,
    }


def main() -> None:
    args = parse_args()
    frame_paths = sorted(
        p for p in args.rgb_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    selected = frame_paths[:: max(args.frame_step, 1)][: args.max_frames]
    if not selected:
        raise FileNotFoundError(f"No RGB frames found in {args.rgb_dir}")

    torch_device = args.device
    if args.device != "cpu" and torch.cuda.is_available():
        torch.cuda.set_device(int(args.device))
        torch_device = f"cuda:{args.device}"

    tracker = YoloTracker(model_path=str(args.weights))
    tracker.model.to(torch_device)

    preloaded = None
    if args.preload:
        preloaded = []
        for rgb_path in selected:
            frame = cv2.imread(str(rgb_path))
            depth_map = find_depth(args.depth_dir, rgb_path, args.source)
            if frame is not None:
                preloaded.append((rgb_path, frame, depth_map))
        selected = [item[0] for item in preloaded]

    warmup_frames = selected[: min(args.warmup, len(selected))]
    for rgb_path in warmup_frames:
        if preloaded is not None:
            match = next((item for item in preloaded if item[0] == rgb_path), None)
            frame = match[1] if match else None
        else:
            frame = cv2.imread(str(rgb_path))
        if frame is not None:
            tracker.track(frame, depth_map=None)
    if args.device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

    depth_history = defaultdict(lambda: deque(maxlen=VELOCITY_WINDOW + 1))
    timing_rows = []

    for ordinal, rgb_path in enumerate(selected):
        frame_idx = ordinal * args.frame_step
        t0 = perf_counter()
        if preloaded is not None:
            _, frame, depth_map = preloaded[ordinal]
            t_read = perf_counter()
            t_depth = t_read
        else:
            frame = cv2.imread(str(rgb_path))
            t_read = perf_counter()
            if frame is None:
                continue

            depth_map = find_depth(args.depth_dir, rgb_path, args.source)
            t_depth = perf_counter()

        detections = tracker.track(frame, depth_map=depth_map)
        if args.device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_track = perf_counter()

        rows = frame_rows_from_detections(
            frame_idx, frame, depth_map, detections, depth_history, args.fps, args.source
        )
        t_rows = perf_counter()

        if args.simulate_wait:
            profile = EDGE_PROFILES[args.edge_profile]
            scale = args.latency_scale if args.latency_scale is not None else profile["latency_scale"]
            overhead = args.sensor_overhead_ms if args.sensor_overhead_ms is not None else profile["sensor_overhead_ms"]
            measured_ms = (t_rows - t0) * 1000.0
            extra_ms = max(0.0, measured_ms * max(scale - 1.0, 0.0) + overhead)
            sleep(extra_ms / 1000.0)
            t_rows = perf_counter()

        timing_rows.append(
            {
                "frame_idx": frame_idx,
                "image": str(rgb_path),
                "detections": len(rows),
                "read_ms": (t_read - t0) * 1000.0,
                "depth_ms": (t_depth - t_read) * 1000.0,
                "track_ms": (t_track - t_depth) * 1000.0,
                "rows_ms": (t_rows - t_track) * 1000.0,
                "total_ms": (t_rows - t0) * 1000.0,
            }
        )

    read = [row["read_ms"] for row in timing_rows]
    depth = [row["depth_ms"] for row in timing_rows]
    track = [row["track_ms"] for row in timing_rows]
    rows_t = [row["rows_ms"] for row in timing_rows]
    total = [row["total_ms"] for row in timing_rows]
    profile = EDGE_PROFILES[args.edge_profile]
    latency_scale = args.latency_scale if args.latency_scale is not None else profile["latency_scale"]
    sensor_overhead_ms = args.sensor_overhead_ms if args.sensor_overhead_ms is not None else profile["sensor_overhead_ms"]
    simulated_total = [value * latency_scale + sensor_overhead_ms for value in total]

    processed_fps = 1000.0 / mean(total) if total else 0.0
    source_equivalent_fps = processed_fps * max(args.frame_step, 1)
    simulated_processed_fps = 1000.0 / mean(simulated_total) if simulated_total else 0.0
    simulated_source_equivalent_fps = simulated_processed_fps * max(args.frame_step, 1)
    real_time_budget_ms = (1000.0 / args.fps) * max(args.frame_step, 1)
    reflex_budget_ms = float(profile["reflex_budget_ms"])

    summary = {
        "weights": str(args.weights),
        "rgb_dir": str(args.rgb_dir),
        "depth_dir": str(args.depth_dir) if args.depth_dir else None,
        "device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "source_fps": args.fps,
        "frame_step": args.frame_step,
        "processed_frames": len(timing_rows),
        "preload": args.preload,
        "edge_profile": args.edge_profile,
        "edge_profile_config": profile,
        "latency_scale": latency_scale,
        "sensor_overhead_ms": sensor_overhead_ms,
        "simulate_wait": args.simulate_wait,
        "real_time_budget_ms_per_processed_frame": real_time_budget_ms,
        "reflex_budget_ms": reflex_budget_ms,
        "processed_fps": processed_fps,
        "source_equivalent_fps": source_equivalent_fps,
        "simulated_processed_fps": simulated_processed_fps,
        "simulated_source_equivalent_fps": simulated_source_equivalent_fps,
        "reflex_budget_pass": percentile(total, 95) < reflex_budget_ms,
        "real_time_pass": percentile(total, 95) < real_time_budget_ms,
        "simulated_reflex_budget_pass": percentile(simulated_total, 95) < reflex_budget_ms,
        "simulated_real_time_pass": percentile(simulated_total, 95) < real_time_budget_ms,
        "read_latency": summarize("read_ms", read),
        "depth_latency": summarize("depth_ms", depth),
        "track_latency": summarize("track_ms", track),
        "row_build_latency": summarize("rows_ms", rows_t),
        "total_latency": summarize("total_ms", total),
        "simulated_total_latency": summarize("simulated_total_ms", simulated_total),
    }

    print(json.dumps(summary, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(timing_rows[0].keys()))
            writer.writeheader()
            writer.writerows(timing_rows)


if __name__ == "__main__":
    main()
