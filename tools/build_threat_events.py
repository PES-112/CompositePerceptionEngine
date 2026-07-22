"""
Convert Stage 1 perception CSV output into routable ThreatEvent JSONL.

Input rows are produced by tools/run_perception.py or the perception pipeline.
The output is the bridge into Reflex Layer, Cognitive Layer, and Physics
Verification work.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.threat_prioritizer import (
    DEFAULT_HIGH_K_THRESHOLD,
    DEFAULT_LOW_K_THRESHOLD,
    DEFAULT_NEAR_STATIC_DISTANCE_M,
    DEFAULT_REFLEX_TTC_S,
    build_threat_frames_from_csv,
    write_threat_events_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ThreatEvent JSONL from perception CSV.")
    parser.add_argument("--csv", type=Path, required=True, help="Stage 1 perception CSV path.")
    parser.add_argument("--out", type=Path, required=True, help="Output ThreatEvent JSONL path.")
    parser.add_argument("--low-k", type=float, default=DEFAULT_LOW_K_THRESHOLD)
    parser.add_argument("--high-k", type=float, default=DEFAULT_HIGH_K_THRESHOLD)
    parser.add_argument("--reflex-ttc", type=float, default=DEFAULT_REFLEX_TTC_S)
    parser.add_argument("--near-static-distance", type=float, default=DEFAULT_NEAR_STATIC_DISTANCE_M)
    parser.add_argument("--summary", type=Path, default=None, help="Optional JSON summary path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = build_threat_frames_from_csv(
        args.csv,
        low_k_threshold=args.low_k,
        high_k_threshold=args.high_k,
        reflex_ttc_s=args.reflex_ttc,
        near_static_distance_m=args.near_static_distance,
    )
    written = write_threat_events_jsonl(frames, args.out)

    route_counts = Counter(event.route for frame in frames for event in frame.events)
    class_counts = Counter(
        event.class_name
        for frame in frames
        for event in frame.events
        if event.route != "ignore"
    )
    summary = {
        "input_csv": str(args.csv),
        "output_jsonl": str(args.out),
        "frames": len(frames),
        "objects": sum(len(frame.objects) for frame in frames),
        "events_written": written,
        "route_counts": dict(route_counts),
        "class_counts_non_ignore": dict(class_counts),
        "thresholds": {
            "low_k": args.low_k,
            "high_k": args.high_k,
            "reflex_ttc_s": args.reflex_ttc,
            "near_static_distance_m": args.near_static_distance,
        },
    }

    print(json.dumps(summary, indent=2))
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
