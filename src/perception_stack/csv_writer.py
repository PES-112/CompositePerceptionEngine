"""
csv_writer.py
=============
CPE Perception Stack — CSV schema definition and writer for Stage 1 output.

Defines the canonical column layout so Stage 1 (pipeline.py) and Stage 2
(fact_sheet_builder.py) share the exact same field names without duplication.
"""

import csv
from pathlib import Path

# ── Canonical CSV schema ─────────────────────────────────────────────────────
# This is the contract between Stage 1 (perception) and Stage 2 (fact sheets).
CSV_FIELDS = [
    "frame_idx",    # int    — frame number in source video/sequence
    "source",       # str    — 'sanpo' or 'uasol'
    "track_id",     # int    — persistent ByteTrack object ID
    "class",        # str    — COCO class label (e.g. 'car', 'person')
    "confidence",   # float  — YOLO detection confidence [0, 1]
    "bbox_x1",      # int    — bounding box left
    "bbox_y1",      # int    — bounding box top
    "bbox_x2",      # int    — bounding box right
    "bbox_y2",      # int    — bounding box bottom
    "cx_px",        # float  — horizontal centre pixel
    "bearing_deg",  # float  — left/right bearing from ego path (negative=left)
    "distance_m",   # float  — median metric depth inside ROI (empty if no depth map)
    "velocity_ms",  # float  — estimated closing velocity m/s (positive = approaching)
]


def write_csv(rows: list[dict], out_path: Path) -> None:
    """
    Write a list of perception row dicts to a CSV file using the canonical schema.

    Args:
        rows:     List of dicts — each must contain keys matching CSV_FIELDS.
        out_path: Destination file path. Parent directories are created if missing.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
