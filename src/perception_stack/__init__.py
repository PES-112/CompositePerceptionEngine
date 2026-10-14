"""
src/perception_stack/__init__.py
================================
Public API for the CPE Perception Stack (Phase 1).

Import from here rather than from individual sub-modules to keep
downstream code stable when internal structure changes.

YoloTracker and the run_perception* entry points are loaded lazily: they pull
in ultralytics, and the runtime threat engine (which only needs physics) must
import without the detector library installed.
"""

from src.perception_stack.depth_loader      import load_depth_map, median_depth_in_box
from src.perception_stack.physics           import (
    compute_bearing, compute_velocity, kinetic_score, bearing_label,
    batch_compute_bearing, batch_kinetic_score,
    CLASS_SEVERITY,
)
from src.perception_stack.csv_writer        import CSV_FIELDS, write_csv, StreamingCSVWriter

_LAZY = {
    "YoloTracker": "src.perception_stack.yolo_tracker",
    "run_perception": "src.perception_stack.pipeline",
    "run_perception_stream": "src.perception_stack.pipeline",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # depth
    "load_depth_map", "median_depth_in_box",
    # physics
    "compute_bearing", "compute_velocity", "kinetic_score",
    "bearing_label", "CLASS_SEVERITY",
    "batch_compute_bearing", "batch_kinetic_score",
    # tracking
    "YoloTracker",
    # csv
    "CSV_FIELDS", "write_csv", "StreamingCSVWriter",
    # orchestrator
    "run_perception", "run_perception_stream",
]
