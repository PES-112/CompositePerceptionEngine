"""
src/perception_stack/__init__.py
================================
Public API for the CPE Perception Stack (Phase 1).

Import from here rather than from individual sub-modules to keep
downstream code stable when internal structure changes.
"""

from src.perception_stack.depth_loader      import load_depth_map, median_depth_in_box
from src.perception_stack.physics           import (
    compute_bearing, compute_velocity, kinetic_score, bearing_label,
    CLASS_SEVERITY,
)
from src.perception_stack.yolo_tracker      import YoloTracker
from src.perception_stack.csv_writer        import CSV_FIELDS, write_csv
from src.perception_stack.fact_sheet_builder import (
    load_perception_csv, build_fact_sheets,
)
from src.perception_stack.pipeline          import run_perception

__all__ = [
    # depth
    "load_depth_map", "median_depth_in_box",
    # physics
    "compute_bearing", "compute_velocity", "kinetic_score",
    "bearing_label", "CLASS_SEVERITY",
    # tracking
    "YoloTracker",
    # csv
    "CSV_FIELDS", "write_csv",
    # fact sheets
    "load_perception_csv", "build_fact_sheets",
    # orchestrator
    "run_perception",
]
