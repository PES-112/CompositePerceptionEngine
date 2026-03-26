"""
depth_loader.py
===============
CPE Perception Stack — Dataset-aware depth map loader.

Handles the two dataset sources for Phase 1:
  sanpo  — lidar ground-truth, 16-bit PNG, scale = 0.001 (mm → m)
  uasol  — stereo-rectified,   16-bit PNG, scale = 0.256
"""

from pathlib import Path

import cv2
import numpy as np

# Physical depth range to accept as valid sensor readings
MIN_DEPTH_M = 0.5    # below this = sensor noise floor
MAX_DEPTH_M = 30.0   # beyond this = not actionable for pedestrian nav

# Scale factors: raw uint16 pixel value × scale → metres
DEPTH_SCALES = {
    "sanpo": 0.001,   # stored in millimetres
    "uasol": 0.256,   # stereo disparity-derived, per UASOL dataset spec
}


def load_depth_map(depth_path: Path, source: str = "sanpo"):
    """
    Load a co-registered 16-bit depth PNG and return a float32 array in metres.
    Returns None if the file is missing (frame has no paired depth).

    Args:
        depth_path: Absolute path to the 16-bit PNG.
        source:     'sanpo' or 'uasol' — selects the correct physical scale factor.
    """
    if not depth_path.exists():
        return None
    scale = DEPTH_SCALES.get(source, DEPTH_SCALES["sanpo"])
    raw = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH).astype(np.float32)
    return raw * scale


def median_depth_in_box(
    depth_map: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
) -> float | None:
    """
    Return the median metric depth (metres) of valid pixels inside a bounding box.
    Pixels outside [MIN_DEPTH_M, MAX_DEPTH_M] are excluded as noise/out-of-range.
    Returns None if no valid pixels exist in the box.
    """
    roi = depth_map[y1:y2, x1:x2]
    valid = roi[(roi > MIN_DEPTH_M) & (roi < MAX_DEPTH_M)]
    if valid.size == 0:
        return None
    return float(np.median(valid))
