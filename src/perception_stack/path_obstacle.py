"""
path_obstacle.py
================
"Obstacle ahead." without knowing what the obstacle is: anything standing in the walking corridor
that isn't flat ground, from a segmentation class map + depth. It replaces the depth sweep
(pipeline.detect_unlabeled_obstacles), which fires on the ground in every frame (methodology §4.21).

Per segmentation frame, `corridor_histogram` counts the pixels of each class that are
  - in the corridor: |lateral offset| <= CORRIDOR_HALF_W (the GRU arrival label's 0.75 m corridor),
  - below head clearance: no more than CLEARANCE_ABOVE_CAM_M above the camera (awnings, high
    branches and signs overhead are not in the way),
binned by depth. `PathObstacleAlerter` turns the count within alert_m into at most one alert per
episode, like the curb alert (src/segmentation/edge_alert.py): confirm over confirm_frames
consecutive frames, re-arm after clear_s without an obstacle.

What counts as an obstacle is a set of class ids, so the same code runs on SANPO's raw panoptic ids
(evaluation) or the walkable-surface model's CPE classes. Curbs and stairs are left to the curb
alert; vegetation is non-walkable ground in the CPE scheme, so it is not an obstacle here either.

Geometry is pinhole with the principal point at the image centre and fy = fx; the camera is
assumed to face the walking direction (a head camera turns with the head: a limitation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CORRIDOR_HALF_W = 0.375         # m; the corridor label's width 0.75 m
CLEARANCE_ABOVE_CAM_M = 0.5     # m above the camera (~2.1 m above ground for a head camera)
DEPTH_BINS = np.array([0.3, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])   # bin edges, m

# Declared defaults; tools/evaluate_path_obstacle.py chooses them on official-train sessions.
ALERT_M = 2.5
MIN_PIXELS = 300                # at 1/4 resolution (552 x 310 for SANPO's 2208 x 1242)
CONFIRM_FRAMES = 2
CLEAR_S = 2.0

# SANPO raw ids (src/segmentation/labels.py): vertical classes the detector doesn't track.
# Pedestrian 12, rider 13, animal 14 and vehicle 21 are the object engine's.
SANPO_OBSTACLE_IDS = (4, 7, 8, 9, 10, 11, 20, 22, 23, 24, 25, 26, 28)
SANPO_DETECTOR_IDS = (12, 13, 14, 21)


def corridor_histogram(class_map: np.ndarray, depth: np.ndarray, fx: float, n_classes: int = 256) -> np.ndarray:
    """[n_classes, len(DEPTH_BINS) - 1] pixel counts in the corridor, below clearance, by class and depth bin.

    class_map and depth have the same shape; fx is in that shape's pixels.
    """
    h, w = class_map.shape
    d = np.asarray(depth, np.float32)
    u = (np.arange(w, dtype=np.float32) - (w - 1) / 2) / fx          # tan of the horizontal angle
    v = (np.arange(h, dtype=np.float32) - (h - 1) / 2) / fx          # tan of the vertical angle (down +)
    ok = np.isfinite(d) & (d > DEPTH_BINS[0]) & (d <= DEPTH_BINS[-1])
    ok &= np.abs(u[None, :] * d) <= CORRIDOR_HALF_W
    ok &= -v[:, None] * d <= CLEARANCE_ABOVE_CAM_M
    b = np.searchsorted(DEPTH_BINS, d[ok], side="left") - 1         # bins are (lo, hi]
    out = np.zeros((n_classes, len(DEPTH_BINS) - 1), np.int32)
    np.add.at(out, (class_map[ok].astype(np.intp), b), 1)
    return out


def pixels_within(hist: np.ndarray, class_ids, max_m: float) -> int:
    """Obstacle pixels closer than max_m (bins whose upper edge is <= max_m)."""
    n_bins = int(np.searchsorted(DEPTH_BINS, max_m + 1e-6, side="right")) - 1
    return int(hist[list(class_ids), :max(0, n_bins)].sum())


@dataclass(frozen=True)
class ObstacleAlert:
    t: float
    pixels: int


class PathObstacleAlerter:
    def __init__(self, alert_m: float = ALERT_M, min_pixels: int = MIN_PIXELS,
                 confirm_frames: int = CONFIRM_FRAMES, clear_s: float = CLEAR_S):
        self.alert_m, self.min_pixels, self.confirm_frames, self.clear_s = alert_m, min_pixels, confirm_frames, clear_s
        self._armed = True
        self._last_seen = float("-inf")
        self._run = 0

    def step(self, t: float, pixels_within_alert_m: int) -> ObstacleAlert | None:
        """One segmentation frame at time t (s), given the obstacle pixel count within alert_m."""
        near = pixels_within_alert_m >= self.min_pixels
        if near:
            self._last_seen = t
            self._run += 1
        else:
            self._run = 0
            if t - self._last_seen >= self.clear_s:
                self._armed = True
        if self._armed and self._run >= self.confirm_frames:
            self._armed = False
            return ObstacleAlert(t, pixels_within_alert_m)
        return None
