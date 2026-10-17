"""
context.py
==========
Per-frame scene context from a walkable-surface class map (§6.3), in the
form the threat engine consumes (priority.SceneContext), plus the per-scene
outputs.

Where an object stands. The ground under an object is found without its box
height: in the object's image columns, take the ground-class pixels whose
depth is close to the object's own depth (the Stage-1 box-median depth). This
works from a row's bearing and distance alone, which is all the GRU dataset
keeps, and from a box when there is one.

Definitions (declared, not fitted; the thresholds are module constants):
  walker surface   majority ground class in the bottom-centre of the image
  surface          majority ground class under the object (None: no ground visible there)
  in_path          object on walkable ground while the walker is too: it shares the walker's surface
  near_boundary    the object's ground mixes walkable and unsafe ground (or curb): it is at an edge
  other_surface    walker on walkable ground, object on road / non-walkable ground
  edge ahead       nearest row up the centre of the image where walkable ground gives way to
                   unsafe ground; its class and median depth (m)

The class map, the depth map and the image may have different resolutions;
everything is addressed in fractions of width / height.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.segmentation.labels import (CURB, GROUND_CLASSES, NON_WALKABLE, ROAD, STAIRS, UNSAFE_GROUND,
                                     WALKABLE_CLASSES)
from src.threat_engine.priority import SceneContext

WALKER_ROWS = (0.8, 1.0)        # bottom fifth of the image ...
WALKER_COLS = (0.3, 0.7)        # ... middle 40%: the ground just ahead of the walker
EDGE_COLS = (0.4, 0.6)          # band scanned for the edge ahead
DEPTH_TOL_FRAC = 0.15           # ground within ±15% (at least ±0.3 m) of the object's depth
DEPTH_TOL_MIN_M = 0.3
MIN_GROUND_PX = 5
BOUNDARY_SHARE = 0.15           # both walkable and unsafe ground at least this share -> near a boundary
CURB_SHARE = 0.10
MAX_EDGE_M = 10.0
GROUND = np.array(sorted(GROUND_CLASSES), dtype=np.uint8)


@dataclass(frozen=True)
class ObjectRef:
    track_id: str
    cx_frac: float              # box centre column / image width
    distance_m: float
    half_width_frac: float = 0.02


@dataclass(frozen=True)
class EdgeAhead:
    kind: int                   # CPE class id (curb, stairs, road, non-walkable ground)
    distance_m: float


@dataclass
class SceneFrame:
    walker_surface: int | None
    edge: EdgeAhead | None
    surfaces: dict[str, int | None] = field(default_factory=dict)
    contexts: dict[str, SceneContext] = field(default_factory=dict)


def _majority(classes: np.ndarray) -> int | None:
    if classes.size < MIN_GROUND_PX:
        return None
    return int(np.bincount(classes, minlength=256).argmax())


def _depth_on(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """depth resampled (nearest) onto a class map's grid."""
    if depth.shape == shape:
        return depth
    rows = (np.arange(shape[0]) * depth.shape[0] // shape[0]).clip(0, depth.shape[0] - 1)
    cols = (np.arange(shape[1]) * depth.shape[1] // shape[1]).clip(0, depth.shape[1] - 1)
    return depth[np.ix_(rows, cols)]


def walker_surface(class_map: np.ndarray) -> int | None:
    h, w = class_map.shape
    patch = class_map[int(WALKER_ROWS[0] * h):int(WALKER_ROWS[1] * h), int(WALKER_COLS[0] * w):int(WALKER_COLS[1] * w)]
    ground = patch[np.isin(patch, GROUND)]
    return _majority(ground) if ground.size >= 0.05 * patch.size else None


def object_ground(class_map: np.ndarray, depth: np.ndarray, obj: ObjectRef, widen: float = 1.0) -> np.ndarray:
    """Ground-class pixels in the object's columns at about the object's depth."""
    h, w = class_map.shape
    hw = max(1, int(round(obj.half_width_frac * widen * w)))
    c = int(round(obj.cx_frac * w))
    lo, hi = max(0, c - hw), min(w, c + hw + 1)
    if lo >= hi:
        return np.zeros(0, np.uint8)
    cls, d = class_map[:, lo:hi], depth[:, lo:hi]
    tol = max(DEPTH_TOL_MIN_M, DEPTH_TOL_FRAC * obj.distance_m)
    keep = np.isin(cls, GROUND) & np.isfinite(d) & (np.abs(d - obj.distance_m) <= tol)
    return cls[keep]


def edge_ahead(class_map: np.ndarray, depth: np.ndarray, walker: int | None) -> EdgeAhead | None:
    """Scan up the centre band from the bottom; the first unsafe-ground row after walkable ground."""
    if walker not in WALKABLE_CLASSES:
        return None
    h, w = class_map.shape
    band = slice(int(EDGE_COLS[0] * w), int(EDGE_COLS[1] * w))
    for r in range(h - 1, -1, -1):
        cls, d = class_map[r, band], depth[r, band]
        ground = np.isin(cls, GROUND)
        if ground.sum() < MIN_GROUND_PX:
            continue
        kind = _majority(cls[ground])
        if kind in UNSAFE_GROUND:
            dist = d[ground & (cls == kind) & np.isfinite(d)]
            if dist.size and float(np.median(dist)) <= MAX_EDGE_M:
                return EdgeAhead(kind, float(np.median(dist)))
            return None
        if np.nanmedian(d[ground]) > MAX_EDGE_M:
            return None
    return None


def scene_frame(class_map: np.ndarray, depth: np.ndarray, objects: list[ObjectRef]) -> SceneFrame:
    depth = _depth_on(np.asarray(depth, np.float32), class_map.shape)
    walker = walker_surface(class_map)
    out = SceneFrame(walker, edge_ahead(class_map, depth, walker))
    walker_ok = walker in WALKABLE_CLASSES
    for obj in objects:
        ground = object_ground(class_map, depth, obj)
        surface = _majority(ground)
        wide = object_ground(class_map, depth, obj, widen=2.0)
        n = max(1, wide.size)
        safe_share = np.isin(wide, list(WALKABLE_CLASSES)).sum() / n
        unsafe_share = np.isin(wide, list(UNSAFE_GROUND)).sum() / n
        near = wide.size >= MIN_GROUND_PX and (
            (wide == CURB).sum() / n >= CURB_SHARE
            or (safe_share >= BOUNDARY_SHARE and unsafe_share >= BOUNDARY_SHARE))
        out.surfaces[obj.track_id] = surface
        out.contexts[obj.track_id] = SceneContext(
            in_path=walker_ok and surface in WALKABLE_CLASSES,
            near_boundary=bool(near),
            other_surface=walker_ok and surface in (ROAD, NON_WALKABLE),
        )
    return out


def flat_ground(class_map: np.ndarray) -> np.ndarray:
    """Pixels the depth obstacle sweep (§6.1) should ignore: flat ground. Curbs and stairs stay in."""
    return np.isin(class_map, np.array(sorted(GROUND_CLASSES - {CURB, STAIRS}), dtype=np.uint8))

