"""
labels.py
=========
SANPO's 31 panoptic classes (labelmap.json) remapped to CPE's walkable-surface
scheme (docs/architecture.md §6.3), and the mask decoding.

SANPO masks are 8-bit RGB PNGs: R = semantic class id (0 = unlabeled), G/B =
instance id. Training stores the R channel as is and remaps at load time
(the Ultralytics dataset YAML's label_mapping, written from SANPO_TO_CPE), so
the scheme can change without re-downloading.

The 17-class detector already finds the objects (people, vehicles, poles,
bollards, ...); this model is for the ground: which surface each object stands
on, where the walkable area ends, and which depth pixels are just ground.
"obstacle" is the catch-all for everything vertical so that ground classes are
not forced onto it.
"""

from __future__ import annotations

import numpy as np

IGNORE = 255

# SANPO labelmap.json ids (verified against gs://gresearch/sanpo_dataset/v0/labelmap.json).
SANPO_CLASSES = {
    0: "unlabeled", 1: "road", 2: "curb", 3: "sidewalk", 4: "guard rail/road barrier", 5: "crosswalk",
    6: "paved trail", 7: "building", 8: "wall/fence", 9: "hand rail", 10: "opening-door",
    11: "opening-gate", 12: "pedestrian", 13: "rider", 14: "animal", 15: "stairs", 16: "water body",
    17: "other walkable surface", 18: "inaccessible surface", 19: "railway track", 20: "obstacle",
    21: "vehicle", 22: "traffic sign", 23: "traffic light", 24: "pole", 25: "bus stop", 26: "bike rack",
    27: "sky", 28: "tree", 29: "vegetation", 30: "terrain",
}

CPE_CLASSES = {
    0: "walkable",          # sidewalk, paved trail, other walkable surface
    1: "crosswalk",
    2: "road",              # road, railway track
    3: "curb",
    4: "stairs",
    5: "non-walkable ground",   # inaccessible surface, water, terrain (grass), vegetation
    6: "obstacle",          # everything vertical: buildings, walls, poles, vehicles, people, ...
    7: "sky",
}
WALKABLE, CROSSWALK, ROAD, CURB, STAIRS, NON_WALKABLE, OBSTACLE, SKY = range(8)

SANPO_TO_CPE = {
    0: IGNORE,
    3: WALKABLE, 6: WALKABLE, 17: WALKABLE,
    5: CROSSWALK,
    1: ROAD, 19: ROAD,
    2: CURB,
    15: STAIRS,
    16: NON_WALKABLE, 18: NON_WALKABLE, 29: NON_WALKABLE, 30: NON_WALKABLE,
    27: SKY,
    **{i: OBSTACLE for i in (4, 7, 8, 9, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 25, 26, 28)},
}
assert sorted(SANPO_TO_CPE) == list(range(31))

# Safe to walk on. Crosswalk counts: it is where the walker is meant to cross.
WALKABLE_CLASSES = frozenset({WALKABLE, CROSSWALK})
# Ground the walker should not step onto unwarned (a false-safe prediction is one of these called walkable).
UNSAFE_GROUND = frozenset({ROAD, CURB, STAIRS, NON_WALKABLE})
GROUND_CLASSES = WALKABLE_CLASSES | UNSAFE_GROUND
# Classes whose boundary drives the edge alert; scored with boundary F.
EDGE_CLASSES = (CURB, STAIRS)

_LUT = np.full(256, IGNORE, dtype=np.uint8)
for _src, _dst in SANPO_TO_CPE.items():
    _LUT[_src] = _dst


def remap(sanpo_ids: np.ndarray) -> np.ndarray:
    """SANPO class ids (any integer array) -> CPE class ids, uint8; unknown ids -> IGNORE."""
    return _LUT[np.asarray(sanpo_ids).astype(np.uint8)]


def read_sanpo_ids(path_or_bytes) -> np.ndarray:
    """SANPO mask PNG (path or bytes) -> its raw class-id channel (H, W) uint8. cv2 reads BGR, so R is channel 2."""
    import cv2

    if isinstance(path_or_bytes, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(path_or_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imread(str(path_or_bytes), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("unreadable mask")
    return img if img.ndim == 2 else img[..., 2]


def read_sanpo_mask(path_or_bytes) -> np.ndarray:
    """SANPO mask PNG -> CPE class map (H, W) uint8."""
    return remap(read_sanpo_ids(path_or_bytes))


def ultralytics_label_mapping() -> dict[int, int | str]:
    """SANPO_TO_CPE in the Ultralytics dataset-YAML form (ignore -> 'ignore_label')."""
    return {k: ("ignore_label" if v == IGNORE else int(v)) for k, v in SANPO_TO_CPE.items()}
