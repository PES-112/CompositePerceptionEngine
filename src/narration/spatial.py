"""
spatial.py
==========
Stereo placement of an alert clip by the object's bearing (pending_work.md P4.3): the phrase
already says "left" or "right"; panning the voice the same way lets the direction be heard
before the word arrives, which is how obstacle sonification for blind walkers maps horizontal
position (stereo panning, e.g. WatchOut). Constant-power panning keeps loudness the same at
every angle, so a hazard to the side is not quieter than one ahead.

Bearings beyond ±PAN_FULL_DEG are panned fully to that side; ground phrases (edges, surface) and
anything straight ahead stay centred. Positive bearing = right, as in the threat engine.
"""

from __future__ import annotations

import math

import numpy as np

PAN_FULL_DEG = 60.0


def pan_gains(bearing_deg: float, full_deg: float = PAN_FULL_DEG) -> tuple[float, float]:
    """(left, right) constant-power gains: (√½, √½) ahead, (1, 0) fully left."""
    pan = max(-1.0, min(1.0, bearing_deg / full_deg))
    angle = (pan + 1.0) * math.pi / 4
    return math.cos(angle), math.sin(angle)


def pan_stereo(mono: np.ndarray, bearing_deg: float) -> np.ndarray:
    """int16 mono clip -> int16 stereo [n, 2] placed at bearing_deg."""
    left, right = pan_gains(bearing_deg)
    x = mono.astype(np.float32)
    return np.clip(np.stack([x * left, x * right], axis=1), -32768, 32767).astype(np.int16)
