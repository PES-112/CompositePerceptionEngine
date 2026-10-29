"""
Stairs from the walker's own motion: "Going up stairs.", "Going down stairs.", "End of stairs."

The segmenter sees the ground 2-3 m ahead, and it is weakest on stairs (IoU 0.47, boundary F 0.06,
methodology §4.19). Once the walker is on a flight, though, the camera pose says so directly: the
head or chest rises or falls about one riser (0.15-0.18 m) per step, at a slope of ~0.5-0.9
(rise / run). Ramps are at most 1:12 (0.083) and steep streets rarely exceed 0.2. Flat walking stays
within a few centimetres. ARKit's gravity-aligned world tracking gives the height on the phone;
SANPO's fixed_camera_poses.csv (Z up) gives it in the replay.

Rule:
- enter (ENTER_FRAMES in a row), over any of ENTER_WINDOWS (seconds, minimum rise): |rise| >= the
  minimum, |rise| >= ENTER_SLOPE * run, run >= MIN_RUN_M (MIN_RUN_M keeps a lift or a tracking jump
  on the spot from counting). The 1 s window answers quickly at a normal pace; 0.25 m over 2 s
  still catches a cautious one step per second (0.17 m/s);
- exit, "End of stairs." (EXIT_FRAMES in a row), over the last EXIT_WINDOW_S: |rise| < EXIT_RISE_M
  while still walking (run >= EXIT_RUN_M). Standing still on a flight keeps the state.

Checked on SANPO walks with labelled stairs (tools/evaluate_stair_motion.py, methodology §4.25).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

# Windows, rises and slope chosen on SANPO train+val walks (tools/evaluate_stair_motion.py, methodology §4.25).
ENTER_WINDOWS = ((1.0, 0.18), (2.0, 0.25))
WINDOW_S = max(w for w, _ in ENTER_WINDOWS)
ENTER_SLOPE = 0.5
MIN_RUN_M = 0.10
ENTER_FRAMES = 2
EXIT_WINDOW_S = 1.0
EXIT_RISE_M = 0.07
EXIT_RUN_M = 0.25
EXIT_FRAMES = 2
WARMUP_S = 1.0                  # tracking settles; SANPO's first poses jump

UP, DOWN, END = "stairs_up", "stairs_down", "stairs_end"


@dataclass(frozen=True)
class StairEvent:
    t: float
    kind: str                   # UP, DOWN or END
    rise_m: float               # over the window, signed (up positive)


class StairMotion:
    """Feed one pose per processed frame; returns a StairEvent when the state changes."""

    def __init__(self) -> None:
        self._hist: deque[tuple[float, float, float, float]] = deque()   # (t, x, y, height)
        self._t0: float | None = None
        self.direction: str | None = None       # UP / DOWN while on stairs
        self._enter_run = 0
        self._enter_dir: str | None = None
        self._exit_run = 0

    @property
    def on_stairs(self) -> bool:
        return self.direction is not None

    def step(self, t: float, x: float, y: float, height: float) -> StairEvent | None:
        if self._t0 is None:
            self._t0 = t
        self._hist.append((t, x, y, height))
        while len(self._hist) > 1 and t - self._hist[1][0] >= WINDOW_S:
            self._hist.popleft()
        t0 = self._hist[0][0]
        if t - self._t0 < WARMUP_S or t - t0 < WINDOW_S - 1e-6:
            return None

        if self.direction is None:
            d, rise = None, 0.0
            for window, min_rise in ENTER_WINDOWS:
                _, xs, ys, hs = self._since(t, window)
                r, run = height - hs, math.hypot(x - xs, y - ys)
                if abs(r) >= min_rise and abs(r) >= ENTER_SLOPE * run and run >= MIN_RUN_M:
                    d, rise = (UP if r > 0 else DOWN), r
                    break
            self._enter_run = self._enter_run + 1 if d is not None and d == self._enter_dir else int(d is not None)
            self._enter_dir = d
            if self._enter_run >= ENTER_FRAMES:
                self.direction, self._enter_run, self._exit_run = d, 0, 0
                return StairEvent(t, d, rise)
            return None

        _, xe, ye, he = self._since(t, EXIT_WINDOW_S)
        flat = abs(height - he) < EXIT_RISE_M and math.hypot(x - xe, y - ye) >= EXIT_RUN_M
        self._exit_run = self._exit_run + 1 if flat else 0
        if self._exit_run >= EXIT_FRAMES:
            self.direction, self._exit_run, self._enter_dir = None, 0, None
            return StairEvent(t, END, height - he)
        return None

    def _since(self, t: float, window: float) -> tuple[float, float, float, float]:
        """The oldest kept pose no more than `window` s old."""
        return next(s for s in self._hist if t - s[0] <= window + 1e-6)
