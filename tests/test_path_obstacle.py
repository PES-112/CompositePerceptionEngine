import numpy as np

from src.perception_stack.path_obstacle import (DEPTH_BINS, PathObstacleAlerter, corridor_histogram,
                                                pixels_within)

FX = 100.0
H, W = 101, 201                 # odd sizes put the principal point on a pixel


def scene(depth_m: float, cls: int = 24, cols=slice(98, 103), rows=slice(40, 61)):
    class_map = np.full((H, W), 3, np.uint8)        # sidewalk
    depth = np.full((H, W), 50.0, np.float32)       # far: out of range
    class_map[rows, cols] = cls
    depth[rows, cols] = depth_m
    return class_map, depth


def test_object_straight_ahead_is_counted_in_its_depth_bin():
    h = corridor_histogram(*scene(2.0), FX, 31)
    assert h[24].sum() == 5 * 21
    assert pixels_within(h, [24], 2.0) == 5 * 21
    assert pixels_within(h, [24], 1.5) == 0


def test_object_beside_the_corridor_is_ignored():
    # 30 px off centre at 2 m is 0.6 m to the side: outside the 0.375 m half-width
    h = corridor_histogram(*scene(2.0, cols=slice(130, 135)), FX, 31)
    assert h[24].sum() == 0


def test_overhead_pixels_are_ignored():
    # rows 0-10 are ~45 px above centre: 0.9 m above the camera at 2 m
    h = corridor_histogram(*scene(2.0, rows=slice(0, 10)), FX, 31)
    assert h[24].sum() == 0


def test_depth_at_the_last_edge_is_in_range():
    h = corridor_histogram(*scene(float(DEPTH_BINS[-1])), FX, 31)
    assert h[24, -1] == 5 * 21


def test_alerter_confirms_then_speaks_once_per_episode():
    a = PathObstacleAlerter(alert_m=2.5, min_pixels=10, confirm_frames=2, clear_s=2.0)
    assert a.step(0.0, 50) is None              # first frame: not confirmed yet
    assert a.step(0.4, 50) is not None          # confirmed
    assert a.step(0.8, 50) is None              # same episode
    assert a.step(1.2, 0) is None
    assert a.step(2.0, 50) is None              # back within clear_s: still the same episode
    for t in (2.4, 2.8, 3.2, 3.6, 4.0, 4.4):
        a.step(t, 0)
    assert a.step(4.8, 50) is None
    assert a.step(5.2, 50) is not None          # new episode after clear_s
