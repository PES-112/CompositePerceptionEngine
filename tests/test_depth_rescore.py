"""depth_rescore samples depth at the box centre scaled to the depth map, and checks size with the given fx."""

import unittest

import numpy as np

try:
    from src.perception_stack.yolo_tracker import depth_rescore
except ImportError:          # ultralytics not installed
    depth_rescore = None

FRAME = (1242, 2208)         # SANPO head camera RGB (h, w)
FX_HEAD = 1376.0             # ZED-M


def person(cx: float, cy: float, height_px: float) -> dict:
    return {"class_name": "person", "confidence": 0.9, "cx": cx,
            "x1": cx - 40, "x2": cx + 40, "y1": cy - height_px / 2, "y2": cy + height_px / 2}


@unittest.skipIf(depth_rescore is None, "ultralytics not installed")
class DepthRescoreTests(unittest.TestCase):
    def test_samples_depth_in_depth_map_coordinates(self):
        depth = np.full((720, 1280), 20.0, dtype=np.float32)
        depth[:, :640] = 2.0                     # left half of the scene is 2 m away
        # A person 2 m away, 1.7 m tall, at frame x 1000 of 2208: depth-map x 580, the near half.
        # Read unscaled, x 1000 lands in the far half (20 m), the box looks 10x too tall and is dropped.
        h = 1.7 * FX_HEAD / 2.0
        kept = depth_rescore([person(1000, 621, h)], depth, FRAME, FX_HEAD)
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(kept[0]["confidence"], 0.9)   # 2 m: no implausible-depth penalty

    def test_size_check_uses_the_given_focal_length(self):
        depth = np.full((720, 1280), 2.0, dtype=np.float32)
        h = 1.7 * FX_HEAD / 2.0 / 4.5             # 4.5x smaller than a person 2 m away should look
        self.assertEqual(depth_rescore([person(1104, 621, h)], depth, FRAME, FX_HEAD), [])
        self.assertEqual(len(depth_rescore([person(1104, 621, h * 2)], depth, FRAME, FX_HEAD)), 1)


if __name__ == "__main__":
    unittest.main()
