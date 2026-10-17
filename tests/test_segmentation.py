import unittest

import numpy as np

from evaluation.segmentation_metrics import SegmentationScore
from src.segmentation.context import ObjectRef, flat_ground, scene_frame, walker_surface
from src.segmentation.labels import (CROSSWALK, CURB, IGNORE, OBSTACLE, ROAD, SANPO_TO_CPE, SKY, STAIRS, WALKABLE,
                                     remap, ultralytics_label_mapping)
from src.threat_engine.priority import SceneContext, context_multiplier

H, W = 100, 200


def street() -> tuple[np.ndarray, np.ndarray]:
    """Sidewalk on the left 60% of the ground, road on the right; sky above row 20. Depth grows up the image."""
    cls = np.full((H, W), WALKABLE, np.uint8)
    cls[:, 120:] = ROAD
    cls[:20] = SKY
    depth = np.repeat((1.0 + (H - np.arange(H)) * 0.1)[:, None], W, axis=1).astype(np.float32)
    return cls, depth


def depth_at(row: int) -> float:
    return 1.0 + (H - row) * 0.1


class LabelTests(unittest.TestCase):
    def test_every_sanpo_class_is_mapped(self):
        self.assertEqual(sorted(SANPO_TO_CPE), list(range(31)))
        np.testing.assert_array_equal(remap(np.array([0, 1, 2, 3, 5, 15, 21, 27, 200])),
                                      [IGNORE, ROAD, CURB, WALKABLE, CROSSWALK, STAIRS, OBSTACLE, SKY, IGNORE])

    def test_ultralytics_mapping_uses_ignore_label(self):
        m = ultralytics_label_mapping()
        self.assertEqual(m[0], "ignore_label")
        self.assertEqual(m[3], WALKABLE)


class ContextTests(unittest.TestCase):
    def test_walker_on_sidewalk(self):
        cls, _ = street()
        self.assertEqual(walker_surface(cls), WALKABLE)

    def test_object_surfaces_and_context(self):
        cls, depth = street()
        d = depth_at(60)
        objs = [ObjectRef("person", 0.3, d), ObjectRef("car", 0.85, d), ObjectRef("edge", 0.6, d, 0.03)]
        f = scene_frame(cls, depth, objs)
        self.assertEqual(f.surfaces["person"], WALKABLE)
        self.assertEqual(f.surfaces["car"], ROAD)
        self.assertEqual(f.contexts["person"], SceneContext(in_path=True, near_boundary=False, other_surface=False))
        self.assertEqual(f.contexts["car"], SceneContext(in_path=False, near_boundary=False, other_surface=True))
        self.assertTrue(f.contexts["edge"].near_boundary)
        self.assertLess(context_multiplier(f.contexts["car"]), 1.0)

    def test_object_with_no_ground_at_its_depth(self):
        cls, depth = street()
        f = scene_frame(cls, depth, [ObjectRef("far", 0.3, 50.0)])
        self.assertIsNone(f.surfaces["far"])
        self.assertEqual(f.contexts["far"], SceneContext())

    def test_edge_ahead_finds_curb(self):
        cls, depth = street()
        cls[40:43, :120] = CURB
        f = scene_frame(cls, depth, [])
        self.assertEqual(f.edge.kind, CURB)
        self.assertAlmostEqual(f.edge.distance_m, depth_at(41), delta=0.15)

    def test_no_edge_when_walker_on_road(self):
        cls, depth = street()
        cls[20:] = ROAD
        self.assertIsNone(scene_frame(cls, depth, []).edge)

    def test_depth_at_another_resolution(self):
        cls, depth = street()
        small = depth[::2, ::4]
        f = scene_frame(cls, small, [ObjectRef("car", 0.85, depth_at(60))])
        self.assertEqual(f.surfaces["car"], ROAD)

    def test_flat_ground_keeps_curbs_and_stairs(self):
        cls = np.array([[WALKABLE, ROAD, CURB, STAIRS, OBSTACLE, SKY]], np.uint8)
        np.testing.assert_array_equal(flat_ground(cls), [[True, True, False, False, False, False]])


class MetricTests(unittest.TestCase):
    def test_perfect_prediction(self):
        cls, _ = street()
        cls[40:43, :120] = CURB
        s = SegmentationScore()
        s.add(cls, cls)
        r = s.result()
        self.assertAlmostEqual(r["miou"], 1.0)
        self.assertEqual(r["false_safe_rate"], 0.0)
        self.assertAlmostEqual(r["boundary_f"]["curb"], 1.0)

    def test_road_called_sidewalk_is_false_safe(self):
        truth, _ = street()
        pred = truth.copy()
        pred[pred == ROAD] = WALKABLE
        s = SegmentationScore()
        s.add(truth, pred)
        r = s.result()
        self.assertAlmostEqual(r["false_safe_rate"], 1.0)
        self.assertAlmostEqual(r["road_as_walkable_rate"], 1.0)
        self.assertEqual(r["walkable_as_unsafe_rate"], 0.0)
        self.assertEqual(r["iou"]["road"], 0.0)

    def test_ignore_pixels_are_not_scored(self):
        truth, _ = street()
        truth[:50] = IGNORE
        pred = truth.copy()
        pred[:50] = ROAD
        s = SegmentationScore()
        s.add(truth, pred)
        self.assertEqual(s.result()["pixels"], 50 * W)


if __name__ == "__main__":
    unittest.main()
