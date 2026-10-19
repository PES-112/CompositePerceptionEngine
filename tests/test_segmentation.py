import unittest

import numpy as np

from evaluation.segmentation_metrics import SegmentationScore
from src.segmentation.context import EdgeAhead, ObjectRef, approach_speed, flat_ground, scene_frame, walker_surface
from src.segmentation.edge_alert import EdgeAlerter
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


class ApproachTests(unittest.TestCase):
    def test_closing_on_the_centre_line_is_positive(self):
        d = np.full(4, 5.0)
        b = np.degrees(np.arcsin([1.6, 1.4, 1.2, 1.0] / d))   # lateral 1.6 -> 1.0 m over 0.6 s
        self.assertAlmostEqual(approach_speed(d, b), 1.0)
        self.assertAlmostEqual(approach_speed(d, -b), 1.0)     # same from the left
        self.assertAlmostEqual(approach_speed(d, b[::-1]), -1.0)

    def test_gaps_use_the_real_time_step(self):
        d = np.array([5.0, np.nan, np.nan, 5.0])
        b = np.degrees(np.arcsin(np.array([1.6, 0, 0, 1.0]) / 5.0))
        self.assertAlmostEqual(approach_speed(d, b), 1.0)
        self.assertEqual(approach_speed(np.array([np.nan, 5.0]), np.array([np.nan, 10.0])), 0.0)


class EdgeAlertTests(unittest.TestCase):
    def test_confirmed_once_per_episode(self):
        a = EdgeAlerter(confirm_frames=2, clear_s=2.0)
        self.assertIsNone(a.step(0.0, EdgeAhead(CURB, 4.0)))
        alert = a.step(0.4, EdgeAhead(CURB, 3.6))
        self.assertEqual((alert.kind, alert.distance_m), ("curb", 3.6))
        self.assertIsNone(a.step(0.8, EdgeAhead(CURB, 3.2)))
        self.assertIsNone(a.step(1.2, EdgeAhead(ROAD, 2.8)))     # same episode: curb then road
        self.assertIsNone(a.step(1.6, EdgeAhead(ROAD, 2.4)))

    def test_far_edges_and_unannounced_kinds_are_silent(self):
        a = EdgeAlerter(confirm_frames=2)
        for t in (0.0, 0.4, 0.8):
            self.assertIsNone(a.step(t, EdgeAhead(CURB, 8.0)))
        b = EdgeAlerter(confirm_frames=2)
        for t in (0.0, 0.4, 0.8):
            self.assertIsNone(b.step(t, EdgeAhead(SANPO_TO_CPE[16], 2.0)))   # non-walkable ground

    def test_flicker_is_not_confirmed(self):
        a = EdgeAlerter(confirm_frames=2)
        for i, edge in enumerate([EdgeAhead(CURB, 3.0), None, EdgeAhead(STAIRS, 3.0), EdgeAhead(CURB, 3.0)]):
            self.assertIsNone(a.step(0.4 * i, edge))

    def test_rearms_after_clear(self):
        a = EdgeAlerter(confirm_frames=2, clear_s=2.0)
        a.step(0.0, EdgeAhead(CURB, 3.0))
        self.assertIsNotNone(a.step(0.4, EdgeAhead(CURB, 3.0)))
        self.assertIsNone(a.step(1.0, None))
        self.assertIsNone(a.step(2.0, EdgeAhead(CURB, 3.0)))     # 1.6 s gap: same episode
        self.assertIsNone(a.step(4.4, None))                     # 2.4 s clear: re-armed
        a.step(4.8, EdgeAhead(STAIRS, 4.0))
        self.assertEqual(a.step(5.2, EdgeAhead(STAIRS, 3.5)).kind, "stairs")


class ColumnTests(unittest.TestCase):
    def test_bearing_to_image_column(self):
        from tools.build_scene_context import column_frac
        fx, width = 1380.0, 2208.0
        edge = np.degrees(np.arctan(width / 2 / fx))
        self.assertAlmostEqual(column_frac(0.0, fx, width, legacy=False), 0.5)
        self.assertAlmostEqual(column_frac(edge, fx, width, legacy=False), 1.0)
        self.assertAlmostEqual(column_frac(-edge, fx, width, legacy=False), 0.0)
        self.assertAlmostEqual(column_frac(35.0, fx, width, legacy=True), 1.0)   # legacy: linear onto ±35°


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


class EdgeAlertScoringTests(unittest.TestCase):
    def test_step_offs_can_come_from_a_reference_track(self):
        from tools.evaluate_edge_alerts import score_session

        # Model track: sees a curb 3 m ahead at frames 12-18, but its walker never leaves walkable ground.
        model = [[f, WALKABLE, CURB if 12 <= f <= 18 else None, 3.0 if 12 <= f <= 18 else None]
                 for f in range(0, 60, 6)]
        # Reference (masks): the walker steps onto road at frame 36.
        gt = [[f, ROAD if f >= 36 else WALKABLE, None, None] for f in range(0, 60, 6)]

        self_ref = score_session(model)
        self.assertEqual(len(self_ref["alerts"]), 1)
        self.assertEqual(self_ref["step_offs"], [])

        vs_gt = score_session(model, ref_rows=gt)
        self.assertEqual(vs_gt["alerts"], self_ref["alerts"])
        self.assertEqual(vs_gt["step_offs"], [[2.4, "road"]])
        self.assertEqual(vs_gt["followed"], [True])
        self.assertAlmostEqual(vs_gt["leads"][0], 2.4 - 18 / 15)
