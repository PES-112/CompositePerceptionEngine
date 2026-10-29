"""
Fixes after the first phone walk (2026-10-27): stairs from the walker's height, the two-wheeler width
check, and the stair phrases.

    python -m pytest tests/test_field_fixes.py -q
"""

from __future__ import annotations

import unittest

from src.narration.templates import edge_phrase, ground_phrases
from src.perception_stack.yolo_tracker import vehicle_class
from src.threat_engine.stairs import DOWN, END, UP, StairMotion


def walk(det: StairMotion, t0: float, seconds: float, speed: float, climb: float, x0=0.0, h0=0.0):
    """5 Hz poses walking along +x at `speed` m/s, rising `climb` m/s. Returns (events, x, h, t)."""
    events, x, h, t = [], x0, h0, t0
    for _ in range(int(seconds * 5)):
        t += 0.2
        x += speed * 0.2
        h += climb * 0.2
        e = det.step(t, x, 0.0, h)
        if e:
            events.append(e)
    return events, x, h, t


class StairMotionTests(unittest.TestCase):
    def test_flat_walk_is_silent(self):
        events, *_ = walk(StairMotion(), 0.0, 30, 1.3, 0.0)
        self.assertEqual(events, [])

    def test_ramp_is_silent(self):
        # 1:12 wheelchair ramp at walking pace
        events, *_ = walk(StairMotion(), 0.0, 20, 1.2, 1.2 / 12)
        self.assertEqual(events, [])

    def test_up_a_flight_then_end(self):
        det = StairMotion()
        e1, x, h, t = walk(det, 0.0, 3, 1.2, 0.0)
        e2, x, h, t = walk(det, t, 5, 0.5, 0.3, x, h)          # ~1.7 steps/s, slope 0.6
        e3, *_ = walk(det, t, 3, 1.2, 0.0, x, h)
        self.assertEqual(e1, [])
        self.assertEqual([e.kind for e in e2], [UP])
        self.assertLess(e2[0].t - 3.0, 1.5)                     # within 1.5 s of the first step
        self.assertEqual([e.kind for e in e3], [END])

    def test_slow_descent(self):
        # one step per second down: 0.17 m drop, 0.28 m tread
        det = StairMotion()
        _, x, h, t = walk(det, 0.0, 3, 1.0, 0.0)
        events, *_ = walk(det, t, 6, 0.28, -0.17, x, h)
        self.assertEqual([e.kind for e in events], [DOWN])

    def test_lift_does_not_count(self):
        det = StairMotion()
        _, x, h, t = walk(det, 0.0, 2, 1.0, 0.0)
        _, x, h, t = walk(det, t, 4, 0.0, 0.0, x, h)            # step in, wait for the doors
        events, *_ = walk(det, t, 8, 0.0, 1.0, x, h)            # going up, no horizontal motion
        self.assertEqual(events, [])

    def test_standing_on_the_flight_keeps_state(self):
        det = StairMotion()
        _, x, h, t = walk(det, 0.0, 2, 1.0, 0.0)
        _, x, h, t = walk(det, t, 3, 0.5, 0.3, x, h)
        self.assertTrue(det.on_stairs)
        events, *_ = walk(det, t, 5, 0.0, 0.0, x, h)
        self.assertEqual(events, [])
        self.assertTrue(det.on_stairs)


class VehicleClassTests(unittest.TestCase):
    FX, W = 1000.0, 1440.0

    def test_narrow_car_is_a_two_wheeler(self):
        # 120 px wide at 6 m = 0.72 m
        self.assertEqual(vehicle_class("car", 600, 720, 6.0, self.FX, self.W), "motorcycle")

    def test_real_car_stays(self):
        # 300 px at 6 m = 1.8 m
        self.assertEqual(vehicle_class("car", 500, 800, 6.0, self.FX, self.W), "car")

    def test_cut_by_the_frame_edge_or_no_depth_stays(self):
        self.assertEqual(vehicle_class("car", 0, 100, 6.0, self.FX, self.W), "car")
        self.assertEqual(vehicle_class("car", 1340, 1440, 6.0, self.FX, self.W), "car")
        self.assertEqual(vehicle_class("car", 600, 720, None, self.FX, self.W), "car")

    def test_other_classes_untouched(self):
        self.assertEqual(vehicle_class("person", 600, 640, 6.0, self.FX, self.W), "person")


class StairPhraseTests(unittest.TestCase):
    def test_phrases(self):
        self.assertEqual(edge_phrase(UP, 0), "Going up stairs.")
        self.assertEqual(edge_phrase(DOWN, 0), "Going down stairs. Careful.")
        self.assertEqual(edge_phrase(END, 0), "End of stairs.")
        self.assertEqual(edge_phrase("curb", 3.2), "Curb ahead, 3 meters.")
        self.assertIn("End of stairs.", ground_phrases())


if __name__ == "__main__":
    unittest.main()
