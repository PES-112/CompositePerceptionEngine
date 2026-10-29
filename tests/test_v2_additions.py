"""Tests for the 2026-10-25 additions: rising trigger, fused GRU inputs, edge memory, stereo panning, CRC."""

import math
import unittest

import numpy as np

from src.narration.spatial import pan_gains, pan_stereo
from src.segmentation.edge_alert import EdgeMemoryAlerter
from src.segmentation.context import EdgeAhead
from src.segmentation.labels import CURB
from src.threat_engine.fusion import EgoPose
from src.threat_engine.priority import Arrival, FusedHistory, RisingArrivalPredictor
from tests.test_threat_engine import track
from tools.replay_threat_engine import CRC_CAP, crc_risk


class Fixed:
    def __init__(self, ps):
        self.ps = iter(ps)

    def predict(self, frame_idx, tracks):
        return {"a": Arrival(next(self.ps), 2.0)}


class RisingTests(unittest.TestCase):
    def test_beta_zero_is_identity_and_only_rises_are_boosted(self):
        seq = [0.1, 0.2, 0.4, 0.4, 0.2]
        same = RisingArrivalPredictor(Fixed(seq), beta=0.0)
        self.assertEqual([round(same.predict(k, [])["a"].p_arrive, 6) for k in range(5)], seq)
        up = RisingArrivalPredictor(Fixed(seq), beta=1.0)
        out = [up.predict(k, [])["a"].p_arrive for k in range(5)]
        self.assertAlmostEqual(out[0], 0.1, 6)                       # no history yet
        self.assertGreater(out[2], 0.4)                              # rising: boosted
        self.assertAlmostEqual(out[4], 0.2, 6)                       # falling: unchanged


class FusedHistoryTests(unittest.TestCase):
    def test_features_align_with_history_and_pad_missing_steps(self):
        h = FusedHistory()
        t = track(track_id="a", distance_m=4.0, closing_speed_ms=1.0)
        t = t.__class__(**{**t.__dict__, "measured": True, "row": {"distance_m": 4.0}})
        h.observe(5, [t])
        x = h.features("a", 6, 4)
        self.assertEqual(x.shape, (4, 6))
        self.assertTrue(np.all(x[3] == 0))                           # frame 6 not observed
        self.assertAlmostEqual(float(x[2, 3]), 1.0 / 5, 6)           # closing speed / 5 at frame 5


class EdgeMemoryTests(unittest.TestCase):
    def test_edge_seen_far_is_announced_by_time_to_reach_after_leaving_view(self):
        alerter = EdgeMemoryAlerter(confirm_frames=2, ttr_s=3.0)
        alerts = []
        for k in range(30):                                          # 1.4 m/s along +y, 2.5 Hz
            t = k * 0.4
            pose = EgoPose((0.0, 1.4 * t), (1.0, 0.0), (0.0, 1.0))
            seen = EdgeAhead(CURB, 9.0 - 1.4 * t) if t < 1.0 else None   # visible only while far
            a = alerter.step(t, seen, pose)
            if a:
                alerts.append(a)
        self.assertEqual(len(alerts), 1)
        self.assertGreater(alerts[0].t, 1.0)                         # after the edge left the image
        self.assertLessEqual(alerts[0].distance_m, 5.0 + 1e-6)       # within ALERT_M (or 3 s at 1.4 m/s)
        self.assertGreater(alerts[0].distance_m, 0.0)


class SpatialTests(unittest.TestCase):
    def test_constant_power_and_sides(self):
        for b in (-90, -30, 0, 20, 60):
            l, r = pan_gains(b)
            self.assertAlmostEqual(l * l + r * r, 1.0, 6)
        self.assertGreater(pan_gains(-30)[0], pan_gains(-30)[1])
        self.assertEqual(pan_stereo(np.full(10, 1000, np.int16), 0.0).shape, (10, 2))


class CRCTests(unittest.TestCase):
    def test_bound_adds_the_finite_sample_correction(self):
        walks = [{"false": 5, "minutes": 1.0}] * 9
        self.assertAlmostEqual(crc_risk(walks), 0.9 * 5 / CRC_CAP + 0.1, 9)


if __name__ == "__main__":
    unittest.main()
