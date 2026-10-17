import math
import unittest

import numpy as np

from src.threat_engine.fusion import FusedTrack, GroundTracker, Measurement, NoiseModel
from src.threat_engine.kalman_arrival import KalmanArrivalPredictor
from tools.replay_threat_engine import session_metrics

DT = 0.2


def row(k, tid, d, bearing=0.0, tte=-1):
    return {"frame_idx": k, "track_id": tid, "class": "person", "confidence": 0.9, "distance_m": d,
            "bearing_deg": bearing, "velocity_ms": 0.0, "bbox_area_px": 100.0,
            "_legacy": {"bearing_deg": bearing, "velocity_ms": 0.0}, "_tte": tte}


def fused(**kw) -> FusedTrack:
    base = dict(track_id="t", class_name="person", confidence=0.9, forward_m=6.0, lateral_m=0.0, distance_m=6.0,
                bearing_deg=0.0, closing_speed_ms=1.0, lateral_speed_ms=0.0, rel_velocity=(0.0, -1.0),
                position_std_m=0.1, speed_std_ms=0.1, hits=5, misses=0, confirmed=True, measured=True)
    base.update(kw)
    return FusedTrack(**base)


class LatencyTests(unittest.TestCase):
    def setUp(self):
        # seen at frame 0, arrives (tte 0) at frame 10
        self.frames = {k: [row(k, "p", 5.0 - 0.2 * k, tte=10 - k if k < 10 else 0)] for k in range(12)}

    def test_no_delay_matches_frame_timing(self):
        m = session_metrics(self.frames, [(9, "p", "threat")], 20)
        self.assertTrue(m["arrivals"][0]["event"])
        self.assertAlmostEqual(m["arrivals"][0]["lead_s"], 0.2)

    def test_latency_can_turn_a_warning_late(self):
        m = session_metrics(self.frames, [(9, "p", "threat")], 20, np.array([0.3]))
        self.assertFalse(m["arrivals"][0]["event"])          # heard 0.1 s after arrival
        m = session_metrics(self.frames, [(5, "p", "threat")], 20, np.array([0.05]))
        self.assertAlmostEqual(m["arrivals"][0]["lead_s"], 0.95)
        self.assertEqual(m["true"], 1)                        # still judged at the deciding frame


class KalmanArrivalTests(unittest.TestCase):
    def test_certain_track_heading_in_arrives(self):
        cov = np.diag([1e-4, 1e-4, 1e-4, 1e-4])
        p, t = KalmanArrivalPredictor().entry(fused(rel_cov=tuple(map(tuple, cov))))
        self.assertGreater(p, 0.99)
        self.assertAlmostEqual(t, 3.0, delta=0.21)                               # 6 m -> 3 m at 1 m/s

    def test_uncertain_velocity_gives_a_middling_probability(self):
        cov = np.diag([0.01, 0.01, 0.6, 0.6])                                    # σ ≈ 0.8 m/s per axis
        p, _ = KalmanArrivalPredictor().entry(fused(rel_cov=tuple(map(tuple, cov))))
        self.assertGreater(p, 0.1)
        self.assertLess(p, 0.9)

    def test_passing_beside_never_arrives(self):
        cov = np.diag([1e-4] * 4)
        p, _ = KalmanArrivalPredictor().entry(fused(lateral_m=3.0, rel_cov=tuple(map(tuple, cov))))
        self.assertEqual(p, 0.0)

    def test_without_covariance_uses_the_mean(self):
        self.assertEqual(KalmanArrivalPredictor().entry(fused())[0], 1.0)

    def test_tracker_supplies_covariance(self):
        tracker = GroundTracker()
        for k in range(6):
            out = tracker.step(k * DT, [Measurement("p", "person", 8.0 - 1.4 * DT * k, 0.0)])
        cov = np.asarray(out[0].rel_cov)
        self.assertEqual(cov.shape, (4, 4))
        np.testing.assert_allclose(cov, cov.T, atol=1e-9)
        self.assertGreater(min(np.linalg.eigvalsh(cov)), 0)


class NoiseModelTests(unittest.TestCase):
    def test_declared_matches_the_shipped_values(self):
        n = NoiseModel.declared()
        self.assertAlmostEqual(n.depth_std(10.0), 0.05 + 0.01 * 100)
        self.assertEqual((n.robust_nu, n.gate_nis), (0.0, 11.62))

    def test_student_t_downweights_a_jump_instead_of_dropping_it(self):
        tracker = GroundTracker(noise=NoiseModel(gate_nis=1e9))
        for k in range(8):
            tracker.step(k * DT, [Measurement("p", "person", 6.0, 0.0)])
        out = tracker.step(8 * DT, [Measurement("p", "person", 1.0, 0.0)])
        self.assertTrue(out[0].measured)
        self.assertGreater(out[0].forward_m, 4.5)                                # moved, but far less than 5 m


if __name__ == "__main__":
    unittest.main()
