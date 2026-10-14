import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset.GRU.samples import N_CLASSES, N_FEATURES, session_samples
from models.GRU.registry import build_model
from src.threat_engine import envelope
from src.threat_engine.announcer import AnnouncementManager
from src.threat_engine.engine import ThreatEngine
from src.threat_engine.fusion import EgoPose, FusedTrack, GroundTracker, Measurement, time_to_radius
from src.threat_engine.priority import (Arrival, GRUArrivalPredictor, PhysicsArrivalPredictor, RankedThreat,
                                        SceneContext, rank)

DT = 0.2   # SANPO processing step: 15 fps, stride 3


def pose_at(x: float, z: float, yaw_deg: float = 0.0) -> EgoPose:
    """Camera at (x, z) on the ground, facing yaw degrees clockwise from +z."""
    y = math.radians(yaw_deg)
    return EgoPose((x, z), (math.cos(y), -math.sin(y)), (math.sin(y), math.cos(y)))


def camera_measurement(tid, cls, world_xz, pose, depth_none=False):
    """What a camera at `pose` would measure for a point at world_xz."""
    rel = np.asarray(world_xz) - np.asarray(pose.position)
    x_cam, z_cam = rel @ np.asarray(pose.right), rel @ np.asarray(pose.forward)
    return Measurement(tid, cls, None if depth_none else z_cam, math.degrees(math.atan2(x_cam, z_cam)))


def track(**kw) -> FusedTrack:
    base = dict(track_id="t", class_name="person", confidence=0.9, forward_m=3.0, lateral_m=0.0,
                distance_m=3.0, bearing_deg=0.0, closing_speed_ms=1.0, lateral_speed_ms=0.0,
                rel_velocity=(0.0, -1.0), position_std_m=0.1, speed_std_ms=0.1, hits=5, misses=0,
                confirmed=True, measured=True)
    base.update(kw)
    return FusedTrack(**base)


class FusionTests(unittest.TestCase):
    def test_static_object_while_walking_closes_at_walking_speed(self):
        tracker = GroundTracker()
        for k in range(15):
            pose = pose_at(0.0, 1.4 * DT * k)                   # walking +z at 1.4 m/s
            out = tracker.step(k * DT, [camera_measurement("pole", "pole", (0.5, 12.0), pose)], pose)
        t = out[0]
        self.assertAlmostEqual(t.closing_speed_ms, 1.4 * math.cos(math.atan2(0.5, t.forward_m)), delta=0.1)
        f = tracker._filters["pole"]
        self.assertLess(np.linalg.norm(f.x[2:]), 0.15)           # the pole itself is still in the world

    def test_head_turn_is_not_object_motion(self):
        def lateral_speed(use_pose):
            tracker = GroundTracker()
            for k in range(10):
                pose = pose_at(0.0, 0.0, yaw_deg=6.0 * k)        # head turns 30 deg/s, walker stands still
                m = camera_measurement("car", "car", (0.0, 8.0), pose)
                out = tracker.step(k * DT, [m], pose if use_pose else None)
            return abs(out[0].lateral_speed_ms)
        self.assertLess(lateral_speed(True), 0.1)
        self.assertGreater(lateral_speed(False), 1.0)            # camera-frame tracking sees a sweeping car

    def test_retreat_is_negative(self):
        tracker = GroundTracker()
        for k in range(10):
            out = tracker.step(k * DT, [Measurement("p", "person", 3.0 + 1.0 * DT * k, 0.0)])
        self.assertLess(out[0].closing_speed_ms, -0.7)

    def test_missing_depth_coasts_then_drops(self):
        tracker = GroundTracker(max_misses=2)
        for k in range(5):
            tracker.step(k * DT, [Measurement("p", "person", 5.0 - DT * k, 0.0)])
        out = tracker.step(5 * DT, [Measurement("p", "person", None, 0.0)])
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0].measured)
        self.assertEqual(out[0].misses, 1)
        self.assertAlmostEqual(out[0].forward_m, 4.0, delta=0.15)  # predicted on, not frozen
        tracker.step(6 * DT, [])
        self.assertEqual(tracker.step(7 * DT, []), [])

    def test_depth_outlier_is_gated(self):
        tracker = GroundTracker()
        for k in range(8):
            tracker.step(k * DT, [Measurement("p", "person", 6.0, 0.0)])
        out = tracker.step(8 * DT, [Measurement("p", "person", 1.0, 0.0)])   # occluder in the box for one frame
        self.assertGreater(out[0].forward_m, 5.5)
        self.assertFalse(out[0].measured)

    def test_sanpo_identity_pose_looks_down_minus_z(self):
        pose = EgoPose.from_sanpo(0, 0, 0, 0, 0, 0, 1)
        np.testing.assert_allclose(pose.forward, (0.0, -1.0), atol=1e-9)
        np.testing.assert_allclose(pose.right, (1.0, 0.0), atol=1e-9)

    def test_time_to_radius(self):
        self.assertEqual(time_to_radius(np.array([0.0, 1.0]), np.array([0.0, -1.0]), 2.0), 0.0)
        self.assertAlmostEqual(time_to_radius(np.array([0.0, 5.0]), np.array([0.0, -1.0]), 3.0), 2.0)
        self.assertIsNone(time_to_radius(np.array([0.0, 5.0]), np.array([0.0, 1.0]), 3.0))
        self.assertIsNone(time_to_radius(np.array([5.0, 5.0]), np.array([0.0, -1.0]), 3.0))   # passes wide


class EnvelopeTests(unittest.TestCase):
    def test_exact_threshold_fires(self):
        self.assertIsNotNone(envelope.check(track(distance_m=2.0, forward_m=2.0, closing_speed_ms=2.0)))

    def test_zero_and_negative_closing_never_fire(self):
        for v in (0.0, -1.0, 0.1):
            self.assertIsNone(envelope.check(track(distance_m=0.5, forward_m=0.5, closing_speed_ms=v)))

    def test_outside_cone_or_behind_never_fires(self):
        self.assertIsNone(envelope.check(track(bearing_deg=45.0, distance_m=1.0, closing_speed_ms=3.0)))
        self.assertIsNone(envelope.check(track(forward_m=-1.0, distance_m=1.0, closing_speed_ms=3.0)))
        self.assertIsNotNone(envelope.check(track(bearing_deg=29.0, distance_m=1.0, closing_speed_ms=3.0)))

    def test_unconfirmed_never_fires_and_short_coast_still_does(self):
        self.assertIsNone(envelope.check(track(confirmed=False, distance_m=0.5, closing_speed_ms=3.0)))
        self.assertIsNotNone(envelope.check(track(misses=2, measured=False, distance_m=0.5, closing_speed_ms=3.0)))
        self.assertIsNone(envelope.check(track(misses=3, measured=False, distance_m=0.5, closing_speed_ms=3.0)))

    def test_overrides_sorted_soonest_first(self):
        found = envelope.overrides([track(track_id="a", distance_m=1.8, closing_speed_ms=2.0),
                                    track(track_id="b", distance_m=0.6, closing_speed_ms=2.0)])
        self.assertEqual([o.track.track_id for o in found], ["b", "a"])


class PriorityTests(unittest.TestCase):
    def test_gamma_zero_is_arrival_only_and_threshold_ignores(self):
        tracks = [track(track_id="bus", class_name="bus"), track(track_id="p", class_name="person"),
                  track(track_id="far", class_name="person")]
        arrivals = {"bus": Arrival(0.4, 3.0), "p": Arrival(0.6, 2.0), "far": Arrival(0.05, 5.0)}
        ranked = rank(tracks, arrivals, gamma=0.0, ignore_threshold=0.1)
        self.assertEqual([r.track.track_id for r in ranked], ["p", "bus"])
        ranked = rank(tracks, arrivals, gamma=1.0, ignore_threshold=0.1)
        self.assertEqual(ranked[0].track.track_id, "bus")       # raw severity lets the bus win

    def test_ties_broken_by_expected_arrival(self):
        tracks = [track(track_id="late"), track(track_id="soon")]
        ranked = rank(tracks, {"late": Arrival(0.5, 3.0), "soon": Arrival(0.5, 1.0)}, gamma=0.0)
        self.assertEqual(ranked[0].track.track_id, "soon")

    def test_context_multiplier(self):
        tracks = [track(track_id="a"), track(track_id="b")]
        arrivals = {"a": Arrival(0.5, 2.0), "b": Arrival(0.45, 2.0)}
        ranked = rank(tracks, arrivals, gamma=0.0, contexts={"b": SceneContext(in_path=True)})
        self.assertEqual(ranked[0].track.track_id, "b")

    def test_physics_predictor(self):
        pred = PhysicsArrivalPredictor()
        out = pred.predict(0, [track(track_id="near", forward_m=4.0, distance_m=4.0, rel_velocity=(0.0, -1.5)),
                               track(track_id="away", forward_m=4.0, distance_m=4.0, rel_velocity=(0.0, 1.0))])
        self.assertGreater(out["near"].p_arrive, 0.9)
        self.assertEqual(out["away"].p_arrive, 0.0)
        beside = track(track_id="beside", forward_m=2.3, lateral_m=1.93, distance_m=3.0, bearing_deg=40.0,
                       rel_velocity=(0.0, 0.0))
        self.assertEqual(pred.predict(0, [beside])["beside"].p_arrive, 0.0)   # inside 3 m but outside the cone

    def test_gru_runtime_features_match_dataset(self):
        """The runtime predictor must encode exactly what dataset/GRU/samples.py trained on."""
        rows = []
        for f in range(14):
            for tid, cls, bearing, dist in (("1", "person", 2.0 + f, 9.0 - 0.4 * f), ("2", "car", -20.0, 7.0)):
                if tid == "2" and f in (3, 4, 9):
                    continue                                    # gaps in a track
                rows.append({"frame_idx": f, "source": "sanpo", "track_id": tid, "class": cls,
                             "confidence": 0.8, "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 40 + f, "bbox_y2": 90,
                             "cx_px": 20, "bearing_deg": bearing, "distance_m": dist, "velocity_ms": 0.7})
        expected = session_samples(pd.DataFrame(rows), "s", tracked_only=True, hazard_m=3.0)

        torch.manual_seed(0)
        model = build_model("gru", {"n_features": N_FEATURES, "n_classes": N_CLASSES, "n_outputs": 5})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.pt"
            torch.save({"arch": "gru", "model_kwargs": {"n_features": N_FEATURES, "n_classes": N_CLASSES,
                                                        "n_outputs": 5},
                        "state_dict": model.state_dict(),
                        "config": {"time_bins": 4, "horizon": 20, "dataset_params": {"history": 8, "max_objects": 32}}},
                       path)
            pred = GRUArrivalPredictor(path)
        for sample in expected:
            f = sample["frame_idx"]
            tracks = [track(track_id=r["track_id"], class_name=r["class"], row=r) for r in rows if r["frame_idx"] == f]
            pred.observe(f, tracks)
            live, x, cls = pred.encode(f, tracks)
            by_id = {t.track_id: i for i, t in enumerate(live)}
            for j, tid in enumerate(sample["track_ids"]):
                np.testing.assert_allclose(x[by_id[tid]], sample["x"][j], rtol=1e-6, atol=1e-6)
            out = pred.predict(f, [])                           # predict() must not crash on an empty frame
            self.assertEqual(out, {})


def ranked_threat(tid, priority):
    return RankedThreat(track(track_id=tid), priority, Arrival(min(priority, 1.0), 2.0), 1.0, 1.0)


class AnnouncerTests(unittest.TestCase):
    def test_override_preempts_a_playing_clip(self):
        mgr = AnnouncementManager()
        self.assertEqual(mgr.step(0.0, [], [ranked_threat("a", 0.5)]).kind, "threat")
        o = envelope.check(track(track_id="b", distance_m=1.0, closing_speed_ms=2.0))
        a = mgr.step(0.2, [o], [ranked_threat("a", 0.5)])
        self.assertEqual((a.kind, a.track_id), ("override", "b"))
        self.assertIsNone(mgr.step(0.4, [o], []))                # rate-limited per track
        self.assertEqual(mgr.step(2.4, [o], []).kind, "override")

    def test_cooldown_and_escalation(self):
        mgr = AnnouncementManager(clip_s=0.0)
        self.assertIsNotNone(mgr.step(0.0, [], [ranked_threat("a", 0.4)]))
        self.assertIsNone(mgr.step(1.0, [], [ranked_threat("a", 0.45)]))
        self.assertIsNotNone(mgr.step(1.2, [], [ranked_threat("a", 0.7)]))   # 1.75x: escalated
        self.assertIsNone(mgr.step(2.0, [], [ranked_threat("a", 0.7)]))
        self.assertIsNotNone(mgr.step(5.3, [], [ranked_threat("a", 0.7)]))   # cooldown over

    def test_hysteresis_keeps_current_object(self):
        mgr = AnnouncementManager(clip_s=0.0, cooldown_s=0.0)
        mgr.step(0.0, [], [ranked_threat("a", 0.5)])
        a = mgr.step(0.2, [], [ranked_threat("b", 0.55), ranked_threat("a", 0.5)])
        self.assertEqual(a.track_id, "a")                        # 0.55 < 1.25 x 0.5
        a = mgr.step(0.4, [], [ranked_threat("b", 0.7), ranked_threat("a", 0.5)])
        self.assertEqual(a.track_id, "b")

    def test_stale_message_is_dropped_not_queued(self):
        mgr = AnnouncementManager(clip_s=3.0)
        mgr.step(0.0, [], [ranked_threat("a", 0.5)])             # clip plays until t = 3.0
        self.assertIsNone(mgr.step(0.2, [], [ranked_threat("b", 0.9)]))
        self.assertIsNone(mgr.step(1.0, [], []))                 # still busy, and b is now stale
        self.assertIsNone(mgr.step(3.2, [], []))                 # b was dropped, not played late


class EngineTests(unittest.TestCase):
    def test_approaching_person_triggers_override_and_narrator_event(self):
        engine = ThreatEngine()
        frames = []
        for k in range(12):
            rows = [{"frame_idx": k, "track_id": "7", "class": "person", "confidence": 0.9,
                     "bearing_deg": 0.0, "distance_m": 6.0 - 0.45 * k},
                    {"frame_idx": k, "track_id": "obs_1_100", "class": "unlabeled_obstacle",
                     "bearing_deg": 0.0, "distance_m": 0.8}]
            frames.append(engine.step(k, k * DT, rows))
        self.assertTrue(all("obs_1_100" not in {t.track_id for t in f.tracks} for f in frames))
        fired = [f for f in frames if f.announcement and f.announcement.kind == "override"]
        self.assertTrue(fired)
        event = fired[0].narrator_event()
        self.assertTrue(event.is_override)
        self.assertEqual(event.object_class, "person")
        self.assertGreater(event.closing_velocity_ms, 1.5)


if __name__ == "__main__":
    unittest.main()
