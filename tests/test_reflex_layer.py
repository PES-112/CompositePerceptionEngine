import unittest

from src.reflex_layer import process_reflex_frame
from src.threat_prioritizer import build_threat_frame


def row(**overrides):
    base = {
        "frame_idx": 10,
        "source": "sanpo",
        "track_id": "7",
        "class": "bicycle",
        "confidence": 0.9,
        "bbox_x1": 10,
        "bbox_y1": 20,
        "bbox_x2": 60,
        "bbox_y2": 100,
        "cx_px": 35,
        "bearing_deg": -5,
        "distance_m": 3.0,
        "velocity_ms": 1.0,
    }
    base.update(overrides)
    return base


class ReflexLayerTests(unittest.TestCase):
    def test_ttc_override_becomes_immediate_narrator_event(self):
        frame = build_threat_frame([row(distance_m=2.0, velocity_ms=3.0)])
        result = process_reflex_frame(frame)

        self.assertTrue(result.has_alert)
        self.assertTrue(result.override_active)
        self.assertEqual(result.reason, "override")
        self.assertEqual(result.narrator_event.track_id, "7")
        self.assertEqual(result.narrator_event.object_class, "bicycle")
        self.assertTrue(result.narrator_event.is_override)
        self.assertIn("OVERRIDE", result.narrator_event.reason)

    def test_non_reflex_event_does_not_trigger_override(self):
        frame = build_threat_frame([
            row(track_id="p1", **{"class": "pothole"}, distance_m=1.2, velocity_ms=0.0)
        ])
        result = process_reflex_frame(frame)

        self.assertFalse(result.has_alert)
        self.assertFalse(result.override_active)
        self.assertEqual(result.reason, "no reflex-route events")
        self.assertEqual(result.reflex_results, ())

    def test_high_kinetic_reflex_uses_physics_fallback_without_slm(self):
        frame = build_threat_frame([row(track_id="car1", **{"class": "car"}, distance_m=5.0, velocity_ms=4.0)])
        result = process_reflex_frame(frame)

        self.assertTrue(result.has_alert)
        self.assertFalse(result.override_active)
        self.assertEqual(result.reason, "physics fallback")
        self.assertEqual(result.narrator_event.track_id, "car1")
        self.assertEqual(result.narrator_event.object_class, "car")
        self.assertFalse(result.narrator_event.is_override)
        self.assertIn("Physics fallback", result.narrator_event.reason)


if __name__ == "__main__":
    unittest.main()
