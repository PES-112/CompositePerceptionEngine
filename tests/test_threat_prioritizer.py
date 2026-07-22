import tempfile
import unittest
from pathlib import Path

from src.threat_prioritizer import (
    build_threat_frame,
    build_threat_frames_from_csv,
    compute_ttc,
    write_threat_events_jsonl,
)


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


class ThreatPrioritizerTests(unittest.TestCase):
    def test_ttc_is_none_for_static_or_unknown_distance(self):
        self.assertIsNone(compute_ttc(None, 4.0))
        self.assertIsNone(compute_ttc(3.0, 0.0))
        self.assertIsNone(compute_ttc(3.0, -1.0))

    def test_ttc_for_closing_object(self):
        self.assertEqual(compute_ttc(3.0, 1.5), 2.0)

    def test_reflex_route_for_low_ttc(self):
        frame = build_threat_frame([row(distance_m=2.0, velocity_ms=3.0)])
        event = frame.events[0]
        self.assertEqual(event.route, "reflex")
        self.assertEqual(event.track_id, "7")
        self.assertLess(event.ttc_s, 1.0)
        self.assertTrue(frame.reflex_results[0].override)

    def test_cognitive_route_for_near_static_hazard(self):
        frame = build_threat_frame([
            row(track_id="p1", **{"class": "pothole"}, distance_m=1.2, velocity_ms=0.0)
        ])
        event = frame.events[0]
        self.assertEqual(event.route, "cognitive")
        self.assertIn("near static pothole", event.reason)

    def test_ignore_route_for_far_static_object(self):
        frame = build_threat_frame([
            row(track_id="b1", **{"class": "bench"}, distance_m=6.0, velocity_ms=0.0)
        ])
        self.assertEqual(frame.events[0].route, "ignore")

    def test_csv_to_jsonl_writes_only_non_ignore_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "perception.csv"
            csv_path.write_text(
                "frame_idx,source,track_id,class,confidence,bbox_x1,bbox_y1,bbox_x2,bbox_y2,cx_px,bearing_deg,distance_m,velocity_ms\n"
                "1,sanpo,1,bicycle,0.9,0,0,10,10,5,0,2.0,3.0\n"
                "1,sanpo,2,bench,0.8,0,0,10,10,5,0,6.0,0.0\n",
                encoding="utf-8",
            )
            frames = build_threat_frames_from_csv(csv_path)
            out = tmp_path / "events.jsonl"
            written = write_threat_events_jsonl(frames, out)
            lines = out.read_text(encoding="utf-8").splitlines()
            self.assertEqual(written, 1)
            self.assertEqual(len(lines), 1)
            self.assertIn('"route":"reflex"', lines[0])


if __name__ == "__main__":
    unittest.main()
