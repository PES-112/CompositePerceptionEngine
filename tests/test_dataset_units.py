"""Stage-1 units: legacy datasets (linear ±35° bearings, 2x speeds) are converted, pinhole ones are not."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset.GRU.build import AUDIT, csv_units
from dataset.GRU.samples import FEATURES, PINHOLE_UNITS, legacy_units
from models.GRU.benchmark import positions
from tools.replay_threat_engine import session_rows

FX, WIDTH = 1978.7, 2208


def sample(bearing_deg: float, speed_ms: float) -> dict:
    x = np.zeros((1, 2, len(FEATURES)), dtype=np.float32)
    x[0, :, FEATURES.index("present")] = 1
    x[0, :, FEATURES.index("distance")] = 0.3
    x[0, :, FEATURES.index("velocity")] = speed_ms / 5
    x[0, :, FEATURES.index("bearing")] = bearing_deg / 35
    return {"frame_idx": 0, "track_ids": ["t1"], "classes": ["person"], "x": x, "distance": np.array([3.0]),
            "bearing": np.array([bearing_deg]), "tte": np.array([-1])}


class UnitsTest(unittest.TestCase):
    def test_marker(self):
        self.assertTrue(legacy_units({"source": {}}))
        self.assertFalse(legacy_units({"source": {"units": PINHOLE_UNITS}}))

    def test_replay_rows(self):
        legacy = session_rows([sample(20.0, 2.0)], FX, WIDTH, legacy=True)[0][0]
        pinhole = session_rows([sample(20.0, 2.0)], FX, WIDTH, legacy=False)[0][0]
        self.assertAlmostEqual(legacy["bearing_deg"], math.degrees(math.atan2(20 / 35 * WIDTH / 2, FX)), places=4)
        self.assertAlmostEqual(legacy["velocity_ms"], 1.0, places=4)
        self.assertAlmostEqual(pinhole["bearing_deg"], 20.0, places=4)
        self.assertAlmostEqual(pinhole["velocity_ms"], 2.0, places=4)

    def test_benchmark_positions(self):
        _, x, z = positions(sample(20.0, 0.0), FX, WIDTH, legacy=False)
        self.assertAlmostEqual(float(x[0, -1]), 3.0 * math.tan(math.radians(20)), places=4)
        self.assertAlmostEqual(float(z[0, -1]), 3.0, places=4)

    def test_build_refuses_chest_camera_csvs(self):
        a = json.loads(AUDIT.read_text(encoding="utf-8"))["sessions"][0]
        with tempfile.TemporaryDirectory() as d:
            csv = Path(d) / f"{a['session_id']}.csv"
            csv.write_text("x")
            sidecar = csv.with_suffix(".frames.json")
            sidecar.write_text(json.dumps({"fx": a["fx"]}))
            self.assertEqual(csv_units([csv]), {"units": PINHOLE_UNITS})
            sidecar.write_text(json.dumps({"fx": a["chest_fx"]}))
            with self.assertRaises(SystemExit):
                csv_units([csv])
            sidecar.unlink()
            self.assertEqual(csv_units([csv]), {})          # no sidecar: legacy CSVs


if __name__ == "__main__":
    unittest.main()
