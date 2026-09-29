import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset.GRU.samples import (
    FEATURES,
    N_CLASSES,
    N_FEATURES,
    arrival_bins,
    collate,
    fold_sessions,
    fold_split,
    load_dataset,
    load_split,
    save_split,
    session_samples,
    split_sessions,
)
from evaluation.kinetic_ablation import encounters, prepare
from models.GRU.evaluate import arrival_pairs, model_scores
from models.GRU.GRU.model import TrackGRU
from models.GRU.GRU_attention.model import TrackGRUAttention
from models.GRU.train import make_loss

D = FEATURES.index("distance")
DD = FEATURES.index("d_distance")
PRESENT = FEATURES.index("present")


def csv_rows():
    """A person walking straight at the camera, and a car passing at a steep bearing."""
    rows = []
    for step in range(12):
        frame = step * 3
        for tid, cls, bearing, dist in (("1", "person", 2.0, 6.0 - 0.5 * step),
                                        ("2", "car", 40.0, 8.0)):
            rows.append({"frame_idx": frame, "source": "sanpo", "track_id": tid, "class": cls,
                         "confidence": 0.9, "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 50, "bbox_y2": 100,
                         "cx_px": 25, "bearing_deg": bearing, "distance_m": dist, "velocity_ms": 1.5})
    return pd.DataFrame(rows)


def models():
    torch.manual_seed(0)
    return (TrackGRU(N_FEATURES, N_CLASSES).eval(),
            TrackGRUAttention(N_FEATURES, N_CLASSES, context_dropout=0.3).eval())


def random_frame(n_objects, history=8):
    x = torch.randn(1, n_objects, history, N_FEATURES)
    cls = torch.randint(0, N_CLASSES, (1, n_objects))
    return x, cls, torch.ones(1, n_objects, dtype=torch.bool)


class DataTests(unittest.TestCase):
    def test_labels_match_the_kinetic_ablation(self):
        raw = csv_rows()
        expected = encounters(prepare(raw))
        for s in session_samples(raw, "s1"):
            enc = expected.get(s["frame_idx"], set())
            self.assertEqual({t for t, y in zip(s["track_ids"], s["y"]) if y}, enc)
            self.assertEqual(s["frame_has_encounter"], bool(enc))

    def test_history_is_front_padded_and_tracks_approach(self):
        samples = session_samples(csv_rows(), "s1", history=4)
        first, last = samples[0], samples[-1]
        person = first["track_ids"].index("1")
        # Frame 0 has no past: only the current (last) step is present.
        self.assertEqual(first["x"][person, :, PRESENT].tolist(), [0, 0, 0, 1])
        person = last["track_ids"].index("1")
        self.assertTrue(np.all(last["x"][person, :, PRESENT] == 1))
        self.assertTrue(np.all(last["x"][person, 1:, DD] < 0))   # distance shrinking every step
        self.assertTrue(np.all(np.diff(last["x"][person, :, D]) < 0))

    def test_tracked_only_drops_untracked_depth_blobs(self):
        raw = csv_rows()
        blob = raw.iloc[[0]].assign(track_id="obs_0_1104", **{"class": "unlabeled_obstacle"}, distance_m=1.0)
        raw = pd.concat([raw, blob], ignore_index=True)
        kept = session_samples(raw, "s1")
        dropped = session_samples(raw, "s1", tracked_only=True)
        self.assertIn("obs_0_1104", kept[0]["track_ids"])
        self.assertTrue(kept[0]["y"][kept[0]["track_ids"].index("obs_0_1104")])   # blob at 1 m, dead ahead
        self.assertTrue(all(not t.startswith("obs_") for s in dropped for t in s["track_ids"]))
        self.assertFalse(dropped[0]["frame_has_encounter"])

    def test_saved_split_loads_back_identically(self):
        samples = session_samples(csv_rows(), "s1")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.npz"
            save_split(samples, path)
            loaded = load_split(path)
        self.assertEqual(len(loaded), len(samples))
        for a, b in zip(samples, loaded):
            for key in ("session", "frame_idx", "frame_has_encounter", "track_ids", "classes"):
                self.assertEqual(a[key], b[key])
            for key in ("x", "cls", "y", "k0", "distance", "bearing"):
                np.testing.assert_array_equal(a[key], b[key])

    def test_split_is_by_session_and_disjoint(self):
        split = split_sessions([f"s{i}" for i in range(20)], seed=0)
        parts = [set(v) for v in split.values()]
        self.assertEqual(sum(map(len, parts)), 20)
        self.assertFalse(parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])

    def test_arrival_times(self):
        # The person is 6.0 - 0.5*step m away at frame 3*step: first under 1.5 m at frame 30, under 3 m at frame 21.
        by_frame = {s["frame_idx"]: s for s in session_samples(csv_rows(), "s1")}
        person = lambda s: s["track_ids"].index("1")   # noqa: E731
        self.assertEqual(by_frame[12]["tte"][person(by_frame[12])], 18)
        self.assertEqual(by_frame[3]["tte"][person(by_frame[3])], -1)          # 27 frames away > 20
        wide = {s["frame_idx"]: s for s in session_samples(csv_rows(), "s1", hazard_m=3.0)}
        self.assertEqual(wide[3]["tte"][person(wide[3])], 18)
        for s in list(by_frame.values()) + list(wide.values()):
            np.testing.assert_array_equal(s["y"], (s["tte"] >= 0).astype(np.float32))
            self.assertEqual(s["tte"][s["track_ids"].index("2")], -1)       # the car is outside the cone
        np.testing.assert_array_equal(arrival_bins(np.array([0, 4, 5, 19, 20, -1]), 4), [0, 0, 1, 3, 3, 4])

    def test_folds_cover_every_session_once_and_spread_encounters(self):
        corpus = {f"pos{i}": session_samples(csv_rows(), f"pos{i}") for i in range(3)}
        corpus.update({f"neg{i}": session_samples(csv_rows()[lambda d: d.track_id == "2"], f"neg{i}")
                       for i in range(6)})
        folds = fold_sessions(corpus, 3, seed=1)
        self.assertEqual(sorted(sid for f in folds for sid in f), sorted(corpus))
        self.assertEqual([sum(sid.startswith("pos") for sid in f) for f in folds], [1, 1, 1])
        split = fold_split(folds, 2)
        self.assertEqual(split["test"], folds[2])
        self.assertEqual(split["val"], folds[0])
        self.assertEqual(set(split["train"]), set(folds[1]))

    def test_cross_validation_dataset_round_trip(self):
        corpus = {f"s{i}": session_samples(csv_rows(), f"s{i}") for i in range(4)}
        folds = fold_sessions(corpus, 2)
        with tempfile.TemporaryDirectory() as tmp:
            save_split([s for sid in sorted(corpus) for s in corpus[sid]], Path(tmp) / "all.npz")
            (Path(tmp) / "manifest.json").write_text(json.dumps({"features": FEATURES, "folds": folds}))
            manifest, split = load_dataset(Path(tmp), fold=1)
            with self.assertRaises(ValueError):
                load_dataset(Path(tmp))
        self.assertEqual({s["session"] for s in split["test"]}, set(folds[1]))
        self.assertEqual(manifest["split"]["test"], folds[1])
        self.assertEqual(len(split["train"]), 0)   # 2 folds: one test, one val, nothing left to train on


class ArrivalTests(unittest.TestCase):
    def test_arrival_pairs(self):
        tte = np.array([0, 5, -1])            # arrives now, arrives later, never
        self.assertEqual(arrival_pairs(tte, np.array([3.0, 2.0, 1.0])), (3.0, 3))
        self.assertEqual(arrival_pairs(tte, np.array([1.0, 2.0, 3.0])), (0.0, 3))
        self.assertEqual(arrival_pairs(tte, np.array([1.0, 1.0, 1.0])), (1.5, 3))
        self.assertEqual(arrival_pairs(np.array([-1, -1]), np.array([1.0, 2.0])), (0.0, 0))

    def test_model_scores(self):
        logits = [np.array([0.3, -1.2])]
        self.assertIs(model_scores(logits, 0)[0], logits)
        early, late = np.array([5.0, 0, 0, 0, 0]), np.array([0, 0, 0, 5.0, 0])
        arrive, soon = model_scores([np.stack([early, late, np.array([0, 0, 0, 0, 9.0])])], 4)
        self.assertGreater(soon[0][0], soon[0][1])          # mass in the first slice = sooner
        self.assertGreater(soon[0][1], soon[0][2])          # "never" is latest of all
        self.assertLess(arrive[0][2], 0.01)

    def test_arrival_loss_targets(self):
        loss = make_loss(4, 20, 1.0, "cpu")
        tte = torch.tensor([[0, 19, -1]])
        target = torch.tensor([0, 3, 4])
        perfect = torch.full((1, 3, 5), -20.0)
        perfect[0, torch.arange(3), target] = 20.0
        batch = {"tte": tte}
        mask = torch.ones(1, 3, dtype=torch.bool)
        self.assertLess(loss(perfect, batch, mask).item(), 1e-6)
        self.assertGreater(loss(-perfect, batch, mask).item(), 10)


class ModelTests(unittest.TestCase):
    def test_padding_never_changes_real_predictions(self):
        x, cls, mask = random_frame(3)
        for model in models():
            base = model(x, cls, mask)
            batch = collate([
                {"x": x[0].numpy(), "cls": cls[0].numpy(), "y": np.zeros(3, np.float32)},
                {"x": np.random.randn(7, 8, N_FEATURES).astype(np.float32),
                 "cls": np.zeros(7, np.int64), "y": np.zeros(7, np.float32)},
            ])
            batch["x"][0, 3:] = 99.0   # garbage in the padded slots
            padded = model(batch["x"], batch["cls"], batch["mask"])[0, :3]
            torch.testing.assert_close(padded, base[0], atol=1e-5, rtol=1e-5)

    def test_object_order_does_not_matter(self):
        x, cls, mask = random_frame(5)
        perm = torch.tensor([3, 0, 4, 1, 2])
        for model in models():
            out = model(x, cls, mask)[0]
            shuffled = model(x[:, perm], cls[:, perm], mask)[0]
            torch.testing.assert_close(shuffled, out[perm], atol=1e-5, rtol=1e-5)

    def test_only_the_attention_model_sees_other_objects(self):
        x, cls, mask = random_frame(3)
        changed = x.clone()
        changed[0, 2] += 5.0   # move a *different* object
        gru, attention = models()
        torch.testing.assert_close(gru(x, cls, mask)[0, 0], gru(changed, cls, mask)[0, 0])
        self.assertFalse(torch.allclose(attention(x, cls, mask)[0, 0], attention(changed, cls, mask)[0, 0]))

    def test_single_object_frames_and_context_dropout_stay_finite(self):
        x, cls, mask = random_frame(1)
        attention = models()[1].train()
        for _ in range(20):
            self.assertTrue(torch.isfinite(attention(x, cls, mask)).all())

    def test_arrival_time_output_shape(self):
        x, cls, mask = random_frame(3)
        for model in (TrackGRU(N_FEATURES, N_CLASSES, n_outputs=5),
                      TrackGRUAttention(N_FEATURES, N_CLASSES, n_outputs=5)):
            self.assertEqual(tuple(model.eval()(x, cls, mask).shape), (1, 3, 5))


if __name__ == "__main__":
    unittest.main()
