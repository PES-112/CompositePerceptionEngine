import unittest

import numpy as np

from models.Ensemble.ensemble import fit_softmax, predict_logits, rank_average, stack_out_of_fold


def fake_samples(sizes, rng):
    return [{"y": np.zeros(n, np.float32), "tte": rng.integers(-1, 21, n)} for n in sizes]


class EnsembleTests(unittest.TestCase):
    def test_rank_average_keeps_an_order_all_components_agree_on(self):
        samples = fake_samples([3, 2], np.random.default_rng(0))
        a = [np.array([3.0, 1.0, 2.0]), np.array([5.0, 4.0])]
        b = [np.array([30.0, 10.0, 20.0]), np.array([50.0, 40.0])]
        out = rank_average(samples, [a, b])
        self.assertEqual([len(o) for o in out], [3, 2])
        flat = np.concatenate(out)
        np.testing.assert_array_equal(np.argsort(flat), np.argsort(np.concatenate(a)))

    def test_softmax_regression_learns_a_separable_problem(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(600, 2))
        target = (x[:, 0] > 0).astype(int) + 2 * (x[:, 1] > 0).astype(int)   # 4 quadrants
        model = fit_softmax(x, target, 4)
        accuracy = (predict_logits(model, x).argmax(1) == target).mean()
        self.assertGreater(accuracy, 0.95)

    def test_stacking_never_sees_the_labels_of_the_fold_it_scores(self):
        rng = np.random.default_rng(1)
        samples = fake_samples([4] * 30, rng)
        x = rng.normal(size=(120, 3))
        fold_of = np.repeat(np.arange(3), 40)
        before, _ = stack_out_of_fold(samples, x, fold_of, time_bins=4, horizon=20)
        for s in samples[:10]:          # the first 40 objects = fold 0
            s["tte"] = rng.integers(-1, 21, 4)
        after, _ = stack_out_of_fold(samples, x, fold_of, time_bins=4, horizon=20)
        np.testing.assert_allclose(after[fold_of == 0], before[fold_of == 0])
        self.assertFalse(np.allclose(after[fold_of == 1], before[fold_of == 1]))


if __name__ == "__main__":
    unittest.main()
