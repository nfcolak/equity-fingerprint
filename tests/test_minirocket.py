import unittest

# torch before anything that may load lightgbm (macOS libomp segfault; forward_minirocket imports forward_v7)
import torch  # noqa: F401

import numpy as np
from sklearn.linear_model import RidgeClassifier

from equity_fingerprint.minirocket import MiniRocket


def sinusoids(n, length=128, seed=0):
    """Two classes: slow (period ~32) vs fast (period ~8) sinusoid with random phase, plus noise; 2 channels."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    t = np.arange(length)
    X = np.empty((n, 2, length))
    for i in range(n):
        per = 32 if y[i] == 0 else 8
        ph = rng.uniform(0, 2 * np.pi)
        s = np.sin(2 * np.pi * t / per + ph)
        X[i, 0] = s + 0.5 * rng.normal(size=length)
        X[i, 1] = rng.normal(size=length)  # pure noise channel
    return X, y


class MiniRocketTests(unittest.TestCase):
    def test_shape_and_determinism(self):
        X, _ = sinusoids(40)
        a = MiniRocket(1000, seed=1).fit(X)
        f1, f2 = a.transform(X), MiniRocket(1000, seed=1).fit(X).transform(X)
        self.assertEqual(f1.shape, (40, a.n_features_))
        self.assertEqual(a.n_features_, 84 * (1000 // 84))
        self.assertTrue(np.array_equal(f1, f2))
        self.assertFalse(np.array_equal(f1, MiniRocket(1000, seed=2).fit(X).transform(X)))
        # transform does not depend on batch size
        self.assertTrue(np.array_equal(f1, a.transform(X, batch=7)))

    def test_dilations_exponential_up_to_limit(self):
        m = MiniRocket(5000).fit(sinusoids(10, 252)[0])
        d = m.dilations_
        self.assertEqual(d[0], 1)
        self.assertLessEqual(d.max(), (252 - 1) // 8)
        self.assertEqual(int(m.feat_per_dilation_.sum()), 5000 // 84)

    def test_biases_depend_only_on_fit_data(self):
        X, _ = sinusoids(40, seed=0)
        Z, _ = sinusoids(40, seed=5)
        a = MiniRocket(500, seed=3).fit(X)
        b = MiniRocket(500, seed=3).fit(X)
        a.transform(Z)  # transforming other data must not change the biases
        self.assertTrue(np.array_equal(a.bias_, b.bias_))
        c = MiniRocket(500, seed=3).fit(Z)
        self.assertFalse(np.array_equal(a.bias_, c.bias_))

    def test_ppv_in_unit_interval(self):
        X, _ = sinusoids(30)
        f = MiniRocket(800, seed=0).fit(X).transform(X)
        self.assertGreaterEqual(float(f.min()), 0.0)
        self.assertLessEqual(float(f.max()), 1.0)
        self.assertGreater(float(f.std()), 0.05)

    def test_separable_with_ridge(self):
        Xtr, ytr = sinusoids(200, seed=1)
        Xte, yte = sinusoids(200, seed=2)
        m = MiniRocket(1500, seed=0).fit(Xtr)
        ftr, fte = m.transform(Xtr), m.transform(Xte)
        mu, sd = ftr.mean(0), ftr.std(0) + 1e-8
        clf = RidgeClassifier(alpha=10.0).fit((ftr - mu) / sd, ytr)
        self.assertGreater(clf.score((fte - mu) / sd, yte), 0.95)

    def test_window_causality(self):
        # a window ending at t (inclusive) is a function of that window only: appending/changing later data
        # leaves the feature of an earlier window unchanged
        rng = np.random.default_rng(0)
        series = rng.normal(size=(2, 400))
        win = 128
        t = 250
        w1 = series[:, t - win + 1:t + 1][None]
        changed = series.copy()
        changed[:, t + 1:] = rng.normal(size=(2, 400 - t - 1)) * 10
        w2 = changed[:, t - win + 1:t + 1][None]
        fit = np.stack([series[:, s:s + win] for s in range(0, 250, 10)])
        m = MiniRocket(600, seed=0).fit(fit)
        self.assertTrue(np.array_equal(m.transform(w1), m.transform(w2)))
        # sanity: changing a day inside the window changes the features
        changed2 = series.copy()
        changed2[:, t] += 5.0
        self.assertFalse(np.array_equal(m.transform(w1), m.transform(changed2[:, t - win + 1:t + 1][None])))


if __name__ == "__main__":
    unittest.main()
