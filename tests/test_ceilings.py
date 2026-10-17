import unittest

import numpy as np

from equity_fingerprint import ceilings as C
from equity_fingerprint import targets as T


class CeilingTests(unittest.TestCase):
    def test_vol_reliability_matches_theory(self):
        rng = np.random.default_rng(0)
        n, H = 3000, 21
        sigma = np.exp(rng.normal(np.log(0.25), 0.5, n)) / np.sqrt(252)  # daily
        w = rng.normal(size=(H, n)) * sigma
        a, b, full = C.half_statistics("vol_1m", w)
        rel = C.split_half_reliability(a, b)
        # theory: var(true log sigma)=0.25, noise var of log std of m iid normals ~ 1/(2(m-1)) per half
        m = H / 2
        noise = 1 / (2 * (m - 1))
        r_theory = 0.25 / (0.25 + noise)
        self.assertAlmostEqual(rel["split_half_r"], r_theory, delta=0.04)
        self.assertAlmostEqual(rel["reliability"], 2 * r_theory / (1 + r_theory), delta=0.04)
        self.assertEqual(rel["n_obs"], n)
        # the full-window statistic should be closer to the truth than a half
        truth = np.log(sigma * np.sqrt(252))
        self.assertLess(np.abs(full - truth).mean(), np.abs(a - truth).mean())

    def test_beta_and_liquidity_halves(self):
        rng = np.random.default_rng(1)
        H, n = 252, 2000
        mkt = rng.normal(size=H) * 0.01
        beta = rng.normal(1, 0.4, n)
        w = mkt[:, None] * beta + rng.normal(size=(H, n)) * 0.01
        a, b, full = C.half_statistics("beta_1y", w, mkt)
        rel = C.split_half_reliability(a, b)
        self.assertGreater(rel["reliability"], 0.5)
        self.assertLess(rel["reliability"], 1.0)
        self.assertAlmostEqual(np.corrcoef(full, beta)[0, 1] ** 2, rel["reliability"], delta=0.08)
        tv = np.exp(rng.normal(size=n) * 2 + 10) * np.exp(rng.normal(size=(21, n)) * 0.3)
        a, b, _ = C.half_statistics("liquidity_1m", tv)
        self.assertGreater(C.split_half_reliability(a, b)["reliability"], 0.9)

    def test_full_matches_target_definition(self):
        import pandas as pd
        rng = np.random.default_rng(2)
        w = rng.normal(size=(21, 5)) * 0.01
        _, _, full = C.half_statistics("vol_1m", w)
        v = pd.DataFrame(w).rolling(21, min_periods=21).std().iloc[-1] * np.sqrt(C.ANN)
        np.testing.assert_allclose(full, np.log(v.clip(lower=T.VOL_FLOOR)))

    def test_nonfinite_excluded(self):
        w = np.random.default_rng(3).normal(size=(21, 10)) * 0.01
        w[3, 0] = np.nan
        a, b, _ = C.half_statistics("vol_1m", w)
        self.assertTrue(np.isnan(a[0]) and np.isnan(b[0]))
        self.assertEqual(C.split_half_reliability(a, b)["n_obs"], 9)

    def test_persistence_uses_train_only(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=300)
        y = 2 * x + 1 + rng.normal(size=300) * 0.5
        xv, yv = rng.normal(size=100), rng.normal(size=100)
        r1 = C.persistence_r2(x, y, {"val": (xv, yv)})
        r2 = C.persistence_r2(x, y, {"val": (xv, yv * 100 + 7)})
        self.assertEqual((r1["slope"], r1["intercept"]), (r2["slope"], r2["intercept"]))
        self.assertAlmostEqual(r1["slope"], 2, delta=0.1)
        self.assertNotEqual(r1["val"]["ols_r2"], r2["val"]["ols_r2"])
        self.assertAlmostEqual(C.r2_score(y, y), 1.0)


if __name__ == "__main__":
    unittest.main()
