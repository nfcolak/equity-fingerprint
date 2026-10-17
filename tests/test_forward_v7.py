import unittest

# torch must be loaded before lightgbm (imported via forward_v7 -> models): the reverse order makes torch
# segfault on macOS (two libomp copies) in later test modules such as test_fp_profile.
import torch  # noqa: F401

import numpy as np
import pandas as pd

from equity_fingerprint import forward_v7 as V


def synth(n=600, k=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    m = pd.Series(rng.normal(0, 0.01, n), index=idx)
    betas = np.array([0.5, 1.0, 1.4, 0.8])[:k]
    r = pd.DataFrame(np.outer(m.to_numpy(), betas) + rng.normal(0, 0.002, (n, k)), index=idx,
                     columns=list("ABCD")[:k])
    return r, m


class WelchTests(unittest.TestCase):
    def test_equals_ols_when_no_clipping_binds(self):
        r, m = synth()
        # bounds are [-2m, 4m] for m>0 and [4m, -2m] for m<0; widen so nothing clips
        w = V.welch_beta(r, m, 252, lo=-1e6, hi=1e6)
        plain = r.rolling(252, min_periods=252).cov(m).div(m.rolling(252, min_periods=252).var(), axis=0)
        pd.testing.assert_frame_equal(w, plain)
        self.assertTrue(w.iloc[:251].isna().all().all())
        # the default bounds do not bind either for a clean beta in [0.5, 1.4] with small noise on big |m| days?
        # check clip is identity exactly where within bounds
        c = V.welch_clip(r, m)
        inside = (r >= np.minimum(-2 * m, 4 * m).to_numpy()[:, None]) & (r <= np.maximum(-2 * m, 4 * m).to_numpy()[:, None])
        self.assertTrue((c.where(inside) == r.where(inside)).values[inside.values].all())

    def test_clipping_binds_on_outlier(self):
        r, m = synth()
        r2 = r.copy()
        r2.iloc[300, 0] = 0.5  # huge idiosyncratic jump
        c = V.welch_clip(r2, m)
        self.assertLessEqual(c.iloc[300, 0], max(-2 * m.iloc[300], 4 * m.iloc[300]) + 1e-15)
        self.assertLess(V.welch_beta(r2, m).iloc[400, 0] - V.welch_beta(r, m).iloc[400, 0], 0.05)
        plain = r2.rolling(252).cov(m).div(m.rolling(252).var(), axis=0)
        self.assertGreater(abs(plain.iloc[400, 0] - 0.5), abs(V.welch_beta(r2, m).iloc[400, 0] - 0.5))

    def test_causal(self):
        r, m = synth()
        t = 400
        for hl in (None, 126):
            win = 252 if hl is None else 300
            a = V.welch_beta(r, m, win, halflife=hl)
            r2, m2 = r.copy(), m.copy()
            r2.iloc[t + 1:] = np.random.default_rng(1).normal(0, 0.1, r2.iloc[t + 1:].shape)
            m2.iloc[t + 1:] = np.random.default_rng(2).normal(0, 0.1, len(m2.iloc[t + 1:]))
            b = V.welch_beta(r2, m2, win, halflife=hl)
            np.testing.assert_array_equal(a.iloc[:t + 1].to_numpy(), b.iloc[:t + 1].to_numpy())
            self.assertFalse(np.allclose(a.iloc[t + 5:].to_numpy(), b.iloc[t + 5:].to_numpy(), equal_nan=True))


class SelectionTests(unittest.TestCase):
    def test_weight_uses_validation_only(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=300)
        lin = y + rng.normal(0, 0.2, 300)   # good
        gbm = y + rng.normal(0, 1.0, 300)   # bad
        w = V.choose_weight(y, lin, gbm)
        self.assertGreaterEqual(w, 0.75)
        # the function takes no test data; permuting anything else cannot change it
        self.assertEqual(w, V.choose_weight(y.copy(), lin.copy(), gbm.copy()))
        # swapped roles flip the choice
        self.assertLessEqual(V.choose_weight(y, gbm, lin), 0.25)

    def test_ties_pick_first(self):
        y = np.arange(5.0)
        self.assertEqual(V.choose_weight(y, y, y), 0.0)

    def test_bootstrap_brackets_and_deterministic(self):
        rng = np.random.default_rng(3)
        g = np.repeat(np.arange(40), 5)
        y = rng.normal(size=200)
        pa = y + rng.normal(0, 0.3, 200)
        pb = y + rng.normal(0, 0.6, 200)
        lo, hi = V.bootstrap_delta_r2(y, pa, pb, g)
        self.assertEqual((lo, hi), V.bootstrap_delta_r2(y, pa, pb, g))
        self.assertLess(lo, V.r2(y, pa) - V.r2(y, pb))
        self.assertGreater(hi, V.r2(y, pa) - V.r2(y, pb))
        self.assertGreater(lo, 0)


if __name__ == "__main__":
    unittest.main()
