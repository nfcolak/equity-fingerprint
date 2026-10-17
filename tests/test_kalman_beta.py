import unittest

import torch  # noqa: F401  (must precede lightgbm, imported via forward_kalman -> forward_v7 -> models)

import numpy as np
import pandas as pd

from equity_fingerprint import kalman_beta as K


def synth(n=900, betas=None, seed=0, noise=0.004):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-01", periods=n)
    m = pd.Series(rng.normal(0, 0.01, n), index=idx)
    if betas is None:
        betas = np.full((n, 2), [0.6, 1.3])
    r = pd.DataFrame(betas * m.to_numpy()[:, None] + rng.normal(0, noise, (n, betas.shape[1])), index=idx,
                     columns=["A", "B"])
    return r, m


class KalmanTests(unittest.TestCase):
    def test_constant_beta_converges(self):
        r, m = synth()
        beta, sd, _ = K.kalman_filter(r, m, 1e-5, float(m.std()))[:3]
        self.assertTrue(beta.iloc[:251].isna().all().all())
        self.assertTrue(np.isfinite(beta.iloc[251]).all())
        last = beta.iloc[-1].to_numpy()
        np.testing.assert_allclose(last, [0.6, 1.3], atol=0.1)
        self.assertTrue((sd.iloc[-1] > 0).all())

    def test_step_change_tracked(self):
        n = 1200
        b = np.full((n, 2), [0.5, 1.5]); b[800:, 0] = 1.5; b[800:, 1] = 0.5
        r, m = synth(n, b, seed=1, noise=0.003)
        beta = K.kalman_filter(r, m, 1e-3, float(m.std()))[0]
        pre, post = beta.iloc[790].to_numpy(), beta.iloc[-1].to_numpy()
        np.testing.assert_allclose(pre, [0.5, 1.5], atol=0.15)
        np.testing.assert_allclose(post, [1.5, 0.5], atol=0.2)
        # a rolling-252 OLS is still far from the new beta shortly after the step; the filter has moved on
        self.assertGreater(beta.iloc[900, 0], 1.0)

    def test_causality_bit_identical(self):
        r, m = synth()
        cut = 600
        r2, m2 = r.copy(), m.copy()
        r2.iloc[cut + 1:] += 0.05
        m2.iloc[cut + 1:] *= 3
        scale = float(m.iloc[:300].std())
        a = K.kalman_filter(r, m, 1e-4, scale)
        b = K.kalman_filter(r2, m2, 1e-4, scale)
        for x, y in zip(a[:2], b[:2]):
            np.testing.assert_array_equal(x.iloc[:cut + 1].to_numpy(), y.iloc[:cut + 1].to_numpy())

    def test_missing_days_prediction_only(self):
        r, m = synth()
        r.iloc[500:510, 0] = np.nan
        beta, sd, ll = K.kalman_filter(r, m, 1e-4, float(m.std()))
        self.assertTrue(ll.iloc[500:510, 0].isna().all())
        self.assertTrue(np.isfinite(beta.iloc[500:510, 0]).all())
        self.assertTrue((beta.iloc[500:510, 0] == beta.iloc[499, 0]).all())  # state carried
        self.assertGreater(sd.iloc[509, 0], sd.iloc[499, 0])  # uncertainty grows without updates

    def test_q_selection_uses_only_train_period(self):
        r, m = synth(900)
        end = str(r.index[600].date())
        q0, s0 = K.select_q(r, m, K.Q_GRID, end)
        r2, m2 = r.copy(), m.copy()
        r2.iloc[601:] = r2.iloc[601:] * -5 + 1.0  # wild post-period data
        m2.iloc[601:] = m2.iloc[601:] * 10
        q1, s1 = K.select_q(r2, m2, K.Q_GRID, end)
        self.assertEqual(q0, q1)
        self.assertEqual(s0, s1)
        self.assertIn(q0, K.Q_GRID)

    def test_kalman_beta_wrapper_uses_given_or_selected_q(self):
        r, m = synth(700)
        beta, sd, q, scores = K.kalman_beta(r, m, train_end=str(r.index[500].date()))
        self.assertIn(q, K.Q_GRID)
        self.assertEqual(set(scores), set(K.Q_GRID))
        _, _, q2, sc2 = K.kalman_beta(r, m, q=1e-3)
        self.assertEqual(q2, 1e-3)
        self.assertEqual(sc2, {})


if __name__ == "__main__":
    unittest.main()
