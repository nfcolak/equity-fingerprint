"""Synthetic tests for static_cv: planted clusters, once-per-repeat OOF, no test-label use, no self reference."""
import unittest

import numpy as np
import pandas as pd

from equity_fingerprint import static_cv as sc

NC, PER = 4, 12


def make_data(seed=0, T=400):
    rng = np.random.default_rng(seed)
    n = NC * PER
    y = np.repeat(np.arange(NC), PER)
    fac = rng.normal(size=(T, NC))
    ret = fac[:, y] * 0.01 + rng.normal(size=(T, n)) * 0.01
    tick = [f"S{i:02d}" for i in range(n)]
    ret = pd.DataFrame(ret, columns=tick)
    fe = pd.DataFrame(rng.normal(size=(n, 3)) + y[:, None] * 0.5, index=tick, columns=["a", "b", "c"])
    C = sc.pairwise_corr(ret).to_numpy()
    return ret, y, C, fe


class TestStaticCV(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ret, cls.y, cls.C, cls.fe = make_data()

    def test_corr_matches_pandas_and_min_days(self):
        ref = self.ret.corr().to_numpy().copy()
        np.fill_diagonal(ref, np.nan)
        np.testing.assert_allclose(self.C, ref, atol=1e-9, equal_nan=True)
        r = self.ret.copy()
        r.iloc[:300, 0] = np.nan
        c = sc.pairwise_corr(r, min_common=252)
        self.assertTrue(c.iloc[0].isna().all())
        self.assertTrue(np.isfinite(c.iloc[1, 2]))

    def test_knn_recovers_planted_classes(self):
        oof = sc.run_sector_repeat(self.C, self.y, self.fe, NC, 0)
        self.assertGreater(sc.class_metrics(self.y, oof["knn"], NC)["macro_f1"], 0.9)
        self.assertGreater(sc.class_metrics(self.y, oof["prof_logit"], NC)["macro_f1"], 0.8)
        base = sc.class_metrics(self.y, oof["baseline"], NC)["macro_f1"]
        self.assertLess(base, 0.2)

    def test_each_stock_predicted_once_per_repeat(self):
        for r in (0, 1):
            oof = sc.run_sector_repeat(self.C, self.y, self.fe, NC, r)
            for m, v in oof.items():
                self.assertEqual(len(v), len(self.y))
                self.assertTrue((v >= 0).all(), m)  # -1 sentinel would mean never predicted
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))
        yy = X["a"].to_numpy() * 2 + rng.normal(size=60) * 0.1
        oof = sc.run_mcap_repeat(X, yy, 0)
        for m, v in oof.items():
            self.assertTrue(np.isfinite(v).all(), m)

    def test_permuted_test_labels_do_not_change_predictions(self):
        n = len(self.y)
        rng = np.random.default_rng(3)
        test = np.sort(rng.choice(n, 10, replace=False))
        train = np.setdiff1d(np.arange(n), test)
        y2 = self.y.copy()
        y2[test] = rng.permutation(y2[test])
        y2[test] = (y2[test] + 1) % NC  # guarantee a change
        a = sc.sector_fold(self.C, self.y, self.fe, train, test, NC)
        b = sc.sector_fold(self.C, y2, self.fe, train, test, NC)
        for m in a:
            np.testing.assert_array_equal(a[m], b[m], err_msg=m)

        X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
        yr = X["a"].to_numpy() + rng.normal(size=n) * 0.1
        yr2 = yr.copy()
        yr2[test] = rng.normal(size=len(test)) * 10
        ra, rb = sc.mcap_fold(X, yr, train, test), sc.mcap_fold(X, yr2, train, test)
        for m in ra:
            np.testing.assert_allclose(ra[m], rb[m], err_msg=m)

    def test_no_self_reference_in_profile(self):
        idx = np.arange(len(self.y))
        P = sc.profile_features(self.C, self.y, idx, idx, NC)
        # the diagonal is NaN in C; poisoning it with a perfect self-correlation must change nothing
        C2 = self.C.copy()
        np.fill_diagonal(C2, 1.0)
        np.testing.assert_allclose(sc.profile_features(C2, self.y, idx, idx, NC), P)
        # own-class top-m mean with self would be higher; check against a manual leave-one-out
        i = 5
        same = [j for j in idx if self.y[j] == self.y[i] and j != i]
        top = np.sort(self.C[i, same])[::-1][:sc.PROF_M].mean()
        self.assertAlmostEqual(P[i, self.y[i]], top)
        # kNN: with k=1 on a reference set containing the stock, prediction uses another stock
        S = sc._masked_sims(self.C, idx, idx)
        self.assertTrue(np.isnan(np.diag(S)).all())


if __name__ == "__main__":
    unittest.main()
