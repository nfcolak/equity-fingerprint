"""Synthetic tests for static_v9_sector: a=0 equals static_cv, EW weights, no test-label use, schema."""
import torch  # noqa: F401  (macOS libomp: torch before lightgbm)
import unittest

import numpy as np
import pandas as pd

from equity_fingerprint import static_cv as sc
from equity_fingerprint import static_v9_sector as v9

NC, PER = 3, 14


def make_data(seed=0, T=300):
    rng = np.random.default_rng(seed)
    n = NC * PER
    y = np.repeat(np.arange(NC), PER)
    fac = rng.normal(size=(T, NC))
    mkt = rng.normal(size=T) * 0.01
    ret = fac[:, y] * 0.01 + rng.normal(size=(T, n)) * 0.01 + np.outer(mkt, rng.uniform(0.5, 1.5, n))
    ret = pd.DataFrame(ret, columns=[f"S{i:02d}" for i in range(n)])
    return ret, pd.Series(mkt, index=ret.index), y


class TestStaticV9Sector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ret, cls.mkt, cls.y = make_data()
        cls.mats = v9.build_matrices_v9(cls.ret, cls.mkt)

    def test_matrix_names(self):
        self.assertEqual(list(self.mats), v9.MATRIX_V9)
        self.assertEqual(v9.MATRIX_V9, sc.MATRIX_NAMES + ["resid_hl2y", "resid_hl5y", "resid_w4_hl5y"])

    def test_prior_a0_equals_knn_predict(self):
        C = self.mats["resid"]
        idx = np.arange(len(self.y))
        tr, te = idx[idx % 3 != 0], idx[idx % 3 == 0]
        S = sc._masked_sims(C, tr, te)
        prior = np.bincount(self.y[tr], minlength=NC) / len(tr)
        maj = int(np.bincount(self.y[tr], minlength=NC).argmax())
        for k in sc.KNN_KS:
            np.testing.assert_array_equal(v9.knn_predict_prior(S, self.y[tr], k, NC, maj, prior, 0),
                                          sc.knn_predict(S, self.y[tr], k, NC, maj))
        S2 = S.copy()
        S2[0, :] = np.nan  # no neighbours -> majority in both
        np.testing.assert_array_equal(v9.knn_predict_prior(S2, self.y[tr], 5, NC, maj, prior, 0),
                                      sc.knn_predict(S2, self.y[tr], 5, NC, maj))
        scan = v9.scan_predictions(S2, self.y[tr], NC, maj)
        np.testing.assert_array_equal(scan[(7, 0)], sc.knn_predict(S2, self.y[tr], 7, NC, maj))

    def test_prior_correction_and_zero_prior(self):
        S = np.array([[0.9, 0.5, 0.5, 0.5]])
        y_ref = np.array([0, 1, 1, 1])
        prior = np.array([0.25, 0.75, 0.0])
        self.assertEqual(v9.knn_predict_prior(S, y_ref, 4, 3, 2, prior, 0)[0], 1)  # 1.5 votes vs 0.9
        self.assertEqual(v9.knn_predict_prior(S, y_ref, 4, 3, 2, prior, 1.0)[0], 0)  # 3.6 vs 2.0
        prior0 = np.array([0.0, 0.75, 0.0])  # class 0 has no prior -> its votes are dropped
        self.assertEqual(v9.knn_predict_prior(S, y_ref, 4, 3, 2, prior0, 1.0)[0], 1)

    def test_ew_weights_with_huge_half_life_equal_plain_residual(self):
        res = sc.resid_on(self.ret, self.mkt.to_numpy())
        n = len(res)
        w = v9.ew_weights(n, 1e12)
        np.testing.assert_allclose(w, 1.0, atol=1e-9)
        np.testing.assert_allclose(sc.pairwise_corr(res * w).to_numpy(), self.mats["resid"], atol=1e-9, equal_nan=True)
        w2 = v9.ew_weights(n, 100)
        self.assertAlmostEqual(float(w2[-1, 0]), 1.0)
        self.assertAlmostEqual(float(w2[-1 - 100, 0]), 0.5 ** 0.5)
        self.assertFalse(np.allclose(self.mats["resid_hl2y"], self.mats["resid"], equal_nan=True))

    def test_knn_auto_arm_reproduces_static_cv(self):
        six = {m: self.mats[m] for m in sc.MATRIX_NAMES}
        fe = pd.DataFrame(np.zeros((len(self.y), 2)), columns=["a", "b"])
        for r in (0,):
            ref = sc.run_sector_repeat(six["raw"], self.y, fe, NC, r, six)["knn_auto"]
            new = v9.run_sector_repeat_v9(self.mats, self.y, NC, r)
            np.testing.assert_array_equal(new["knn_auto"], ref)
            self.assertTrue(all((v >= 0).all() for v in new.values()))

    def test_test_fold_labels_do_not_change_predictions(self):
        n = len(self.y)
        rng = np.random.default_rng(3)
        test = np.sort(rng.choice(n, 9, replace=False))
        train = np.setdiff1d(np.arange(n), test)
        y2 = self.y.copy()
        y2[test] = (y2[test] + 1) % NC
        s1, s2 = [], []
        a = v9.sector_fold_v9(self.mats, self.y, train, test, NC, s1)
        b = v9.sector_fold_v9(self.mats, y2, train, test, NC, s2)
        for arm in v9.ARMS:
            np.testing.assert_array_equal(a[arm], b[arm], err_msg=arm)
        self.assertEqual(s1, s2)
        self.assertEqual(len(s1), 4)
        for arm, mn, k, al in s1:
            names, use_a = v9.ARM_SPEC[arm]
            self.assertIn(mn, names)
            self.assertIn(k, sc.KNN_KS)
            if not use_a:
                self.assertEqual(al, 0)

    def test_schema_deltas_and_selection(self):
        res, sel = v9.run_experiment(self.mats, self.y, NC, repeats=2, n_boot=50)
        self.assertEqual(list(res.columns), ["target", "model", "metric", "mean", "sd", "ci_lo", "ci_hi",
                                             "n_stocks", "n_repeats", "n_classes"])
        self.assertEqual(list(sel.columns), ["arm", "matrix", "k", "a", "count"])
        self.assertEqual(sel["count"].sum(), 2 * 5 * 4)
        base = res[~res.metric.str.startswith("delta_")]
        self.assertEqual(len(base), 4 * 4)
        d = res[res.metric.str.startswith("delta_")]
        self.assertEqual(set(d.metric), {f"delta_{m}_vs_knn_auto" for m in v9.CLASS_METRICS})
        self.assertEqual(set(d.model), set(v9.ARMS) - {"knn_auto"})
        m = res.set_index(["model", "metric"])["mean"]
        for arm in d.model.unique():
            self.assertAlmostEqual(m[(arm, "delta_macro_f1_vs_knn_auto")],
                                   m[(arm, "macro_f1")] - m[("knn_auto", "macro_f1")], places=12)
        for _, r in d.iterrows():
            self.assertLessEqual(r.ci_lo, r.ci_hi)
        ref = res[res.model == "knn_auto"].assign(n_repeats=2)
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as t:
            f = pathlib.Path(t) / "static_cv.csv"
            ref.to_csv(f, index=False)
            self.assertLess(v9.reproduction_check(res, f), 1e-9)

    def test_delta_ci_uses_same_resample(self):
        truth = np.arange(40) % 3
        p = [truth.copy()]
        ci = v9.bootstrap_ci_delta(truth, p, p, lambda a, b: sc.fast_class_metrics(a, b, 3), n_boot=30)
        for lo, hi in ci.values():
            self.assertEqual((lo, hi), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
