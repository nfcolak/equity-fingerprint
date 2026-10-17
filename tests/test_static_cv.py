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
        X = pd.DataFrame(rng.normal(size=(60, 4)), columns=["a", "b", "c", "log_tv_21"])
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

        X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "log_tv_21"])
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

    # ------------------------------------------------------------------ v7 additions
    def _mats(self):
        rng = np.random.default_rng(5)
        mkt = rng.normal(size=len(self.ret)) * 0.01
        beta = rng.uniform(0.5, 1.5, self.ret.shape[1])
        ret = self.ret + np.outer(mkt, beta)
        ret.iloc[:50, 3] = np.nan
        return ret, pd.Series(mkt, index=ret.index)

    def test_residualisation_removes_market_beta(self):
        ret, mkt = self._mats()
        res = sc.resid_on(ret, mkt.to_numpy())
        self.assertTrue(res.iloc[:50, 3].isna().all())
        for j in range(ret.shape[1]):
            ok = res.iloc[:, j].notna().to_numpy()
            self.assertAlmostEqual(np.polyfit(mkt.to_numpy()[ok], res.iloc[ok, j].to_numpy(), 1)[0], 0.0, places=8)
            self.assertAlmostEqual(res.iloc[ok, j].mean(), 0.0, places=10)
        mats = sc.build_matrices(ret, mkt)
        self.assertEqual(list(mats), sc.MATRIX_NAMES)
        # raw correlations are dominated by the common market; residual ones keep the planted clusters
        same = self.y[:, None] == self.y[None, :]
        off = ~np.eye(len(self.y), dtype=bool)
        self.assertGreater(np.nanmean(mats["raw"][~same & off]), 0.2)
        self.assertLess(abs(np.nanmean(mats["resid"][~same & off])), 0.1)
        w = sc.winsorise_sd(ret.assign(S00=ret["S00"].where(np.arange(len(ret)) != 7, 50.0)))
        self.assertLessEqual(w["S00"].abs().max(), 4.0 * ret["S00"].where(np.arange(len(ret)) != 7, 50.0).std() + 1e-12)

    def test_permuted_test_labels_do_not_change_new_knn(self):
        ret, mkt = self._mats()
        mats = sc.build_matrices(ret, mkt)
        n = len(self.y)
        rng = np.random.default_rng(3)
        test = np.sort(rng.choice(n, 10, replace=False))
        train = np.setdiff1d(np.arange(n), test)
        y2 = self.y.copy()
        y2[test] = (y2[test] + 1) % NC
        sel_a, sel_b = [], []
        a = sc.sector_fold(mats["raw"], self.y, self.fe, train, test, NC, mats, sel_a)
        b = sc.sector_fold(mats["raw"], y2, self.fe, train, test, NC, mats, sel_b)
        for m in ("knn_resid", "knn_auto", "knn"):
            np.testing.assert_array_equal(a[m], b[m], err_msg=m)
        self.assertEqual(sel_a, sel_b)
        self.assertIn(sel_a[0][0], sc.MATRIX_NAMES)
        oof = sc.run_sector_repeat(mats["raw"], self.y, self.fe, NC, 0, mats)
        self.assertGreater(sc.class_metrics(self.y, oof["knn_resid"], NC)["macro_f1"], 0.8)
        self.assertGreater(sc.class_metrics(self.y, oof["knn_auto"], NC)["macro_f1"], 0.8)

    def test_fast_metrics_match_sklearn(self):
        rng = np.random.default_rng(0)
        t, q = rng.integers(0, NC, 80), rng.integers(0, NC, 80)
        a, b = sc.class_metrics(t, q, NC), sc.fast_class_metrics(t, q, NC)
        for k in a:
            self.assertAlmostEqual(a[k], b[k], places=10)
        u, v = rng.normal(size=50), rng.normal(size=50)
        a, b = sc.reg_metrics(u, v), sc.fast_reg_metrics(u, v)
        for k in a:
            self.assertAlmostEqual(a[k], b[k], places=10)

    def test_intervals_bracket_the_mean(self):
        oof = [sc.run_sector_repeat(self.C, self.y, self.fe, NC, r) for r in range(3)]
        rows = sc.summarize("sector", ["knn", "baseline"], lambda t, q: sc.class_metrics(t, q, NC),
                            lambda t, ps: sc.bootstrap_ci_mean(t, ps, lambda a, b: sc.fast_class_metrics(a, b, NC), n_boot=200),
                            self.y, oof, len(self.y), NC)
        self.assertEqual(len(rows), 8)
        for r in rows:
            self.assertTrue(np.isfinite(r["ci_lo"]) and np.isfinite(r["ci_hi"]), r)
            self.assertLessEqual(r["ci_lo"], r["mean"] + 1e-12, r)
            self.assertGreaterEqual(r["ci_hi"], r["mean"] - 1e-12, r)
        rng = np.random.default_rng(2)
        yt = rng.normal(size=60)
        preds = [yt + rng.normal(size=60) * 0.3 for _ in range(4)]
        ci = sc.bootstrap_ci_mean(yt, preds, sc.fast_reg_metrics, n_boot=300)
        mean_r2 = np.mean([sc.fast_reg_metrics(yt, p)["r2"] for p in preds])
        self.assertLess(ci["r2"][0], mean_r2)
        self.assertGreater(ci["r2"][1], mean_r2)

    def test_mcap_has_logtv21_model(self):
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(size=(60, 4)), columns=["a", "b", "c", "log_tv_21"])
        yy = X["log_tv_21"].to_numpy() * 2 + rng.normal(size=60) * 0.1
        oof = sc.run_mcap_repeat(X, yy, 0)
        self.assertIn("ridge_logtv21", oof)
        self.assertGreater(sc.reg_metrics(yy, oof["ridge_logtv21"])["r2"], 0.9)

    def test_oof_numeric_columns(self):
        parts = sc._oof_rows("log_mcap", ["ridge"], ["A", "B"], np.array([1.5, 2.5]), [{"ridge": np.array([1.0, 2.0])}], True)
        d = parts[0]
        self.assertEqual(d["y_true_num"].dtype, float)
        self.assertEqual(list(d["y_pred_num"]), [1.0, 2.0])
        parts = sc._oof_rows("sector", ["knn"], ["A", "B"], ["x", "y"], [{"knn": ["x", "x"]}], False)
        self.assertTrue(parts[0]["y_true_num"].isna().all())


if __name__ == "__main__":
    unittest.main()
