"""Synthetic tests for static_v9_mcap: multi-date conversion, no test-label use, reproduction of static_cv,
paired deltas, TabPFN merge (stub OOF file), schema."""
import unittest

# torch must be loaded before lightgbm (imported via static_cv -> models): the reverse order makes libomp hang.
import torch  # noqa: F401

import tempfile
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd

from equity_fingerprint import static_cv as sc
from equity_fingerprint import static_v9_mcap as S
from equity_fingerprint import tabpfn_mcap as T
from equity_fingerprint.features import FE_BASE_FEATURES, Panel, base_features, month_end_positions
from equity_fingerprint.models import Prep
from equity_fingerprint.targets import log_mcap_target

def synth_panel(n_days=1000, n=30, seed=0, start="2015-01-01"):
    """Small random-walk panel with every attribute base_features / ext_features / log_mcap_target read."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    cols = [f"S{i:02d}" for i in range(n)]
    mkt = pd.Series(rng.normal(0, 0.008, n_days), index=idx)
    smi = pd.Series(0.8 * mkt.to_numpy() + rng.normal(0, 0.004, n_days), index=idx)
    beta = rng.uniform(0.4, 1.6, n)
    ret = pd.DataFrame(np.outer(mkt, beta) + rng.normal(0, 0.012, (n_days, n)), index=idx, columns=cols)
    ret.iloc[0] = np.nan
    adj = np.exp(ret.fillna(0).cumsum()) * rng.uniform(20, 200, n)
    close = adj
    opn = close * (1 + rng.normal(0, 0.003, (n_days, n)))
    high = np.maximum(close, opn) * (1 + rng.uniform(0, 0.01, (n_days, n)))
    low = np.minimum(close, opn) * (1 - rng.uniform(0, 0.01, (n_days, n)))
    vol = pd.DataFrame(np.exp(rng.normal(11, 0.6, (n_days, n)) + rng.normal(0, 1, n)), index=idx, columns=cols)
    span = pd.DataFrame(True, index=idx, columns=cols)
    info = pd.DataFrame({"shares_outstanding": np.exp(rng.normal(17, 1, n))}, index=cols)
    return Panel(calendar=idx, tickers=cols, adj=adj, close=close, open=opn, high=high, low=low, volume=vol,
                 tv=close * vol, ret=ret, mkt=mkt, rows=span.cumsum().where(span), span=span, info=info, smi=smi)


class StubReg:
    def __init__(self, repeat):
        pass

    def fit(self, X, y):
        self.m = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.m)


def const_rows(n=10, lags=3, drop=()):
    """Hand-made multi-date rows: uninformative feature, per-lag price shift s_k, all stocks the same."""
    g = np.repeat(np.arange(n), lags)
    lag = np.tile(np.arange(lags), n)
    shift = 0.15 * lag * (1 + 0.5 * lag)
    keep = np.array([(gi, li) not in drop for gi, li in zip(g, lag)])
    X = pd.DataFrame({"a": np.zeros(len(g))})
    return S.MDRows(g[keep], lag[keep], X[keep].reset_index(drop=True), shift[keep])


class MultiDateTests(unittest.TestCase):
    def test_constant_price_ratio_recovers_label(self):
        # all stocks share log mcap Y and the same price path: y_t = Y - shift_t, features carry no signal,
        # so the intercept is mean(Y - shift) and prediction + shift averaged over dates is exactly Y
        Y = 21.5
        rows = const_rows(10, 3, drop={(7, 1)})                # test stock 7 lacks one date: fewer rows
        y = np.full(10, Y)
        tr, te = np.arange(0, 6), np.arange(6, 10)
        s = 0.15 * np.arange(3) * (1 + 0.5 * np.arange(3))     # per-lag shift
        for pr in S._md_predict(rows, y, tr, te, ["a"], 3, 1).values():
            np.testing.assert_allclose(pr[[0, 2, 3]], Y, atol=1e-9)
            # intercept = Y - mean(s) (train stocks complete); stock 7 averages only its lags 0 and 2
            np.testing.assert_allclose(pr[1], Y - s.mean() + (s[0] + s[2]) / 2, atol=1e-9)
        # K=1 uses only lag 0 (shift 0); (K=2, step=2) uses lags 0 and 2: complete for every stock
        for k, st in ((1, 1), (2, 2)):
            for pr in S._md_predict(rows, y, tr, te, ["a"], k, st).values():
                np.testing.assert_allclose(pr, Y, atol=1e-9)

    def test_row_selection_by_k_and_step(self):
        rows = const_rows(4, 6)
        sel = lambda k, s: sorted(set(rows.lag[np.isin(rows.lag, [s * j for j in range(k)])]))
        self.assertEqual(sel(3, 1), [0, 1, 2])
        self.assertEqual(sel(2, 3), [0, 3])

    def test_md_fold_constant_case_and_selection_tie_rule(self):
        rows = const_rows(20, 6)
        y = np.full(20, 3.0)
        pred, best = S.md_fold(rows, y, np.arange(15), np.arange(15, 20), ["a"])
        np.testing.assert_allclose(pred, 3.0, atol=1e-9)
        self.assertEqual(best[:2], (1, 1))                      # all configs tie at ~0 error -> smaller K

    def test_rows_from_panel_match_log_mcap_target_at_each_date(self):
        p = synth_panel()
        tick, y, Xb, Xe, _ = T.mcap_inputs_v9(p)
        rows = S.build_md_rows(p, tick)
        allpos = month_end_positions(p)
        self.assertEqual(sorted(set(rows.lag)), list(range(S.MD_MAX_LAG + 1)))
        self.assertEqual(list(rows.X.columns), T.FE_EXT)
        for k in (0, 1, 7, 15):
            pos = allpos[-1 - k]
            b = base_features(p, np.array([pos]))
            yt = log_mcap_target(p, b)
            ytm = dict(zip(b["ticker"], yt))
            m = rows.lag == k
            for gi, sh in zip(rows.g[m], rows.shift[m]):
                self.assertAlmostEqual(y[gi] - sh, ytm[tick[gi]], places=9)
        self.assertTrue((rows.shift[rows.lag == 0] == 0).all())
        self.assertTrue((rows.shift[rows.lag > 0] != 0).any())


class FoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = synth_panel()
        cls.tick, cls.y, cls.Xb, cls.Xe, _ = T.mcap_inputs_v9(cls.p)
        cls.rows = S.build_md_rows(cls.p, cls.tick)
        cls.tr, cls.te = T.outer_folds(len(cls.y), 0)[0]

    def test_inputs_match_static_cv(self):
        t0, y0, X0, d0 = sc.mcap_inputs(self.p)
        self.assertEqual(t0, self.tick)
        np.testing.assert_array_equal(y0, self.y)
        pd.testing.assert_frame_equal(X0, self.Xb)

    def test_linear_prep_copy_equals_models_prep(self):
        a = Prep(FE_BASE_FEATURES, linear=True).fit(self.Xb.iloc[:20]).transform(self.Xb).to_numpy()
        b = T.LinearPrep(FE_BASE_FEATURES).fit(self.Xb.iloc[:20]).transform(self.Xb)
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_ridge_and_lgbm_reproduce_static_cv(self):
        ref = sc.mcap_fold(self.Xb, self.y, self.tr, self.te)
        got = S.pair_fold(self.Xb, self.y, self.tr, self.te)
        np.testing.assert_allclose(got["ridge"], ref["ridge"], atol=1e-9)
        np.testing.assert_allclose(got["lgbm"], ref["lgbm"], atol=1e-9)
        lo, hi = np.minimum(got["ridge"], got["lgbm"]), np.maximum(got["ridge"], got["lgbm"])
        self.assertTrue(((got["avg"] >= lo - 1e-9) & (got["avg"] <= hi + 1e-9)).all())

    def test_garbage_test_labels_leave_predictions_unchanged(self):
        y2 = self.y.copy()
        y2[self.te] = np.random.default_rng(1).normal(100, 50, len(self.te))
        for X in (self.Xb, self.Xe):
            a, b = S.pair_fold(X, self.y, self.tr, self.te), S.pair_fold(X, y2, self.tr, self.te)
            for k in a:
                np.testing.assert_array_equal(a[k], b[k], err_msg=k)
        for cols in (FE_BASE_FEATURES, T.FE_EXT):
            a, sa = S.md_fold(self.rows, self.y, self.tr, self.te, cols)
            b, sb = S.md_fold(self.rows, y2, self.tr, self.te, cols)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(sa, sb)


def fast_lgbm(g):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=15, learning_rate=0.2, verbose=-1, random_state=42, n_jobs=1, **g)


class RunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._patch = mock.patch.object(S, "_lgbm", fast_lgbm)   # fewer trees: speed only (ridge arms unaffected)
        cls._patch.start()
        cls.p = synth_panel()
        cls.tick, cls.y, cls.Xb, cls.Xe, _ = T.mcap_inputs_v9(cls.p)

    @classmethod
    def tearDownClass(cls):
        cls._patch.stop()

    def _ref_csv(self, d, repeats):
        reps = [sc.run_mcap_repeat(self.Xb, self.y, r) for r in range(repeats)]
        rows = sc.summarize("log_mcap", ["ridge"], sc.reg_metrics,
                            lambda t, ps: sc.bootstrap_ci_mean(t, ps, sc.fast_reg_metrics, n_boot=5),
                            self.y, reps, len(self.y), np.nan)
        f = Path(d) / "ref_static_cv.csv"
        pd.DataFrame(rows).to_csv(f, index=False)
        return f

    def test_run_schema_deltas_reproduction_and_tabpfn_skip(self):
        with tempfile.TemporaryDirectory() as d:
            ref = self._ref_csv(d, 1)
            res, ok = S.run(repeats=1, out_dir=d, panel=self.p, ref_csv=ref, n_boot=30)
            self.assertTrue(ok)
            self.assertEqual(list(res.columns), ["target", "model", "metric", "mean", "sd", "ci_lo", "ci_hi",
                                                 "n_stocks", "n_repeats", "n_classes"])
            self.assertEqual(list(dict.fromkeys(res["model"])), S.ARMS_STATIC)
            for a in S.ARMS_STATIC[1:]:
                self.assertEqual(set(res[res.model == a]["metric"]),
                                 {"r2", "mae", "delta_r2_vs_ridge", "delta_mae_vs_ridge"})
            self.assertEqual(set(res[res.model == "ridge"]["metric"]), {"r2", "mae"})
            self.assertTrue((res["n_stocks"] == len(self.tick)).all())
            r2 = res.set_index(["model", "metric"])["mean"]
            self.assertAlmostEqual(r2[("lgbm", "delta_r2_vs_ridge")], r2[("lgbm", "r2")] - r2[("ridge", "r2")], places=9)
            self.assertAlmostEqual(r2[("lgbm", "delta_mae_vs_ridge")], r2[("lgbm", "mae")] - r2[("ridge", "mae")], places=9)
            dl = res[res.metric.str.startswith("delta")]
            self.assertTrue((dl.ci_lo <= dl.ci_hi).all())
            oof = pd.read_parquet(Path(d) / "static_v9_mcap_oof.parquet")
            self.assertEqual(set(oof["model"]), set(S.ARMS_STATIC))
            self.assertTrue(np.isfinite(oof["y_pred_num"]).all())
            self.assertEqual(len(oof), len(S.ARMS_STATIC) * len(self.tick))
            self.assertTrue((Path(d) / "static_v9_mcap.csv").exists())
            self.assertFalse(np.isnan(res["mean"]).any())

    def test_tabpfn_arms_appear_when_oof_file_exists(self):
        with tempfile.TemporaryDirectory() as d:
            T.tabpfn_oof(self.Xb, self.Xe, self.y, self.tick, 1, StubReg).to_parquet(Path(d) / T.TABPFN_FILE)
            res, ok = S.run(repeats=1, out_dir=d, panel=self.p, ref_csv=Path(d) / "missing.csv", n_boot=20)
            self.assertIsNone(ok)
            self.assertEqual(list(dict.fromkeys(res["model"])), S.ARMS_STATIC + S.ARMS_TABPFN)
            oof = pd.read_parquet(Path(d) / "static_v9_mcap_oof.parquet")
            a = oof[oof.model == "avg_ridge_md_ext_tabpfn_ext"].set_index("ticker")["y_pred_num"]
            b = oof[oof.model == "ridge_md_ext"].set_index("ticker")["y_pred_num"]
            c = oof[oof.model == "tabpfn_ext"].set_index("ticker")["y_pred_num"]
            np.testing.assert_allclose(a, 0.5 * b.reindex(a.index) + 0.5 * c.reindex(a.index), atol=1e-12)

    def test_merge_rejects_incomplete_file(self):
        with tempfile.TemporaryDirectory() as d:
            df = T.tabpfn_oof(self.Xb, self.Xe, self.y, self.tick, 1, StubReg)
            df.iloc[1:].to_parquet(Path(d) / T.TABPFN_FILE)
            with self.assertRaises(ValueError):
                S.merge_tabpfn([{"ridge_md_ext": np.zeros(len(self.tick))}], self.tick, Path(d) / T.TABPFN_FILE)

    def test_bootstrap_delta_pairs_resamples(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=60)
        a = [y + rng.normal(0, 0.3, 60) for _ in range(3)]
        ci = S.bootstrap_ci_delta(y, a, a, n_boot=40)
        for k, (lo, hi) in ci.items():
            self.assertEqual((lo, hi), (0.0, 0.0), k)           # identical arms -> zero difference in every resample
        b = [y + rng.normal(0, 1.0, 60) for _ in range(3)]
        lo, hi = S.bootstrap_ci_delta(y, a, b, n_boot=100)["r2"]
        self.assertGreater(lo, 0.0)
        self.assertEqual(S.bootstrap_ci_delta(y, a, b, n_boot=100), S.bootstrap_ci_delta(y, a, b, n_boot=100))


if __name__ == "__main__":
    unittest.main()
