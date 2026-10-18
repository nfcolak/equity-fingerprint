"""Synthetic tests for the issuer-grouped folds of static_cv / static_v9_mcap: groups=None is the old code path,
share lines of one issuer share every outer and inner fold, test-fold label garbage changes nothing."""
import unittest

# torch must be loaded before lightgbm (imported via static_cv -> models): the reverse order makes libomp hang.
import torch  # noqa: F401

from unittest import mock

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from equity_fingerprint import static_cv as sc
from equity_fingerprint import static_v9_mcap as S
from equity_fingerprint import tabpfn_mcap as T
from equity_fingerprint.features import FE_BASE_FEATURES

NC, PER = 4, 12


def make_sector(seed=0, T_=400):
    """4 planted clusters of 12 stocks; stocks 2k and 2k+1 (k < 24) are two share lines of one issuer."""
    rng = np.random.default_rng(seed)
    n = NC * PER
    y = np.repeat(np.arange(NC), PER)
    fac = rng.normal(size=(T_, NC))
    ret = fac[:, y] * 0.01 + rng.normal(size=(T_, n)) * 0.01
    tick = [f"S{i:02d}" for i in range(n)]
    ret = pd.DataFrame(ret, columns=tick)
    fe = pd.DataFrame(rng.normal(size=(n, 3)) + y[:, None] * 0.5, index=tick, columns=["a", "b", "c"])
    return y, sc.pairwise_corr(ret).to_numpy(), fe, np.arange(n) // 2


def fast_lgbm(g):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=15, learning_rate=0.2, verbose=-1, random_state=42, n_jobs=1, **g)


def make_mcap(n=40, seed=1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.lognormal(size=(n, len(T.FE_EXT))), columns=T.FE_EXT)
    y = np.log(X["log_tv_21"].to_numpy() + 1.0) * 2 + rng.normal(size=n) * 0.1
    g = np.repeat(np.arange(n // 2), 2)                               # pairs of share lines
    return X, y, g


def make_rows(X, y, lags=3):
    n = len(y)
    gi = np.repeat(np.arange(n), lags)
    lag = np.tile(np.arange(lags), n)
    Xr = X.iloc[gi].reset_index(drop=True)
    return S.MDRows(gi, lag, Xr, 0.1 * lag.astype(float))


class Recorder:
    """Wraps a split function and stores the (train, test) index arrays it returned."""

    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def __call__(self, *a, **k):
        out = self.fn(*a, **k)
        self.calls.append((a, k, out))
        return out


def assert_pairs_together(tc, folds, groups):
    for tr, te in folds:
        tc.assertFalse(set(groups[tr]) & set(groups[te]), "an issuer sits in both train and test")


class SectorGroupedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.y, cls.C, cls.fe, cls.g = make_sector()

    def test_inner_splits_default_is_stratified_kfold(self):
        y_tr = self.y[:40]
        want = list(StratifiedKFold(5, shuffle=True, random_state=sc.INNER_SEED).split(np.zeros(40), y_tr))
        got = sc._inner_splits(y_tr)
        for (a, b), (c, d) in zip(want, got):
            np.testing.assert_array_equal(a, c)
            np.testing.assert_array_equal(b, d)

    def test_groups_none_equals_old_path(self):
        for r in (0, 1):
            new = sc.run_sector_repeat(self.C, self.y, self.fe, NC, r)
            folds = StratifiedKFold(5, shuffle=True, random_state=r).split(np.zeros(len(self.y)), self.y)
            old = {m: np.full(len(self.y), -1) for m in sc.SECTOR_MODELS}
            for tr, te in folds:
                pred = sc.sector_fold(self.C, self.y, self.fe, tr, te, NC)
                for m in old:
                    old[m][te] = pred[m]
            for m in old:
                np.testing.assert_array_equal(new[m], old[m], err_msg=m)

    def test_share_lines_share_outer_and_inner_folds(self):
        outer, inner = Recorder(sc.sector_fold), Recorder(sc._inner_splits)
        with mock.patch.object(sc, "sector_fold", outer), mock.patch.object(sc, "_inner_splits", inner):
            for r in (0, 1, 2):
                oof = sc.run_sector_repeat(self.C, self.y, self.fe, NC, r, groups=self.g)
                for m, v in oof.items():
                    self.assertTrue((v >= 0).all(), m)            # every stock predicted exactly once
        self.assertEqual(len(outer.calls), 15)
        for a, _k, _ in outer.calls:
            tr, te = np.asarray(a[3]), np.asarray(a[4])
            self.assertFalse(set(self.g[tr]) & set(self.g[te]))
        self.assertTrue(inner.calls)
        for _a, k, splits in inner.calls:
            self.assertIsNotNone(k.get("groups"))
            gtr = np.asarray(k["groups"])
            assert_pairs_together(self, splits, gtr)

    def test_test_label_garbage_leaves_grouped_predictions_unchanged(self):
        n = len(self.y)
        folds = list(sc.StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(n), self.y, self.g))
        tr, te = folds[0]
        y2 = self.y.copy()
        y2[te] = (y2[te] + 1) % NC
        a = sc.sector_fold(self.C, self.y, self.fe, tr, te, NC, groups=self.g)
        b = sc.sector_fold(self.C, y2, self.fe, tr, te, NC, groups=self.g)
        for m in a:
            np.testing.assert_array_equal(a[m], b[m], err_msg=m)

    def test_grouped_recovers_planted_classes(self):
        oof = sc.run_sector_repeat(self.C, self.y, self.fe, NC, 0, groups=self.g)
        self.assertGreater(sc.class_metrics(self.y, oof["knn"], NC)["macro_f1"], 0.9)


class McapGroupedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.g = make_mcap()
        cls._patch = mock.patch.object(sc, "_lgbm", fast_lgbm)        # fewer trees: speed only
        cls._patch.start()

    @classmethod
    def tearDownClass(cls):
        cls._patch.stop()

    def test_groups_none_equals_old_path(self):
        Xc = self.X[["log_tv_21", FE_BASE_FEATURES[0], FE_BASE_FEATURES[1]]]
        new = sc.run_mcap_repeat(Xc, self.y, 1)
        old = {m: np.full(len(self.y), np.nan) for m in sc.MCAP_MODELS}
        for tr, te in KFold(5, shuffle=True, random_state=1).split(np.zeros(len(self.y))):
            pred = sc.mcap_fold(Xc, self.y, tr, te)
            for m in old:
                old[m][te] = pred[m]
        for m in old:
            np.testing.assert_array_equal(new[m], old[m], err_msg=m)

    def test_share_lines_share_outer_and_inner_folds(self):
        Xc = self.X[["log_tv_21", FE_BASE_FEATURES[0], FE_BASE_FEATURES[1]]]
        rec = Recorder(sc._group_kfold)
        with mock.patch.object(sc, "_group_kfold", rec):
            oof = sc.run_mcap_repeat(Xc, self.y, 2, groups=self.g)
        for m, v in oof.items():
            self.assertTrue(np.isfinite(v).all(), m)
        self.assertEqual(len(rec.calls), 1 + 5)                       # outer + one inner per outer fold
        for (_n, groups, _seed), _, splits in [(a, k, o) for a, k, o in rec.calls]:
            assert_pairs_together(self, splits, np.asarray(groups))

    def test_test_label_garbage_leaves_grouped_predictions_unchanged(self):
        Xc = self.X[["log_tv_21", FE_BASE_FEATURES[0], FE_BASE_FEATURES[1]]]
        tr, te = sc._group_kfold(len(self.y), self.g, 0)[0]
        y2 = self.y.copy()
        y2[te] = np.random.default_rng(2).normal(100, 50, len(te))
        a = sc.mcap_fold(Xc, self.y, tr, te, self.g)
        b = sc.mcap_fold(Xc, y2, tr, te, self.g)
        for m in a:
            np.testing.assert_allclose(a[m], b[m], err_msg=m)


class V9GroupedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.g = make_mcap()
        cls.Xb = cls.X[FE_BASE_FEATURES]
        cls.rows = make_rows(cls.X, cls.y)
        cls._patches = [mock.patch.object(S, "_lgbm", fast_lgbm)]
        for p in cls._patches:
            p.start()

    @classmethod
    def tearDownClass(cls):
        for p in cls._patches:
            p.stop()

    def test_inner_folds_none_is_kfold(self):
        train = np.arange(5, 35)
        want = list(KFold(5, shuffle=True, random_state=S.INNER_SEED).split(train))
        for (a, b), (c, d) in zip(want, S._inner_folds(train, None)):
            np.testing.assert_array_equal(a, c)
            np.testing.assert_array_equal(b, d)

    def test_groups_none_equals_old_path(self):
        tr, te = T.outer_folds(len(self.y), 0)[0]
        a = S.pair_fold(self.Xb, self.y, tr, te)
        b = S.pair_fold(self.Xb, self.y, tr, te, None)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
        # ridge / lgbm still equal static_cv.mcap_fold on the per-stock folds
        with mock.patch.object(sc, "_lgbm", fast_lgbm):
            ref = sc.mcap_fold(self.Xb, self.y, tr, te)
        np.testing.assert_allclose(a["ridge"], ref["ridge"], atol=1e-9)
        np.testing.assert_allclose(a["lgbm"], ref["lgbm"], atol=1e-9)
        m1, s1 = S.md_fold(self.rows, self.y, tr, te, FE_BASE_FEATURES)
        m2, s2 = S.md_fold(self.rows, self.y, tr, te, FE_BASE_FEATURES, None)
        np.testing.assert_array_equal(m1, m2)
        self.assertEqual(s1, s2)

    def test_share_lines_share_outer_and_inner_folds(self):
        rec = Recorder(S._group_kfold)
        with mock.patch.object(S, "_group_kfold", rec):
            oof = S.run_repeat(self.Xb, self.X, self.rows, self.y, 1, groups=self.g)
        for m, v in oof.items():
            self.assertTrue(np.isfinite(v).all(), m)
        self.assertTrue(len(rec.calls) > 5)
        for (_n, groups, _seed), _, splits in [(a, k, o) for a, k, o in rec.calls]:
            assert_pairs_together(self, splits, np.asarray(groups))

    def test_test_label_garbage_leaves_grouped_predictions_unchanged(self):
        tr, te = sc._group_kfold(len(self.y), self.g, 0)[0]
        y2 = self.y.copy()
        y2[te] = np.random.default_rng(1).normal(100, 50, len(te))
        for X in (self.Xb, self.X):
            a, b = S.pair_fold(X, self.y, tr, te, self.g), S.pair_fold(X, y2, tr, te, self.g)
            for k in a:
                np.testing.assert_array_equal(a[k], b[k], err_msg=k)
        for cols in (FE_BASE_FEATURES, T.FE_EXT):
            a, sa = S.md_fold(self.rows, self.y, tr, te, cols, self.g)
            b, sb = S.md_fold(self.rows, y2, tr, te, cols, self.g)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(sa, sb)


class IssuerIdsTests(unittest.TestCase):
    def test_share_lines_share_id_unknown_tickers_get_own(self):
        uni = pd.DataFrame({"ticker": ["LISN.SW", "LISP.SW", "ROG.SW", "RO.SW", "ROP.SW", "X.SW"],
                            "company": ["Lindt", "Lindt", "Roche", "Roche", "Roche", np.nan]})
        ids = sc.issuer_ids(["LISP.SW", "LISN.SW", "RO.SW", "ROP.SW", "NEW.SW", "X.SW", "ROG.SW"], uni)
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(ids[2], ids[3])
        self.assertEqual(ids[3], ids[6])
        self.assertEqual(len({ids[0], ids[2], ids[4], ids[5]}), 4)
        self.assertEqual(ids.dtype, int)

    def test_reads_csv_path(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "u.csv"
            pd.DataFrame({"ticker": ["A", "B"], "company": ["Z", "Z"]}).to_csv(f, index=False)
            np.testing.assert_array_equal(sc.issuer_ids(["A", "B"], f), [0, 0])


if __name__ == "__main__":
    unittest.main()
