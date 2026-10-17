"""Synthetic tests for tabpfn_mcap: spike feature, causal extended features, TabPFN OOF with a stub regressor.
No real TabPFN model is ever created; this module must not load lightgbm or torch."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from equity_fingerprint import tabpfn_mcap as T
from equity_fingerprint.features import FE_BASE_FEATURES, Panel


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
    """Predicts the mean of the training labels; records what it was fitted on."""
    fits = []

    def __init__(self, repeat):
        self.repeat = repeat

    def fit(self, X, y):
        self.m = float(np.mean(y))
        StubReg.fits.append((self.repeat, X.shape, len(y)))
        return self

    def predict(self, X):
        return np.full(len(X), self.m)


class SpikeTests(unittest.TestCase):
    def test_third_friday(self):
        self.assertEqual(T.third_friday(2016, 3), pd.Timestamp("2016-03-18"))
        self.assertEqual(T.third_friday(2015, 12), pd.Timestamp("2015-12-18"))
        self.assertEqual(T.third_friday(2016, 1), pd.Timestamp("2016-01-15"))

    def test_planted_spike_on_third_fridays(self):
        cal = pd.bdate_range("2015-01-01", periods=600)
        vol = np.ones((600, 3))
        for yy in (2015, 2016):
            for mm in T.SPIKE_MONTHS:
                vol[cal.get_loc(T.third_friday(yy, mm)), 0] = 4.0
        pos = cal.get_loc(pd.Timestamp("2016-12-30"))
        ev = T.spike_events(cal, pos)
        self.assertEqual(len(ev), 8)
        self.assertTrue(all(cal[e] == T.third_friday(cal[e].year, cal[e].month) for e in ev))
        f = T.rebal_spike(vol, cal, pos)
        np.testing.assert_allclose(f, [np.log(4.0), 0.0, 0.0], atol=1e-12)

    def test_non_trading_day_falls_back_to_previous_trading_day(self):
        cal = pd.bdate_range("2015-01-01", periods=600)
        holiday = T.third_friday(2016, 3)
        cal = cal.delete(cal.get_loc(holiday))
        vol = np.ones((len(cal), 1))
        for yy, mm in [(2015, 3), (2015, 6), (2015, 9), (2015, 12), (2016, 6), (2016, 9)]:
            vol[cal.get_loc(T.third_friday(yy, mm)), 0] = 5.0
        thu = cal.get_loc(holiday - pd.Timedelta(days=1))
        vol[thu, 0] = 5.0
        pos = cal.get_loc(pd.Timestamp("2016-12-30"))
        ev = T.spike_events(cal, pos)
        self.assertIn(thu, ev)
        self.assertEqual(len(ev), 8)
        np.testing.assert_allclose(T.rebal_spike(vol, cal, pos), [np.log(5.0)], atol=1e-12)

    def test_fewer_than_four_events_is_nan_and_future_events_are_ignored(self):
        cal = pd.bdate_range("2015-01-01", periods=600)
        vol = np.full((600, 2), 2.0)
        # Mar, Jun, Sep 2015 up to 2015-10-30: 3 events
        self.assertTrue(np.isnan(T.rebal_spike(vol, cal, cal.get_loc(pd.Timestamp("2015-10-30")))).all())
        # the December 2015 event (18th) is still in the future on 2015-12-17
        self.assertTrue(np.isnan(T.rebal_spike(vol, cal, cal.get_loc(pd.Timestamp("2015-12-17")))).all())
        self.assertTrue(np.isfinite(T.rebal_spike(vol, cal, cal.get_loc(pd.Timestamp("2015-12-18")))).all())

    def test_zero_median_and_missing_volume(self):
        cal = pd.bdate_range("2015-01-01", periods=600)
        vol = np.ones((600, 3))
        vol[:, 1] = 0.0                      # median 0 everywhere -> no valid ratio
        vol[:, 2] = np.nan                   # not listed
        pos = cal.get_loc(pd.Timestamp("2016-12-30"))
        f = T.rebal_spike(vol, cal, pos)
        self.assertTrue(np.isnan(f[1:]).all())
        self.assertAlmostEqual(f[0], 0.0)


class CausalTests(unittest.TestCase):
    def test_new_features_ignore_data_after_the_feature_date(self):
        p = synth_panel()
        from equity_fingerprint.features import base_features, month_end_positions
        mp = month_end_positions(p)
        t = int(mp[-8])
        pos = mp[mp <= t][-6:]
        base = base_features(p, pos)
        a = T.ext_features(p, base)
        self.assertTrue(a.notna().any().all(), a.isna().all()[lambda s: s].index.tolist())
        p2 = synth_panel()
        g = np.random.default_rng(5)
        sl = slice(t + 1, None)
        for name in ("volume", "tv", "ret", "close", "high", "low", "open"):
            fr = getattr(p2, name)
            fr.iloc[sl] = g.uniform(0.5, 1e6, fr.iloc[sl].shape)
        p2.mkt.iloc[sl] = g.normal(0, 0.1, len(p2.mkt.iloc[sl]))
        p2.smi.iloc[sl] = g.normal(0, 0.1, len(p2.smi.iloc[sl]))
        b = T.ext_features(p2, base)
        self.assertEqual(list(a.columns), T.EXT_FEATURES)
        for c in T.EXT_FEATURES:
            np.testing.assert_array_equal(a[c].to_numpy(), b[c].to_numpy(), err_msg=c)
        # sanity: the perturbation is visible at a later date
        later = base_features(p, mp[-3:])
        self.assertFalse(np.allclose(T.ext_features(p, later).to_numpy(), T.ext_features(p2, later).to_numpy(),
                                     equal_nan=True))

    def test_feature_lists(self):
        self.assertEqual(T.FE_EXT, FE_BASE_FEATURES + [T.SPIKE_FEATURE] + T.VOLSTRUCT + T.SMI)
        self.assertEqual(len(set(T.FE_EXT)), len(T.FE_EXT))


class TabPFNOofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = synth_panel()
        cls.tickers, cls.y, cls.Xb, cls.Xe, cls.date = T.mcap_inputs_v9(cls.p)

    def test_inputs(self):
        n = len(self.tickers)
        self.assertGreaterEqual(n, 25)
        self.assertEqual(list(self.Xb.columns), FE_BASE_FEATURES)
        self.assertEqual(list(self.Xe.columns), T.FE_EXT)
        self.assertEqual(len(self.Xe), n)
        self.assertTrue(self.Xb.notna().all().all())
        np.testing.assert_array_equal(self.Xe[FE_BASE_FEATURES].to_numpy(), self.Xb.to_numpy())

    def test_stub_oof_schema_and_once_per_stock(self):
        StubReg.fits = []
        df = T.tabpfn_oof(self.Xb, self.Xe, self.y, self.tickers, 2, StubReg)
        self.assertEqual(list(df.columns), ["feature_set", "repeat", "ticker", "y_pred"])
        n = len(self.tickers)
        self.assertEqual(len(df), 2 * 2 * n)
        self.assertEqual(len(StubReg.fits), 2 * 5 * 2)
        for (fs, r), g in df.groupby(["feature_set", "repeat"]):
            self.assertEqual(sorted(g["ticker"]), sorted(self.tickers))
            self.assertTrue(np.isfinite(g["y_pred"]).all())
        shapes = {s for _, s, _ in StubReg.fits}
        self.assertEqual({s[1] for s in shapes}, {len(FE_BASE_FEATURES), len(T.FE_EXT)})

    def test_test_fold_labels_are_not_used(self):
        a = T.tabpfn_oof(self.Xb, self.Xe, self.y, self.tickers, 1, StubReg)
        for te in [te for _, te in T.outer_folds(len(self.y), 0)]:
            y2 = self.y.copy()
            y2[te] = np.random.default_rng(3).normal(1e3, 1e3, len(te))
            b = T.tabpfn_oof(self.Xb, self.Xe, y2, self.tickers, 1, StubReg)
            rows = a["ticker"].isin(np.array(self.tickers)[te])
            np.testing.assert_array_equal(a.loc[rows, "y_pred"].to_numpy(), b.loc[rows, "y_pred"].to_numpy())

    def test_run_writes_parquet(self):
        with tempfile.TemporaryDirectory() as d:
            T.run(repeats=1, out_dir=d, panel=self.p, factory=StubReg)
            df = pd.read_parquet(Path(d) / T.TABPFN_FILE)
            self.assertEqual(list(df.columns), ["feature_set", "repeat", "ticker", "y_pred"])
            self.assertEqual(set(df["feature_set"]), {"base", "ext"})

    def test_folds_match_static_cv_convention(self):
        from sklearn.model_selection import KFold
        for r in (0, 3):
            ref = list(KFold(5, shuffle=True, random_state=r).split(np.zeros(41)))
            for (a, b), (c, d) in zip(ref, T.outer_folds(41, r)):
                np.testing.assert_array_equal(b, d)

    def test_import_loads_neither_lightgbm_nor_torch(self):
        src = str(Path(T.__file__).resolve().parents[1])
        code = ("import sys, equity_fingerprint.tabpfn_mcap; "
                "sys.exit(int('lightgbm' in sys.modules or 'torch' in sys.modules or 'tabpfn' in sys.modules))")
        env = {**os.environ, "PYTHONPATH": src}
        self.assertEqual(subprocess.run([sys.executable, "-c", code], env=env).returncode, 0)


if __name__ == "__main__":
    unittest.main()
