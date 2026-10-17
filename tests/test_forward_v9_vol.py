import unittest

# torch before anything that loads lightgbm (forward_v9_vol -> forward_v7 -> models): the reverse order makes
# torch segfault on macOS (two libomp copies) in later test modules.
import torch  # noqa: F401

from unittest import mock

import numpy as np
import pandas as pd

from equity_fingerprint import features as F
from equity_fingerprint import forward_minirocket as M
from equity_fingerprint import forward_v9_vol as V

TICKERS = list("ABCDEFGH")


def synth_panel(seed=0, k=len(TICKERS)):
    """Small synthetic Panel (2013-03 .. 2025-03, k stocks) with slowly varying volatility."""
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2013-03-01", "2025-03-31")
    n = len(cal)
    h = np.zeros(n)
    for i in range(1, n):
        h[i] = 0.98 * h[i - 1] + 0.15 * rng.normal()
    sig = 0.01 * np.exp(0.5 * h)
    mkt = sig * 0.6 * rng.normal(size=n)
    ret = mkt[:, None] * rng.uniform(0.6, 1.4, k)[None, :] + sig[:, None] * rng.uniform(0.8, 1.5, k) * rng.normal(size=(n, k))
    ret[0] = np.nan
    mkt[0] = np.nan
    ret_df = pd.DataFrame(ret, index=cal, columns=TICKERS[:k])
    close = pd.DataFrame(100 * np.exp(np.nancumsum(ret, axis=0)), index=cal, columns=ret_df.columns)
    wiggle = lambda s: np.exp(np.abs(rng.normal(0, 0.004, close.shape))) ** s  # noqa: E731
    high = close * wiggle(1)
    low = close / wiggle(1)
    opn = close.shift(1).fillna(close)
    high, low = np.maximum(high, opn), np.minimum(low, opn)
    volume = pd.DataFrame(rng.uniform(1e4, 1e5, close.shape).round(), index=cal, columns=close.columns)
    return F.Panel(calendar=cal, tickers=list(close.columns), adj=close, close=close, open=opn, high=high, low=low,
                   volume=volume, tv=close * volume, ret=ret_df, mkt=pd.Series(mkt, index=cal),
                   rows=pd.DataFrame(np.repeat(np.arange(1, n + 1)[:, None], k, 1), index=cal, columns=close.columns),
                   span=pd.DataFrame(True, index=cal, columns=close.columns))


def synth_extra(panel, seed=1):
    rng = np.random.default_rng(seed)
    idx = panel.calendar[::1]
    vix = pd.Series(18 * np.exp(np.cumsum(rng.normal(0, 0.02, len(idx)))), index=idx)
    stx = pd.Series(3500 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    return pd.DataFrame({"^VIX": vix, "^STOXX50E": stx})


def obs_frame(panel, last_pos=None):
    pos = F.month_end_positions(panel)
    pos = pos[pos >= 600]
    if last_pos is not None:
        pos = pos[pos <= last_pos]
    rows = [(int(p), t, panel.calendar[p]) for p in pos for t in panel.tickers]
    return pd.DataFrame(rows, columns=["pos", "ticker", "date"])


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = synth_panel()
        cls.extra = synth_extra(cls.panel)

    def test_columns_and_dummies(self):
        obs = obs_frame(self.panel)
        f = V.new_features(self.panel, obs, self.extra)
        self.assertEqual(list(f.columns), V.NEW_COLS)
        mons = f[V.MONTH].to_numpy()
        self.assertTrue(set(np.unique(mons)) <= {0.0, 1.0})
        jan = pd.DatetimeIndex(obs["date"]).month == 1
        self.assertTrue((mons[jan] == 0).all())  # January is the base
        self.assertTrue((mons[~jan].sum(1) == 1).all())

    def test_hexp_matches_definition(self):
        obs = obs_frame(self.panel)
        f = V.new_features(self.panel, obs, self.extra)
        r = self.panel.ret["B"]
        e = (r ** 2).ewm(halflife=5, min_periods=5).mean()
        want = np.log(np.sqrt(e * 252).clip(lower=0.005))
        rows = obs["ticker"] == "B"
        np.testing.assert_allclose(f.loc[rows, "hexp5"].to_numpy(), want.to_numpy()[obs.loc[rows, "pos"]])

    def test_seasonal_is_shifted_window(self):
        obs = obs_frame(self.panel)
        f = V.new_features(self.panel, obs, self.extra)
        r = self.panel.ret["C"].to_numpy()
        i = obs.index[(obs["ticker"] == "C") & (obs["pos"] > 800)][0]
        p = int(obs.loc[i, "pos"])
        end = p - (252 - 21)
        rv = np.sqrt(np.mean(r[end - 20:end + 1] ** 2) * 252)
        self.assertAlmostEqual(f.loc[i, "seas1"], float(np.log(max(rv, 0.005))))
        rv252 = np.sqrt(np.mean(r[end - 251:end + 1] ** 2) * 252)
        self.assertAlmostEqual(f.loc[i, "seas1_rel"], float(np.log(max(rv, 0.005)) - np.log(max(rv252, 0.005))))

    def test_causal_in_panel_and_extra(self):
        t = 1500
        obs = obs_frame(self.panel, last_pos=t)
        a = V.new_features(self.panel, obs, self.extra)
        rng = np.random.default_rng(5)
        ret = self.panel.ret.copy()
        ret.iloc[t + 1:] = rng.normal(0, 0.1, ret.iloc[t + 1:].shape)
        extra = self.extra.copy()
        extra.iloc[t + 1:] = rng.uniform(5, 50, extra.iloc[t + 1:].shape)
        panel2 = F.Panel(**{**self.panel.__dict__, "ret": ret})
        b = V.new_features(panel2, obs, extra)
        np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
        # sanity: the perturbation does change features of later dates
        late = obs_frame(self.panel)
        late = late[late["pos"] > t + 60]
        self.assertFalse(np.allclose(V.new_features(self.panel, late, self.extra).to_numpy(),
                                     V.new_features(panel2, late, extra).to_numpy(), equal_nan=True))

    def test_extra_ffilled_on_swiss_calendar(self):
        cal = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"])
        ex = pd.DataFrame({"^VIX": [10.0, 20.0], "^STOXX50E": [1.0, 2.0]},
                          index=pd.DatetimeIndex(["2020-01-02", "2020-01-04"]))  # a Saturday close
        out = V.extra_series(ex, cal)
        self.assertEqual(out["^VIX"].tolist(), [10.0, 10.0, 20.0])
        with self.assertRaises(KeyError):
            V.extra_series(ex[["^VIX"]], cal)


class StackTests(unittest.TestCase):
    def test_nonneg_sum_one_and_recovers_blend(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=400)
        P = np.column_stack([y + rng.normal(0, s, 400) for s in (0.2, 0.5, 1.0, 2.0)])
        w = V.stack_weights(y, P)
        self.assertTrue((w >= 0).all())
        self.assertAlmostEqual(float(w.sum()), 1.0, places=9)
        self.assertGreater(w[0], 0.5)
        np.testing.assert_array_equal(w, V.stack_weights(y.copy(), P.copy()))

    def test_degenerate_input_still_valid(self):
        y = np.arange(5.0)
        w = V.stack_weights(y, np.column_stack([y, y, -y]))
        self.assertTrue((w >= 0).all())
        self.assertAlmostEqual(float(w.sum()), 1.0, places=9)


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = synth_panel()
        cls.extra = synth_extra(cls.panel)
        cls.time_df = M.build_time_frame(cls.panel)
        cls.mr = {}
        with mock.patch.object(M, "NUM_FEATURES", 840):  # small MiniRocket: keeps the ridge eigendecompositions fast
            cls.rows_with = M.run_target(cls.panel, cls.time_df, "vol_1m", log=lambda *_: None, preds_out=cls.mr)[0]
            cls.rows_without = M.run_target(cls.panel, cls.time_df, "vol_1m", log=lambda *_: None)[0]
        cls.d = V.build_dataset(cls.panel, cls.time_df, cls.extra)
        cls.out, cls.info = V.run_arms(cls.d, cls.mr)

    def test_preds_out_does_not_change_rows(self):
        pd.testing.assert_frame_equal(self.rows_with, self.rows_without)
        self.assertEqual(set(self.mr), {"linear_fe", "ridge_mr", "ridge_mrfe", "lgbm_femrpca", "avg_mrfe"})
        pv, pt, iv, it = self.mr["ridge_mrfe"]
        self.assertEqual((len(pv), len(pt)), (len(iv), len(it)))

    def test_dataset_rows_match_minirocket(self):
        d0 = M.target_frame(self.time_df, "vol_1m").reset_index(drop=True)
        self.assertEqual(len(self.d), len(d0))
        pd.testing.assert_series_equal(self.d["vol_1m"], d0["vol_1m"])
        for c in V.NEW_COLS + V.FE3:
            self.assertIn(c, self.d.columns)

    def test_schema(self):
        out = self.out
        self.assertEqual(list(out.columns), ["target", "model", "features", "split", "metric", "value", "n_obs",
                                             "ci_lo", "ci_hi", "selected"])
        order = ["linear_fe", "lgbm_fe", "linear_fehexp", "linear_feseas", "linear_femonth", "linear_fevix",
                 "linear_feall", "lgbm_feall", "avg_feall", "linear_fe3", "ridge_mrfe", "stack"]
        self.assertEqual(list(dict.fromkeys(out["model"])), order)
        self.assertEqual(set(out["target"]), {"vol_1m"})
        sel = out.groupby("model")["selected"].agg(["min", "max"])
        self.assertTrue((sel["min"] == sel["max"]).all())
        self.assertEqual(int(sel["max"].sum()), 1)
        val = out[(out["split"] == "val") & (out["metric"] == "r2")].set_index("model")["value"]
        self.assertEqual(self.info["selected"], val.index[int(np.argmax(val.to_numpy()))])
        self.assertEqual(int(out[out["selected"] == 1]["model"].iloc[0] == self.info["selected"]), 1)
        for m in order:
            for split in ("val", "test"):
                for metric in ("r2", "mae", "rmse", "qlike"):
                    self.assertEqual(len(out[(out["model"] == m) & (out["split"] == split)
                                             & (out["metric"] == metric)]), 1)
        dl = out[out["metric"] == "delta_r2_vs_linear_fe"]
        self.assertEqual(set(dl["model"]), set(order) - {"linear_fe"})
        self.assertTrue((dl["split"] == "test").all())
        self.assertTrue((dl["ci_lo"] <= dl["ci_hi"]).all())
        dr = out[out["metric"] == "delta_r2_vs_ridge_mrfe"]
        self.assertEqual(set(dr["model"]), set(order) - {"ridge_mrfe"})
        # ci only on delta rows
        self.assertTrue(out[~out["metric"].str.startswith("delta_")][["ci_lo", "ci_hi"]].isna().all().all())

    def test_stack_weight_rows(self):
        w = self.out[(self.out["model"] == "stack") & self.out["metric"].str.startswith("weight_")]
        self.assertEqual(sorted(w["metric"]), sorted(f"weight_{a}" for a in V.STACK_ARMS))
        self.assertTrue((w["value"] >= 0).all())
        self.assertAlmostEqual(float(w["value"].sum()), 1.0, places=9)
        self.assertEqual(set(w["split"]), {"val"})

    def test_test_label_permutation_leaves_selection_and_weights(self):
        d2 = self.d.copy()
        te = d2["split"] == "test"
        d2.loc[te, "vol_1m"] = np.random.default_rng(9).permutation(d2.loc[te, "vol_1m"].to_numpy())
        out2, info2 = V.run_arms(d2, self.mr)
        self.assertEqual(info2["selected"], self.info["selected"])
        self.assertEqual(info2["avg_weight"], self.info["avg_weight"])
        self.assertEqual(info2["stack_weights"], self.info["stack_weights"])
        v1 = self.out[self.out["split"] == "val"].reset_index(drop=True)
        v2 = out2[out2["split"] == "val"].reset_index(drop=True)
        pd.testing.assert_frame_equal(v1, v2)
        t1 = self.out[(self.out["split"] == "test") & (self.out["metric"] == "r2")]["value"].to_numpy()
        t2 = out2[(out2["split"] == "test") & (out2["metric"] == "r2")]["value"].to_numpy()
        self.assertFalse(np.allclose(t1, t2))

    def test_reproduction_check_shape(self):
        import tempfile
        from pathlib import Path
        ref = self.out[self.out["model"].isin(["linear_fe", "lgbm_fe"])
                       & self.out["metric"].isin(["r2", "mae", "rmse", "qlike"])]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "metrics.csv"
            ref.to_csv(p, index=False)
            chk = V.reproduction_check(self.out, p)
        self.assertEqual(len(chk), 16)
        self.assertLess(float(chk["abs_diff"].max()), 1e-12)
        self.assertTrue(chk["n_equal"].all())


if __name__ == "__main__":
    unittest.main()
