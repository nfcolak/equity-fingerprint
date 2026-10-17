import unittest

# torch must be loaded before lightgbm (imported via forward_v9_beta -> forward_v7 -> models), see test_forward_v7.
import torch  # noqa: F401

import numpy as np
import pandas as pd

from equity_fingerprint import features as F
from equity_fingerprint import forward_v9_beta as B
from equity_fingerprint.features import Panel


def synth_panel(n=900, k=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-01", periods=n)
    cols = list("ABCDEFGH")[:k]
    m = pd.Series(rng.normal(0, 0.01, n), index=idx)
    betas = np.linspace(0.4, 1.6, k)
    r = pd.DataFrame(np.outer(m.to_numpy(), betas) + rng.normal(0, 0.004, (n, k)), index=idx, columns=cols)
    tv = pd.DataFrame(np.exp(rng.normal(10, 1, (n, k)) + np.linspace(0, 3, k)), index=idx, columns=cols)
    return Panel(ret=r, mkt=m, tv=tv)


def synth_dataset(n_tick=10, per=40, seed=1):
    """Observation table with every column run_experiment needs; beta_1y driven by the bases plus noise."""
    rng = np.random.default_rng(seed)
    n = n_tick * per
    d = pd.DataFrame({"ticker": np.repeat([f"T{i}" for i in range(n_tick)], per)})
    d["split"] = np.tile(np.repeat(["train", "val", "test"], [per - 16, 8, 8]), n_tick)
    for c in F.FE_FEATURES:
        d[c] = rng.normal(size=n)
    d["beta_252"] = rng.normal(1, 0.3, n)
    d["beta_welch252"] = d["beta_252"] + rng.normal(0, 0.1, n)
    d["beta_welch_ew"] = d["beta_welch252"] + rng.normal(0, 0.1, n)
    d["beta_vasicek"] = 0.7 * d["beta_welch252"] + 0.3
    d["beta_vasicek_size"] = 0.6 * d["beta_welch252"] + 0.4
    d["se_beta_252"] = 0.1
    for c in B.HAR_B:
        d[c] = d["beta_252"] + rng.normal(0, 0.2, n)
    d.loc[::17, "beta_welch_ew"] = np.nan  # exercise the fallback chain
    d["beta_1y"] = 0.6 * d["beta_welch_ew"].fillna(1.0) + 0.4 + 0.3 * d[F.FE_FEATURES[0]] + rng.normal(0, 0.2, n)
    return d


class ColumnTests(unittest.TestCase):
    def test_columns_are_causal(self):
        p = synth_panel()
        t = 600
        a = B.beta_panel_cols(p)
        p2 = Panel(ret=p.ret.copy(), mkt=p.mkt.copy(), tv=p.tv.copy())
        g = np.random.default_rng(5)
        p2.ret.iloc[t + 1:] = g.normal(0, 0.1, p2.ret.iloc[t + 1:].shape)
        p2.mkt.iloc[t + 1:] = g.normal(0, 0.1, len(p2.mkt.iloc[t + 1:]))
        p2.tv.iloc[t + 1:] = g.uniform(1, 1e6, p2.tv.iloc[t + 1:].shape)
        b = B.beta_panel_cols(p2)
        self.assertEqual(set(a), set(B.NEW_COLS))
        for k in B.NEW_COLS:
            np.testing.assert_array_equal(a[k].iloc[:t + 1].to_numpy(), b[k].iloc[:t + 1].to_numpy(), err_msg=k)
            self.assertFalse(np.allclose(a[k].iloc[t + 5:].to_numpy(), b[k].iloc[t + 5:].to_numpy(),
                                         equal_nan=True), k)

    def test_vasicek_shrinks_toward_cross_section(self):
        p = synth_panel()
        c = B.beta_panel_cols(p)
        bw = B.welch_beta(p.ret, p.mkt, 252)
        t = 700
        self.assertTrue(np.isfinite(c["beta_vasicek"].iloc[t]).all())
        self.assertGreater(c["se_beta_252"].iloc[t].min(), 0)
        # shrunk value lies between the own estimate and the same-date cross-sectional mean
        lo = np.minimum(bw.iloc[t], bw.iloc[t].mean()) - 1e-12
        hi = np.maximum(bw.iloc[t], bw.iloc[t].mean()) + 1e-12
        v = c["beta_vasicek"].iloc[t]
        self.assertTrue(((v >= lo) & (v <= hi)).all())
        self.assertTrue(c["beta_vasicek"].iloc[:250].isna().all().all())

    def test_add_beta_cols_lookup(self):
        p = synth_panel()
        d = pd.DataFrame({"pos": [400, 500, 800], "ticker": ["B", "A", "F"]})
        out = B.add_beta_cols(d, p)
        c = B.beta_panel_cols(p)
        for k in B.NEW_COLS:
            np.testing.assert_array_equal(out[k].to_numpy(),
                                          [c[k].loc[p.ret.index[i], t] for i, t in zip(d["pos"], d["ticker"])])


class BaseTests(unittest.TestCase):
    def test_fallback_chain_and_train_mean(self):
        d = pd.DataFrame({"beta_252": [1.0, np.nan, np.nan, np.nan],
                          "beta_welch252": [1.1, 1.2, np.nan, np.nan],
                          "beta_welch_ew": [1.3, np.nan, np.nan, np.nan],
                          "beta_vasicek": [0.9, np.nan, np.nan, np.nan],
                          "beta_vasicek_size": [0.8, np.nan, np.nan, np.nan]})
        b = B.make_bases(d, 7.0)
        np.testing.assert_allclose(b["ols"], [1.0, 7.0, 7.0, 7.0])
        np.testing.assert_allclose(b["w252"], [1.1, 1.2, 7.0, 7.0])
        np.testing.assert_allclose(b["wew"], [1.3, 1.2, 7.0, 7.0])
        np.testing.assert_allclose(b["vck"], [0.9, 1.2, 7.0, 7.0])
        np.testing.assert_allclose(b["mix"], [0.5 * 1.3 + 0.5 * 0.8, 1.2, 7.0, 7.0])


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = synth_dataset()
        cls.out, cls.info = B.run_experiment(cls.d)

    def test_residual_with_zero_base_equals_level(self):
        d = self.d
        tr, va, te = (d[d["split"] == s] for s in ("train", "val", "test"))
        cols = F.FE_FEATURES + B.WELCH_COLS
        z = lambda x: np.zeros(len(x))  # noqa: E731
        res = B.fit_residual(tr, va, te, cols, z(tr), z(va), z(te))
        lvl = B.fit_predict(tr, va, te, cols)
        for kind in ("linear", "lgbm"):
            for i in (0, 1):
                np.testing.assert_allclose(res[kind][i], lvl[kind][i], atol=1e-12)

    def test_schema_and_arms(self):
        out = self.out
        self.assertEqual(list(out.columns), B.OUT_COLS)
        order = ["fe", "fewelch", "resols", "resw252", "reswew", "resvck", "resmix", "resmixhar"]
        want = [f"{k}_{n}" for n in order for k in ("linear", "lgbm", "avg")]
        self.assertEqual(sorted(out["model"].unique()), sorted(want))
        self.assertEqual(list(dict.fromkeys(out["model"])), want)
        self.assertEqual(out.groupby("model")["selected"].agg(["min", "max"]).eval("min == max").all(), True)
        self.assertEqual(out[out["selected"] == 1]["model"].nunique(), 1)
        self.assertEqual(out[out["selected"] == 1]["model"].iloc[0], self.info["selected"])
        w = out[out["metric"] == "weight_linear"]
        self.assertEqual(set(w["model"]), {m for m in want if m.startswith("avg_")})
        self.assertTrue(w["value"].isin(B.V7.WEIGHTS).all())
        for ref in B.REFS:
            dl = out[out["metric"] == f"delta_r2_vs_{ref}"]
            self.assertNotIn(ref, set(dl["model"]))
            self.assertEqual(len(dl), len(want) - 1)
            self.assertTrue((dl["split"] == "test").all())
            self.assertTrue((dl["ci_lo"] <= dl["ci_hi"]).all())
        # selection is the best validation r2, first on ties
        v = out[(out["split"] == "val") & (out["metric"] == "r2")].set_index("model")["value"]
        self.assertEqual(v.idxmax(), self.info["selected"])

    def test_test_permutation_leaves_selection_and_weights(self):
        d2 = self.d.copy()
        te = d2["split"] == "test"
        d2.loc[te, "beta_1y"] = np.random.default_rng(9).permutation(d2.loc[te, "beta_1y"].to_numpy())
        out2, info2 = B.run_experiment(d2)
        self.assertEqual(info2["selected"], self.info["selected"])
        self.assertEqual(info2["weights"], self.info["weights"])
        a = self.out[self.out["split"] == "val"].reset_index(drop=True)
        b = out2[out2["split"] == "val"].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
        self.assertFalse(np.allclose(self.out[self.out["split"] == "test"]["value"],
                                     out2[out2["split"] == "test"]["value"], equal_nan=True))

    def test_reproduction_check_matches_itself(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "v7.csv"
            self.out.to_csv(f, index=False)
            chk = B.reproduction_check(self.out, f)
        self.assertEqual(len(chk), 8)
        self.assertTrue(chk["ok"].all())


if __name__ == "__main__":
    unittest.main()
