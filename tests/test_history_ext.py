"""Synthetic tests for history_ext: spliced market, long panel from the cache (extra rows in train only,
validation/test keys and targets equal the control, train-row beta on the spliced market), quality rule, forward
rows schema and selection on validation, knn_auto fold equality with static_cv, sector pairing."""
import torch  # noqa: F401  (macOS libomp: torch before lightgbm, which history_ext imports via forward_v7)
import contextlib
import hashlib
import io
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from equity_fingerprint import features as F
from equity_fingerprint import forward_v7 as V
from equity_fingerprint import history_ext as H
from equity_fingerprint import static_cv as sc
from equity_fingerprint.data import load_index

from _synth_markets import make_synth

CUT = pd.Timestamp("2010-01-04")        # first cached day (the "cache" starts here, as in the real data)
SSH_START = pd.Timestamp("2013-03-01")  # ^SSHI starts here; ^SSMI is complete
ONE_GRID = [dict(num_leaves=8, min_child_samples=20)]
_FX = None
_TMP = None


def snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def build_fixture(tmp: Path) -> dict:
    """Synthetic Swiss data cut to a cache from CUT (^SSHI from SSH_START; ^SSMI complete), as in the real data."""
    s = make_synth(tmp, T=5100, start="2007-01-02", n_ch=24, specials=False)
    dd = Path(s["data_dir"])
    prices = pd.read_parquet(dd / "raw" / "prices.parquet")
    index = pd.read_parquet(dd / "raw" / "index.parquet")
    for df in (prices, index):
        df["date"] = pd.to_datetime(df["date"])
    prices[prices["date"] >= CUT].to_parquet(dd / "raw" / "prices.parquet", index=False)
    cache_i = index[(index["date"] >= CUT) & ~((index["ticker"] == "^SSHI") & (index["date"] < SSH_START))]
    cache_i.to_parquet(dd / "raw" / "index.parquet", index=False)
    panels = {"ctrl": F.build_panel(dd), "long2011": H.build_long_panel(CUT, dd)}
    ds = {"ctrl": H.arm_datasets(panels["ctrl"]),
          "long2011": H.arm_datasets(panels["long2011"], panels["long2011"].calendar[0])}
    return dict(root=tmp, dd=dd, panels=panels, ds=ds)


def setUpModule():
    global _FX, _TMP
    _TMP = Path(tempfile.mkdtemp(prefix="hist_ext_"))
    _FX = build_fixture(_TMP)


def tearDownModule():
    shutil.rmtree(_TMP, ignore_errors=True)


def eval_view(d: pd.DataFrame, target: str) -> dict:
    keep = ["date", "ticker", target, f"end_{target}"]
    return {s: d[d["split"] == s].sort_values(["date", "ticker"], kind="stable")[keep].reset_index(drop=True)
            for s in ("val", "test")}


def planted_sector(seed=0, T_=400, nc=4, per=12):
    rng = np.random.default_rng(seed)
    n = nc * per
    y = np.repeat(np.arange(nc), per)
    idx = pd.bdate_range("2018-01-01", periods=T_)
    mk = rng.normal(size=T_) * 0.01
    ret = pd.DataFrame((rng.normal(size=(T_, nc)) * 0.01)[:, y] + mk[:, None] * rng.uniform(0.5, 1.5, n)
                       + rng.normal(size=(T_, n)) * 0.01, index=idx, columns=[f"S{i:02d}" for i in range(n)])
    mats = sc.build_matrices(ret, pd.Series(mk, index=idx))
    fe = pd.DataFrame(rng.normal(size=(n, 3)) + y[:, None] * 0.5, index=ret.columns, columns=["a", "b", "c"])
    return y, mats, fe, np.arange(n) // 2


# --------------------------------------------------------------------------- splice
class SpliceTests(unittest.TestCase):
    def test_start_before_the_cache_is_an_error(self):
        with self.assertRaises(ValueError):
            H.build_long_panel("2007-01-01", _FX["dd"])
        self.assertFalse(hasattr(H, "download_early"))  # the 2000-2009 arm is gone

    def test_spliced_market_is_continuous_and_equals_sshi_afterwards(self):
        d = pd.bdate_range("2009-01-01", periods=300)
        rng = np.random.default_rng(1)
        ssm = pd.Series(900 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=d)
        ssh = pd.Series(1000 * np.exp(np.cumsum(rng.normal(0, 0.01, 150))), index=d[150:])
        idx = pd.concat([pd.DataFrame({"date": s.index, "ticker": t, "close": s.to_numpy()})
                         for t, s in (("^SSMI", ssm), ("^SSHI", ssh))])
        sp = H.spliced_market(idx)
        pd.testing.assert_series_equal(sp.iloc[150:], ssh, check_names=False, check_freq=False)
        lr = np.log(sp).diff()
        np.testing.assert_allclose(lr.iloc[:150].dropna().to_numpy(), np.log(ssm).diff().iloc[1:150].to_numpy())
        self.assertAlmostEqual(float(lr.iloc[150]), float(np.log(ssm.iloc[150] / ssm.iloc[149])))
        with self.assertRaises(ValueError):
            H.spliced_market(idx[idx["ticker"] == "^SSHI"])

    def test_month_end_positions_from_matches_the_original(self):
        p = _FX["panels"]["ctrl"]
        np.testing.assert_array_equal(H.month_end_positions_from(p, F.OBS_START), F.month_end_positions(p))


# --------------------------------------------------------------------------- long panel and datasets
class LongPanelTests(unittest.TestCase):
    def test_calendar_and_market(self):
        pc, p11 = (_FX["panels"][a] for a in ("ctrl", "long2011"))
        self.assertEqual(p11.calendar[0], pc.calendar[0])
        self.assertTrue(pc.mkt[pc.calendar < SSH_START].isna().all())
        self.assertTrue(p11.mkt.iloc[1:].notna().all())  # ^SSMI stands in before ^SSHI
        after = pc.calendar >= SSH_START + pd.Timedelta(days=5)
        np.testing.assert_array_equal(p11.mkt[after].to_numpy(), pc.mkt[after].to_numpy())  # then ^SSHI itself

    def test_validation_and_test_keys_and_targets_equal_the_control(self):
        for arm in ("long2011",):
            for t in H.TARGETS:
                ctrl, other = (eval_view(_FX["ds"][a][t], t) for a in ("ctrl", arm))
                H.check_same_eval(ctrl, other, t)  # raises on any difference
                self.assertGreater(len(ctrl["test"]), 100)
                self.assertGreater(len(ctrl["val"]), 100)

    def test_check_same_eval_detects_differences(self):
        t = "beta_1y"
        ctrl = eval_view(_FX["ds"]["ctrl"][t], t)
        bad = {k: v.copy() for k, v in ctrl.items()}
        bad["test"].loc[5, t] += 1e-3
        with self.assertRaises(ValueError):
            H.check_same_eval(ctrl, bad, t)
        bad = {k: v.copy() for k, v in ctrl.items()}
        bad["val"] = bad["val"].iloc[1:].reset_index(drop=True)
        with self.assertRaises(ValueError):
            H.check_same_eval(ctrl, bad, t)
        bad = {k: v.copy() for k, v in ctrl.items()}
        bad["test"].loc[3, "ticker"] = "ZZZ.SW"
        with self.assertRaises(ValueError):
            H.check_same_eval(ctrl, bad, t)

    def test_extra_rows_are_train_rows_before_the_control_start(self):
        for arm in ("long2011",):
            for t in H.TARGETS:
                c, o = _FX["ds"]["ctrl"][t], _FX["ds"][arm][t]
                kc = set(zip(c["date"], c["ticker"]))
                ko = o[["date", "ticker"]].apply(tuple, axis=1)
                extra = o[~ko.isin(kc)]
                self.assertGreater(len(extra), 100, (arm, t))
                self.assertTrue((extra["split"] == "train").all())
                self.assertLess(extra["date"].max(), F.OBS_START)
                self.assertTrue(kc <= set(ko))  # no control row is lost
                for s in ("val", "test"):
                    self.assertEqual(int((c["split"] == s).sum()), int((o["split"] == s).sum()))

    def test_long2011_validation_and_test_rows_match_the_control_in_every_column(self):
        for t in H.TARGETS:
            for s in ("val", "test"):
                a, b = (_FX["ds"][k][t] for k in ("ctrl", "long2011"))
                a = a[a["split"] == s].reset_index(drop=True)
                b = b[b["split"] == s].reset_index(drop=True)
                pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-12)

    def test_train_row_beta_target_uses_the_spliced_market(self):
        dd = _FX["dd"]
        idx_all = load_index(dd)
        sm = H.spliced_market(idx_all)
        ssh = H._close(idx_all, F.MARKET)
        for arm in ("long2011",):
            p, d = _FX["panels"][arm], _FX["ds"][arm]["beta_1y"]
            cal = p.calendar
            m_sp = np.log(sm.reindex(cal.union(sm.index)).ffill().reindex(cal)).diff()
            m_h = np.log(ssh.reindex(cal.union(ssh.index)).ffill().reindex(cal)).diff()

            def beta(row, m):
                r, mm = p.ret[row["ticker"]].iloc[row["pos"] + 1:row["pos"] + 253], m.iloc[row["pos"] + 1:row["pos"] + 253]
                return r.cov(mm) / mm.var()

            early = d[(d["split"] == "train") & (d["date"] < pd.Timestamp("2011-12-01"))]
            self.assertGreater(len(early), 20)
            for _, row in early.iloc[::max(len(early) // 6, 1)].iterrows():
                self.assertLess(row["date"], SSH_START - pd.Timedelta(days=365))  # window entirely before ^SSHI
                self.assertAlmostEqual(row["beta_1y"], beta(row, m_sp), places=9)
            for s in ("val", "test"):  # validation / test rows: ^SSHI
                part = d[d["split"] == s]
                for _, row in part.iloc[::max(len(part) // 5, 1)].iterrows():
                    self.assertAlmostEqual(row["beta_1y"], beta(row, m_h), places=9)
        # the control has no such rows at all
        self.assertTrue((_FX["ds"]["ctrl"]["beta_1y"]["date"] >= F.OBS_START).all())

    def test_features_before_sshi_use_ssmi(self):
        p = _FX["panels"]["long2011"]
        ssm = H._close(load_index(_FX["dd"]), "^SSMI")
        m = np.log(ssm.reindex(p.calendar)).diff()
        pos = H.month_end_positions_from(p, p.calendar[0])
        pos = pos[(p.calendar[pos] > pd.Timestamp("2011-06-01")) & (p.calendar[pos] < SSH_START - pd.Timedelta(days=30))]
        base = F.base_features(p, pos)
        self.assertGreater(len(base), 50)
        self.assertTrue(base["beta_63"].notna().all() and base["beta_252"].notna().all())
        for _, row in base.iloc[::max(len(base) // 8, 1)].iterrows():
            r, mm = p.ret[row["ticker"]].iloc[row["pos"] - 62:row["pos"] + 1], m.iloc[row["pos"] - 62:row["pos"] + 1]
            self.assertAlmostEqual(row["beta_63"], r.cov(mm) / mm.var(), places=9)
            self.assertAlmostEqual(row["corr_63"], r.corr(mm), places=9)
        # the control panel has no market there
        pc = _FX["panels"]["ctrl"]
        self.assertTrue(F.base_features(pc, pos)["beta_63"].isna().all())

    def test_features_are_causal_under_future_perturbation(self):
        t_cut = pd.Timestamp("2011-09-30")
        with tempfile.TemporaryDirectory() as tmp:
            dd = Path(tmp) / "data"
            shutil.copytree(_FX["dd"], dd, ignore=shutil.ignore_patterns("early"))
            rng = np.random.default_rng(5)
            for name, cols in (("prices", ["open", "high", "low", "close", "adj_close", "volume"]),
                               ("index", ["open", "high", "low", "close", "adj_close"])):
                path = dd / "raw" / f"{name}.parquet"
                df = pd.read_parquet(path)
                df["date"] = pd.to_datetime(df["date"])
                fut = df["date"] > t_cut
                df.loc[fut, cols] = df.loc[fut, cols].to_numpy() * rng.lognormal(0, 0.2, (int(fut.sum()), len(cols)))
                df.to_parquet(path, index=False)
            p1, p2 = _FX["panels"]["long2011"], H.build_long_panel(CUT, dd)
            pos = H.month_end_positions_from(p1, p1.calendar[0])
            pos = pos[p1.calendar[pos] <= t_cut]
            a, b = F.base_features(p1, pos), F.base_features(p2, pos)
            self.assertGreater(len(a), 50)
            pd.testing.assert_frame_equal(a, b)
            late = H.month_end_positions_from(p1, p1.calendar[0])
            late = late[p1.calendar[late] > t_cut + pd.Timedelta(days=60)]
            self.assertFalse(F.base_features(p1, late).equals(F.base_features(p2, late)))  # the perturbation is real

    def test_ctrl_rows_equal_the_forward_v7_dataset_rows(self):
        p = _FX["panels"]["ctrl"]
        v7 = V.build_dataset(p).sort_values(["date", "ticker"]).reset_index(drop=True)
        mine = _FX["ds"]["ctrl"]["beta_1y"].sort_values(["date", "ticker"]).reset_index(drop=True)
        self.assertEqual(len(v7), len(mine))
        self.assertTrue((v7["ticker"] == mine["ticker"]).all() and (v7["date"] == mine["date"]).all())
        np.testing.assert_allclose(v7["beta_1y"], mine["beta_1y"], rtol=1e-12)
        self.assertTrue((v7["split"] == mine["split"]).all())

    def test_long_sector_matrices_need_every_control_stock(self):
        p = _FX["panels"]["long2011"]
        tickers = sorted(p.ret.columns)[:8]
        mats = H.long_sector_matrices(p, tickers)
        self.assertEqual(sorted(mats), sorted(sc.MATRIX_NAMES))
        self.assertEqual(mats["raw"].shape, (8, 8))
        with self.assertRaises(ValueError):
            H.long_sector_matrices(p, tickers + ["MISSING.SW"])


# --------------------------------------------------------------------------- quality rule
def toy_table(rows):
    return pd.DataFrame(rows, columns=["year", "n_stocks", "zero_vol_share", "nan_vol_share"])


class QualityTests(unittest.TestCase):
    def test_year_rule_on_a_toy_table(self):
        rows = [(y, 100, 0.05, 0.05) for y in range(2000, 2014)]
        rows[3] = (2003, 79, 0.05, 0.05)   # too few stocks
        rows[8] = (2008, 100, 0.16, 0.05)  # too many zero volumes
        t = H.apply_year_rule(toy_table(rows))
        self.assertEqual(H.admitted_start_year(t), 2009)
        self.assertEqual(t.set_index("year")["admitted"].loc[[2007, 2008, 2009, 2013]].tolist(), [False, False, True, True])
        self.assertEqual(t.set_index("year")["passes"].loc[[2003, 2008, 2004]].tolist(), [False, False, True])

    def test_year_rule_boundaries_and_nan(self):
        ok = H.apply_year_rule(toy_table([(2000, 80, 0.15, 0.15), (2001, 80, 0.15, 0.15)]))
        self.assertEqual(H.admitted_start_year(ok), 2000)
        nan_fail = H.apply_year_rule(toy_table([(2000, 80, 0.15, 0.15), (2001, 80, 0.15, 0.1501)]))
        self.assertIsNone(H.admitted_start_year(nan_fail))
        self.assertFalse(nan_fail["admitted"].any())
        nan_val = H.apply_year_rule(toy_table([(2000, 90, 0.0, 0.0), (2001, 90, np.nan, 0.0), (2002, 90, 0.0, 0.0)]))
        self.assertEqual(H.admitted_start_year(nan_val), 2002)

    def test_year_rule_treats_a_missing_year_as_failing(self):
        t = H.apply_year_rule(toy_table([(2000, 90, 0, 0), (2001, 90, 0, 0), (2003, 90, 0, 0), (2004, 90, 0, 0)]))
        self.assertEqual(H.admitted_start_year(t), 2003)

    def test_all_years_pass_starts_at_the_first_year(self):
        t = H.apply_year_rule(toy_table([(y, 120, 0.0, 0.0) for y in range(2000, 2005)]))
        self.assertEqual(H.admitted_start_year(t), 2000)
        self.assertTrue(t["admitted"].all())

    def test_quality_table_numbers(self):
        dates = pd.bdate_range("2001-01-01", "2004-12-31")
        n = 90
        tick = [f"T{i:02d}.SW" for i in range(n)]
        rng = np.random.default_rng(0)
        adj = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.005, (len(dates), n)), axis=0)), index=dates, columns=tick)
        adj.loc[dates[dates >= "2002-06-03"][0]:, tick[0]] *= 0.5      # a 2:1 unadjusted split in 2002
        adj.loc[dates[dates >= "2003-03-03"][0]:, tick[1]] *= 10.0     # a 10x artifact in 2003
        vol = pd.DataFrame(1000.0, index=dates, columns=tick)
        flat = np.arange(vol.size).reshape(vol.shape)
        vol = vol.mask((flat % 5 == 0) & (dates.year == 2002)[:, None])                 # 20% NaN in 2002
        vol = vol.mask((flat % 4 == 1) & (dates.year == 2003)[:, None], 0.0)            # ~25% zero in 2003
        long = pd.concat([pd.DataFrame({"date": dates, "ticker": t, "open": adj[t], "high": adj[t], "low": adj[t],
                                        "close": adj[t], "adj_close": adj[t], "volume": vol[t]}) for t in tick])
        info = pd.DataFrame({"ticker": tick + ["USD.SW"], "currency": ["CHF"] * n + ["USD"]})
        long = pd.concat([long, long[long["ticker"] == tick[0]].assign(ticker="USD.SW")])
        q = H.quality_table(long, info).set_index("year")
        self.assertEqual(q.index.tolist(), [2001, 2002, 2003, 2004])
        self.assertEqual(q["n_stocks"].tolist(), [90] * 4)  # the non-CHF line is excluded
        for y in (2001, 2002, 2003, 2004):
            sub = long[(long["date"].dt.year == y) & long["ticker"].isin(tick)]
            self.assertAlmostEqual(q.loc[y, "nan_vol_share"], sub["volume"].isna().mean())
            self.assertAlmostEqual(q.loc[y, "zero_vol_share"], (sub["volume"] == 0).mean())
            self.assertEqual(q.loc[y, "n_cells"], len(sub))
        self.assertEqual(q["artifact_returns"].tolist(), [0, 0, 1, 0])
        self.assertEqual(q["split_like_jumps"].tolist(), [0, 1, 1, 0])
        self.assertEqual(q["passes"].tolist(), [True, False, False, True])
        self.assertEqual(H.admitted_start_year(q.reset_index()), 2004)
        self.assertEqual(list(H.quality_table(long, info).columns), H.QUALITY_COLUMNS)

    def test_quality_table_counts_missing_volume_inside_the_span(self):
        dates = pd.bdate_range("2001-01-01", "2001-12-31")
        full = pd.DataFrame({"date": dates, "ticker": "A.SW", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                             "adj_close": 1.0, "volume": 5.0})
        other = full.assign(ticker="B.SW")  # keeps the calendar complete
        gap = pd.concat([full.drop(full.index[10:20]), other])  # ten missing days inside A's listing span
        q = H.quality_table(gap).set_index("year")
        self.assertEqual(q.loc[2001, "n_cells"], 2 * len(full))
        self.assertAlmostEqual(q.loc[2001, "nan_vol_share"], 10 / (2 * len(full)))
        self.assertEqual(q.loc[2001, "zero_vol_share"], 0.0)


# --------------------------------------------------------------------------- forward rows
class ForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with mock.patch.object(V, "LGBM_GRID", ONE_GRID):
            cls.out = H.forward_experiment(_FX["ds"], H.TARGETS, n_boot=60)

    def test_schema(self):
        o = self.out
        self.assertEqual(list(o.columns), H.COLUMNS)
        self.assertEqual(len(set(H.COLUMNS)), len(H.COLUMNS))
        self.assertTrue((o["selection"] == "ch_val").all() and (o["region"] == "CH").all())
        self.assertEqual(set(o["target"]), set(H.TARGETS))
        models = set(o["model"])
        for arm in H.ARMS:
            for k in H.KINDS:
                self.assertIn(f"{k}_{arm}", models)
        self.assertTrue({"r2", "mae", "rmse", "delta_r2_vs_ctrl", "weight_linear", "n_rows"} <= set(o["metric"]))
        self.assertFalse(o["metric"].str.contains("qlike").any())

    def test_one_selected_model_per_target_and_it_has_the_best_validation_r2(self):
        o = self.out
        for t in H.TARGETS:
            sub = o[(o["target"] == t) & (o["split"] == "val") & (o["metric"] == "r2") & o["model"].str.contains("_")]
            sel = o[(o["target"] == t) & (o["selected"] == 1)]["model"].unique()
            self.assertEqual(len(sel), 1)
            self.assertEqual(sel[0], sub.loc[sub["value"].idxmax(), "model"])

    def test_weights_on_the_grid_and_row_counts(self):
        o = self.out
        w = o[o["metric"] == "weight_linear"]
        self.assertTrue(w["value"].isin(V.WEIGHTS).all())
        self.assertEqual(len(w), len(H.ARMS) * len(H.TARGETS))
        for t in H.TARGETS:
            n = o[(o["target"] == t) & (o["metric"] == "r2") & (o["split"] == "test")]["n_obs"].unique()
            self.assertEqual(len(n), 1)  # every arm is scored on the same test rows
            rows = o[(o["target"] == t) & (o["metric"] == "n_rows") & (o["features"] == "data") & (o["split"] == "train")]
            c = rows.loc[rows["model"] == "ctrl", "value"].iloc[0]
            self.assertGreater(rows.loc[rows["model"] == "long2011", "value"].iloc[0], c)

    def test_primary_row_and_wider_interval(self):
        o = self.out
        for t in H.TARGETS:
            for split in ("val", "test"):
                r = o[(o["target"] == t) & (o["model"] == "avg_long2011") & (o["metric"] == "delta_r2_vs_ctrl")
                      & (o["split"] == split)]
                self.assertEqual(len(r), 1)
                r = r.iloc[0]
                self.assertEqual(r["ci_lo"], min(r["ci_lo_ticker"], r["ci_lo_month"]))
                self.assertEqual(r["ci_hi"], max(r["ci_hi_ticker"], r["ci_hi_month"]))
                self.assertLess(r["ci_lo"], r["ci_hi"])
                a = o[(o["target"] == t) & (o["model"] == "avg_long2011") & (o["metric"] == "r2") & (o["split"] == split)]
                b = o[(o["target"] == t) & (o["model"] == "avg_ctrl") & (o["metric"] == "r2") & (o["split"] == split)]
                self.assertAlmostEqual(r["value"], a["value"].iloc[0] - b["value"].iloc[0])
        self.assertFalse(((self.out["model"].str.endswith("_ctrl")) & (self.out["metric"] == "delta_r2_vs_ctrl")).any())

    def test_ctrl_only_run_has_no_deltas(self):
        with mock.patch.object(V, "LGBM_GRID", ONE_GRID):
            out = H.forward_experiment({"ctrl": _FX["ds"]["ctrl"]}, ("vol_1m",), n_boot=20)
        self.assertFalse((out["metric"] == "delta_r2_vs_ctrl").any())
        with self.assertRaises(ValueError):
            H.forward_experiment({"long2011": _FX["ds"]["long2011"]}, ("vol_1m",))

    def test_reproduction_check_reads_metrics_layout(self):
        o = self.out
        ctrl = o[o["model"].isin(["linear_ctrl", "lgbm_ctrl"]) & o["metric"].isin(["r2", "mae", "rmse"])]
        ref = ctrl.assign(model=ctrl["model"].str.replace("_ctrl", "_fe"))[["target", "model", "features", "split", "metric", "value", "n_obs"]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            ref.to_csv(path, index=False)
            chk = H.reproduction_check(o, path)
            self.assertEqual(len(chk), 2 * 2 * 2 * len(H.TARGETS))
            self.assertLess(float(chk["abs_diff"].max()), 1e-12)  # csv round trip
            self.assertTrue(chk["n_equal"].all())
            ref.loc[ref.index[0], "value"] += 0.01
            ref.to_csv(path, index=False)
            self.assertGreater(H.reproduction_check(o, path)["abs_diff"].max(), 0.009)

    def test_selection_and_weight_use_validation_rows_only(self):
        t = "vol_1m"
        d = _FX["ds"]["ctrl"][t].copy()
        te = (d["split"] == "test").to_numpy()
        d.loc[te, t] = np.random.default_rng(3).permutation(d.loc[te, t].to_numpy())
        with mock.patch.object(V, "LGBM_GRID", ONE_GRID):
            a, b = H.fit_arm(_FX["ds"]["ctrl"][t], t), H.fit_arm(d, t)
        self.assertEqual(a["weight"], b["weight"])
        for k in H.KINDS:
            np.testing.assert_array_equal(a["preds"][k][0], b["preds"][k][0])
            np.testing.assert_array_equal(a["preds"][k][1], b["preds"][k][1])
        self.assertNotAlmostEqual(V.r2(a["test"][t], a["preds"]["avg"][1]), V.r2(b["test"][t], b["preds"]["avg"][1]))


# --------------------------------------------------------------------------- sector arm
class SectorTests(unittest.TestCase):
    def test_knn_auto_repeat_equals_static_cv(self):
        y, mats, fe, groups = planted_sector()
        nc = int(y.max()) + 1
        for g in (None, groups):
            for repeat in (0, 1):
                sels_ref, sels = [], []
                ref = sc.run_sector_repeat(mats["raw"], y, fe, nc, repeat, mats, sels_ref, g)["knn_auto"]
                mine = H.knn_auto_repeat(mats, y, nc, repeat, g, sels)
                np.testing.assert_array_equal(ref, mine)
                self.assertEqual(sels_ref, sels)
                self.assertTrue((mine >= 0).all())

    def test_grouped_folds_keep_share_lines_together(self):
        y, mats, fe, groups = planted_sector()
        seen = []
        orig = H.knn_auto_fold

        def spy(mats_, y_, tr, te, nc, g=None, sel=None):
            seen.append((np.asarray(tr), np.asarray(te)))
            return orig(mats_, y_, tr, te, nc, g, sel)

        with mock.patch.object(H, "knn_auto_fold", spy):
            H.knn_auto_repeat(mats, y, int(y.max()) + 1, 0, groups)
        self.assertEqual(len(seen), 5)
        for tr, te in seen:
            self.assertFalse(set(groups[tr]) & set(groups[te]))

    def test_test_fold_labels_are_not_read(self):
        y, mats, fe, groups = planted_sector()
        nc = int(y.max()) + 1
        tr, te = np.arange(len(y))[len(y) // 4:], np.arange(len(y))[:len(y) // 4]
        base = H.knn_auto_fold(mats, y, tr, te, nc)
        y2 = y.copy()
        y2[te] = (y2[te] + 1) % nc
        np.testing.assert_array_equal(base, H.knn_auto_fold(mats, y2, tr, te, nc))

    def test_committed_loader(self):
        classes = ["A", "B", "C"]
        rows = [dict(target="sector", model=m, repeat=r, ticker=t, y_true="A", y_pred=p)
                for r in range(2) for m in ("knn_auto", "knn") for t, p in (("X", "B"), ("Y", "C" if r else "A"))]
        rows.append(dict(target="log_mcap", model="knn_auto", repeat=0, ticker="X", y_true=None, y_pred=None))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oof.parquet"
            pd.DataFrame(rows).to_parquet(path)
            out = H.committed_knn_auto(path, ["Y", "X"], classes, 2)
            self.assertEqual([o.tolist() for o in out], [[0, 1], [2, 1]])
            with self.assertRaises(ValueError):
                H.committed_knn_auto(path, ["Y", "X", "Z"], classes, 2)
            with self.assertRaises(ValueError):
                H.committed_knn_auto(path, ["Y", "X"], classes, 3)

    def test_experiment_schema_pairing_and_reproduction(self):
        y, mats, fe, groups = planted_sector()
        classes = ["c0", "c1", "c2", "c3"]
        y2, mats2, _, _ = planted_sector(seed=7)
        committed = [H.knn_auto_repeat(mats, y, 4, r) for r in range(2)]
        rows, sel, info = H.sector_experiment(classes, y, {"ctrl": mats, "long2011": mats, "alt": mats2},
                                              groups, repeats=2, committed=committed, n_boot=30)
        self.assertEqual(list(rows.columns), H.COLUMNS)
        self.assertEqual(info["rerun_mismatches"], 0)
        self.assertEqual(set(rows["folds"]), {"ticker", "issuer"})
        self.assertEqual(set(rows["model"]), {"knn_auto", "knn_auto_long"})
        self.assertEqual(set(rows.loc[rows["model"] == "knn_auto_long", "features"]), {"long2011", "alt"})
        self.assertEqual(set(rows.loc[(rows["model"] == "knn_auto") & (rows["folds"] == "ticker"), "features"]), {"committed"})
        self.assertEqual(set(rows.loc[(rows["model"] == "knn_auto") & (rows["folds"] == "issuer"), "features"]), {"ctrl"})
        self.assertTrue((rows["selection"] == "inner_cv").all() and (rows["target"] == "sector").all())
        self.assertTrue((rows["n_obs"] == len(y)).all() and (rows["n_repeats"] == 2).all())
        d = rows[(rows["metric"] == "delta_macro_f1_vs_knn_auto")]
        self.assertEqual(len(d), 4)  # 2 long arms x 2 fold schemes
        same = d[d["features"] == "long2011"]
        self.assertTrue((same["value"] == 0).all() and (same["ci_lo"] == 0).all() and (same["ci_hi"] == 0).all())
        self.assertTrue(((d["ci_lo"] <= d["ci_hi"])).all())
        self.assertEqual(set(sel.columns), {"folds", "arm", "matrix", "k", "count"})
        self.assertEqual(int(sel.loc[sel["folds"] == "ticker", "count"].sum()), 3 * 2 * 5)  # 3 arms x 2 repeats x 5 folds
        # a changed committed prediction is reported
        bad = [c.copy() for c in committed]
        bad[1][0] = (bad[1][0] + 1) % 4
        _, _, info2 = H.sector_experiment(classes, y, {"ctrl": mats, "long2011": mats}, groups, repeats=2,
                                          committed=bad, n_boot=10)
        self.assertEqual(info2["rerun_mismatches"], 1)

    def test_cli_end_to_end_on_synthetic_data(self):
        dd = str(_FX["dd"])
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(V, "LGBM_GRID", ONE_GRID):
            out, ref = Path(tmp) / "out", Path(tmp) / "ref"
            ref.mkdir()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    H.main([])  # a mode is required
                H.main(["--quality", "--data-dir", dd, "--out-dir", str(out)])
            q = pd.read_csv(out / "history_quality.csv")
            self.assertEqual(list(q.columns), H.QUALITY_COLUMNS)
            self.assertEqual(int(q["year"].min()), 2010)  # the cached years
            self.assertIn("admitted start year: None", buf.getvalue())  # 24 synthetic stocks < 80
            self.assertFalse((out / "history_ext.csv").exists())
            # run 1: forward arms of vol_1m without references (reproduction skipped), no sector
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                H.main(["--arms", "ctrl,long2011", "--targets", "vol_1m", "--no-sector", "--data-dir", dd,
                        "--out-dir", str(out), "--ref-dir", str(ref)])
            text = buf.getvalue()
            self.assertIn("[reproduction] skipped", text)
            first = pd.read_csv(out / "history_ext.csv")
            self.assertEqual(set(first["model"]) - {"ctrl", "long2011"}, {f"{k}_{a}" for k in H.KINDS for a in H.ARMS[:2]})
            # reference files built from run 1 and from the control kNN, then run 2 with the sector arm
            c = first[first["model"].isin(["linear_ctrl", "lgbm_ctrl"]) & first["metric"].isin(["r2", "mae"])]
            c.assign(model=c["model"].str.replace("_ctrl", "_fe")).to_csv(ref / "metrics.csv", index=False)
            p = _FX["panels"]["ctrl"]
            tickers, classes, y, *_ = sc.sector_inputs(p)
            mats = sc.sector_matrices(p, tickers)
            oof = [pd.DataFrame({"target": "sector", "model": "knn_auto", "repeat": r, "ticker": tickers,
                                 "y_pred": [classes[i] for i in H.knn_auto_repeat(mats, y, len(classes), r)]})
                   for r in range(2)]
            pd.concat(oof).to_parquet(ref / "static_cv_oof.parquet")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                H.main(["--arms", "--targets", "vol_1m", "--repeats", "2", "--data-dir", dd,
                        "--out-dir", str(out), "--ref-dir", str(ref)])
            text = buf.getvalue()
            self.assertIn("0 differing predictions", text)
            m = re.search(r"\[reproduction\] max abs diff ([0-9.e+-]+), n_obs equal: True", text)
            self.assertIsNotNone(m)
            self.assertLess(float(m.group(1)), 1e-12)
            both = pd.read_csv(out / "history_ext.csv")
            self.assertEqual(list(both.columns), H.COLUMNS)
            self.assertEqual(set(both["target"]), {"vol_1m", "sector"})
            self.assertIn("avg_long2011", set(both["model"]))
            self.assertEqual(set(both.loc[both["target"] == "sector", "folds"]), {"ticker", "issuer"})
            self.assertEqual(set(both.loc[both["target"] == "sector", "features"]),
                             {"committed", "ctrl", "long2011"})
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["history_ext.csv", "history_quality.csv"])

    def test_no_torch_in_the_module(self):
        import inspect
        src = inspect.getsource(H)
        self.assertNotIn("import torch", src)
        self.assertNotIn("tabpfn", src)


if __name__ == "__main__":
    unittest.main()
