"""Synthetic tests for multi_static: argpartition kNN, Swiss-query evaluation (no test-label use, Swiss
tickers only, issuer-grouped folds, foreign refs restricted to Swiss classes, nested a = 0, menu entries),
per-region metrics on a hand-computed case, share-count gap filter, output schema."""
import torch  # noqa: F401  (macOS libomp: torch before lightgbm, which multi_static imports via static_cv)
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from equity_fingerprint import data
from equity_fingerprint import features as F
from equity_fingerprint import multi_market as mm
from equity_fingerprint import multi_static as ms
from equity_fingerprint import static_cv as sc
from equity_fingerprint import static_v9_sector as v9

from _synth_markets import make_synth

NC = 4


def toy_problem(seed=0, n_ch=40, n_eu=100):
    """Menu matrices on planted clusters: Swiss rows first, then EU. Foreign class NC is not a Swiss class
    (label -1)."""
    rng = np.random.default_rng(seed)
    cls = np.concatenate([np.repeat(np.arange(NC), n_ch // NC), rng.integers(0, NC + 1, n_eu)])
    region = np.array(["CH"] * n_ch + ["EU"] * n_eu)
    latent = np.eye(NC + 1)[cls] * 1.0 + rng.normal(size=(len(cls), NC + 1)) * 0.9
    base = np.corrcoef(latent)

    def mat(noise):
        m = base + rng.normal(size=base.shape) * noise
        m = (m + m.T) / 2
        np.fill_diagonal(m, np.nan)
        return m

    d = mat(0.05)
    y = np.where(cls == NC, -1, cls)
    y[:n_ch] = cls[:n_ch]
    groups = np.arange(len(cls))
    for i in range(5):                      # five Swiss issuers with two share lines (same class)
        groups[2 * i + 1] = groups[2 * i]
    return dict(mats={"d_local": d, "w_local": mat(0.05)}, y=y, region=region, groups=groups, n_ch=n_ch)


class ArgpartitionTests(unittest.TestCase):
    def test_equals_originals_with_ties_nans_and_short_reference(self):
        rng = np.random.default_rng(0)
        for n_ref, rounding in ((60, 1), (60, 8), (8, 8), (3, 8)):
            S = np.round(rng.normal(size=(30, n_ref)), rounding)
            S[rng.random(S.shape) < 0.15] = np.nan
            S[3] = np.nan                                   # a query without any similarity
            S[4, : n_ref // 2] = np.nan                     # fewer finite neighbours than k
            y_ref = rng.integers(0, NC, n_ref)
            y_ref[:2] = [0, 1]
            i0, w0 = v9._neighbours(S, 10)
            i1, w1 = ms.neighbours_ap(S, 10)
            np.testing.assert_array_equal(i0, i1)
            np.testing.assert_array_equal(w0, w1)
            a = v9.scan_predictions(S, y_ref, NC, 2)
            b = ms.scan_predictions_ap(S, y_ref, NC, 2)
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                np.testing.assert_array_equal(a[key], b[key], err_msg=str(key))
            prior = np.bincount(y_ref, minlength=NC) / n_ref
            for k in (3, 7):
                for al in (0, 0.5, 1.0):
                    np.testing.assert_array_equal(v9.knn_predict_prior(S, y_ref, k, NC, 2, prior, al),
                                                  ms.knn_predict_prior_ap(S, y_ref, k, NC, 2, prior, al))

    def test_a_zero_is_plain_knn(self):
        rng = np.random.default_rng(1)
        S = rng.normal(size=(25, 40))
        y_ref = rng.integers(0, NC, 40)
        out = ms.scan_predictions_ap(S, y_ref, NC, 1)
        for k in sc.KNN_KS:
            np.testing.assert_array_equal(out[(k, 0)], sc.knn_predict(S, y_ref, k, NC, 1))


class FoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = toy_problem()

    def folds(self, repeat=0):
        t = self.t
        return ms.swiss_folds(t["y"][:t["n_ch"]], t["groups"][:t["n_ch"]], repeat)

    def run_fold(self, y, arm, tr, te, **kw):
        t = self.t
        return ms.ch_query_fold(t["mats"], y, t["region"], t["groups"], tr, te, NC, arm, t["n_ch"], **kw)

    def test_swiss_test_labels_are_not_read(self):
        tr, te = self.folds()[0]
        y2 = self.t["y"].copy()
        y2[te] = (y2[te] + 1) % NC
        for arm in ms.ARM_REFS:
            a, sa = self.run_fold(self.t["y"], arm, tr, te)
            b, sb = self.run_fold(y2, arm, tr, te)
            np.testing.assert_array_equal(a, b, err_msg=arm)
            self.assertEqual(sa, sb)

    def test_queries_must_be_swiss(self):
        tr, te = self.folds()[0]
        with self.assertRaises(ValueError):
            self.run_fold(self.t["y"], "ch_eu", tr, np.array([self.t["n_ch"] + 1]))

    def test_repeat_predicts_every_swiss_stock_once(self):
        t = self.t
        oof = ms.ch_query_repeat(t["mats"], t["y"], t["region"], t["groups"], t["n_ch"], NC, 0,
                                 ["ch_menu", "ch_eu"])
        for arm, v in oof.items():
            self.assertEqual(len(v), t["n_ch"])
            self.assertTrue(((v >= 0) & (v < NC)).all(), arm)
        self.assertGreater(np.mean(oof["ch_eu"] == t["y"][:t["n_ch"]]), 0.5)

    def test_share_classes_share_outer_and_inner_fold(self):
        t = self.t
        n_ch = t["n_ch"]
        seen = []
        real = ms._inner_splits

        def spy(strata, seed, groups):
            out = real(strata, seed, groups)
            seen.append((np.asarray(groups), out))
            return out

        for r in range(4):
            for tr, te in self.folds(r):
                for i in range(5):
                    a, b = 2 * i, 2 * i + 1
                    self.assertEqual(a in te, b in te, (r, i))
                with mock.patch.object(ms, "_inner_splits", side_effect=spy):
                    self.run_fold(t["y"], "ch_menu", tr, te)
        self.assertTrue(seen)
        shared = 0
        for groups, splits in seen:
            for itr, iva in splits:
                for g in set(groups[iva]):
                    self.assertFalse(set(np.flatnonzero(groups == g)) & set(itr) and
                                     set(np.flatnonzero(groups == g)) & set(iva), "issuer split across inner folds")
                    shared += int((groups == g).sum() > 1)
        self.assertGreater(shared, 0)
        # without groups the same pair would be split in at least one repeat
        split_somewhere = False
        for r in range(8):
            for _tr, te in ms.swiss_folds(t["y"][:n_ch], np.arange(n_ch), r):
                split_somewhere |= any((2 * i in te) != (2 * i + 1 in te) for i in range(5))
        self.assertTrue(split_somewhere)

    def test_foreign_refs_restricted_to_swiss_classes_and_arm_regions(self):
        t = self.t
        tr, te = self.folds()[0]
        for arm, regs in ms.ARM_REFS.items():
            calls = []
            real = ms._masked_sims

            def spy(C, ref, query):
                calls.append(np.asarray(ref))
                return real(C, ref, query)

            with mock.patch.object(ms, "_masked_sims", side_effect=spy):
                self.run_fold(t["y"], arm, tr, te)
            self.assertTrue(calls)
            for ref in calls:
                foreign = ref[ref >= t["n_ch"]]
                self.assertTrue((t["y"][foreign] >= 0).all(), arm)             # only Swiss classes
                self.assertTrue(np.isin(t["region"][foreign], regs).all(), arm)
                self.assertTrue(set(ref[ref < t["n_ch"]]) <= set(tr))          # Swiss refs from the train fold
            if regs:
                self.assertTrue((np.concatenate(calls) >= t["n_ch"]).any(), arm)
        self.assertTrue((t["y"][t["n_ch"]:] < 0).any())                         # the toy has a non-Swiss class

    def test_a_zero_nested_equals_plain_knn(self):
        t = self.t
        tr, te = self.folds()[0]
        sels = []
        extra = np.flatnonzero((t["region"] == "EU") & (t["y"] >= 0))
        pred, sel = ms.sector_search_fold(t["mats"], t["y"], tr, te, extra, ["w_local"], NC, t["groups"],
                                          a_grid=(0,))
        self.assertEqual(sel[2], 0)
        refs = np.concatenate([tr, extra])
        maj = int(np.bincount(t["y"][tr], minlength=NC).argmax())
        S = sc._masked_sims(t["mats"]["w_local"], refs, te)
        np.testing.assert_array_equal(pred, sc.knn_predict(S, t["y"][refs], sel[1], NC, maj))
        # the full a grid can only choose a from the grid, ties go to a = 0
        _, sel_full = ms.sector_search_fold(t["mats"], t["y"], tr, te, extra, ["w_local"], NC, t["groups"])
        self.assertIn(sel_full[2], v9.A_GRID)
        del sels

    def test_arms_are_eu_only_and_search_the_full_menu(self):
        t = self.t
        self.assertEqual(ms.FOREIGN_REGIONS, ("EU",))
        self.assertEqual(ms.CH_ARMS, ["ch_knn_auto", "ch_menu", "ch_eu"])
        self.assertEqual(ms.PRIMARY, ("ch_eu", "ch_knn_auto"))
        for arm in ms.ARM_REFS:
            self.assertEqual(ms.arm_entries(arm), ["d_local", "w_local"])
        with self.assertRaises(KeyError):
            ms.arm_entries("ch_us")
        sels = []
        ms.ch_query_repeat(t["mats"], t["y"], t["region"], t["groups"], t["n_ch"], NC, 0, selections=sels)
        self.assertEqual(len(sels), 5 * len(ms.ARM_REFS))
        self.assertTrue(all(entry in ms.MENU for _, _, _, _, entry, _, _ in sels))

    def test_pooled_repeat_has_all_stocks(self):
        t = self.t
        sel = np.flatnonzero(t["y"] >= 0)
        y = t["y"][sel]
        region = t["region"][sel]
        strata = pd.factorize(pd.Series(region) + "|" + pd.Series(y).astype(str))[0]
        groups = t["groups"][sel]
        M = {"w_local": t["mats"]["w_local"][np.ix_(sel, sel)]}
        sels = []
        oof = ms.pooled_sector_repeat(M, y, strata, groups, NC, 0, sels)
        self.assertTrue((oof["all_knn"] >= 0).all() and (oof["baseline"] >= 0).all())
        self.assertEqual(len(sels), 5)
        self.assertTrue(all(s[4] == "w_local" for s in sels))
        self.assertGreater(np.mean(oof["all_knn"] == y), np.mean(oof["baseline"] == y))


class MetricTests(unittest.TestCase):
    def test_present_label_macro_matches_sklearn(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            yt = rng.integers(0, 3, 30)               # class 3 never present
            yp = rng.integers(0, 4, 30)
            m = ms.fast_present_metrics(yt, yp, 4)
            lab = sorted(set(yt))
            self.assertAlmostEqual(m["macro_f1"], f1_score(yt, yp, labels=lab, average="macro", zero_division=0))
            self.assertAlmostEqual(m["weighted_f1"], f1_score(yt, yp, labels=list(range(4)), average="weighted",
                                                              zero_division=0))
            self.assertAlmostEqual(m["accuracy"], float(np.mean(yt == yp)))

    def test_region_rows_hand_computed(self):
        # CH: truth [0,0,1,1], pred [0,1,1,1] -> f1(0)=2/3, f1(1)=0.8 -> macro 11/15, accuracy 0.75
        # EU: truth [2,2], pred [2,0]       -> only label 2 present: f1(2)=2/3 (tp1, fp0, fn1) -> macro 2/3
        truth = np.array([0, 0, 1, 1, 2, 2])
        pred = np.array([0, 1, 1, 1, 2, 0])
        region = np.array(["CH", "CH", "CH", "CH", "EU", "EU"])
        rows = pd.DataFrame(ms.region_rows("sector", ["m"], truth, [{"m": pred}], region, "pooled",
                                           lambda t, q: ms.fast_present_metrics(t, q, 3),
                                           ms.class_boot(3, 10), 3))
        get = lambda reg, met: rows[(rows.region == reg) & (rows.metric == met)].iloc[0]
        self.assertAlmostEqual(get("CH", "macro_f1")["mean"], 11 / 15)
        self.assertAlmostEqual(get("CH", "accuracy")["mean"], 0.75)
        self.assertAlmostEqual(get("EU", "macro_f1")["mean"], 2 / 3)
        self.assertAlmostEqual(get("EU", "accuracy")["mean"], 0.5)
        # ALL: labels 0, 1, 2 all present; tp = [1, 2, 1], pred counts = [2, 3, 1], support = [2, 2, 2]
        f1 = [2 * 1 / (2 + 2), 2 * 2 / (2 + 3), 2 * 1 / (2 + 1)]
        self.assertAlmostEqual(get("ALL", "macro_f1")["mean"], float(np.mean(f1)))
        self.assertAlmostEqual(get("ALL", "weighted_f1")["mean"], float(np.mean(f1)))
        self.assertEqual(get("CH", "macro_f1")["n_stocks"], 4)
        self.assertEqual(set(rows["eval"]), {"pooled"})
        self.assertEqual(list(rows[rows.metric == "accuracy"]["region"]), ["CH", "EU", "ALL"])


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # the synthetic history starts in 2018: first month-end row needs 252 days of history
        cls._patch = mock.patch.object(F, "OBS_START", pd.Timestamp("2019-02-01"))
        cls._patch.start()
        cls._tmp = tempfile.TemporaryDirectory()
        cls.s = make_synth(cls._tmp.name, n_ch=36)
        # US lines are loaded on purpose: multi_static must drop them itself (FOREIGN_REGIONS)
        cls.swiss, cls.panels, cls.table, cls.drops = mm.load_markets(cls.s["data_dir"], cls.s["pretrain_dir"],
                                                                      ("EU", "US"))
        cls.universe = data.load_universe(cls.s["data_dir"])
        cls.sd = ms.build_sector_data(cls.swiss, cls.panels, cls.table, cls.universe)

    @classmethod
    def tearDownClass(cls):
        cls._patch.stop()
        cls._tmp.cleanup()

    def test_menu_matrices(self):
        sd = self.sd
        n = len(sd.tickers)
        self.assertEqual(sd.tickers[:sd.n_ch], sc.sector_inputs(self.swiss)[0])
        for m in sd.mats.values():
            self.assertEqual(m.shape, (n, n))
            self.assertTrue(np.isnan(np.diag(m)).all())
            np.testing.assert_allclose(m, m.T, equal_nan=True)
        self.assertEqual(set(sd.region), {"CH", "EU"})                          # the US lines are not used
        self.assertTrue((self.table["region"] == "US").any())                    # although the fixture has them
        d, w = sd.mats["d_local"], sd.mats["w_local"]
        off_diag = ~np.eye(n, dtype=bool)
        self.assertGreater(np.isfinite(d[off_diag]).mean(), 0.95)
        self.assertGreater(np.isfinite(w[off_diag]).mean(), 0.9)
        # Swiss-Swiss daily similarity equals the static_cv SPI-residual matrix
        n_ch = sd.n_ch
        np.testing.assert_allclose(d[:n_ch, :n_ch], sd.ctrl_mats["resid"], atol=1e-9, equal_nan=True)
        # same-sector stocks of different markets are closer on average (planted sector factors)
        sec = np.array([s for s in sd.sector])
        same = sec[:, None] == sec[None, :]
        off = ~np.eye(n, dtype=bool)
        self.assertGreater(np.nanmean(w[same & off]), np.nanmean(w[~same & off]) + 0.1)

    def test_groups_and_labels(self):
        sd = self.sd
        g = sd.groups
        self.assertEqual(g[0], g[1])
        self.assertEqual(g[2], g[3])
        self.assertEqual(len(set(g[:sd.n_ch])), sd.n_ch - 2)
        self.assertTrue(set(g[:sd.n_ch]).isdisjoint(set(g[sd.n_ch:])))
        np.testing.assert_array_equal(g[:sd.n_ch], sc.issuer_ids(sd.tickers[:sd.n_ch], self.universe))
        self.assertEqual(sd.classes, sorted(set(sd.sector[:sd.n_ch])))
        self.assertTrue((sd.y >= 0).all())

    def test_dropped_lines_not_present(self):
        self.assertNotIn(self.s["specials"]["wedge_dup"], self.sd.tickers)
        self.assertFalse(any(t.endswith(".SW") for t in self.sd.tickers[self.sd.n_ch:]))

    def test_available_arms(self):
        sd = self.sd
        self.assertEqual(ms.available_arms(sd.region, sd.y, sd.n_ch), ["ch_menu", "ch_eu"])
        no_eu = np.where(sd.region == "EU", "XX", sd.region)
        self.assertEqual(ms.available_arms(no_eu, sd.y, sd.n_ch), ["ch_menu"])

    def test_evaluation_a_sector(self):
        sd = self.sd
        rows, oof, sels, reps = ms.evaluate_ch_query_sector(sd, repeats=1, n_boot=20)
        res = pd.DataFrame(rows)
        self.assertEqual(set(res["eval"]), {"ch_query"})
        self.assertEqual(set(res["region"]), {"CH"})
        self.assertEqual(set(res[res.metric == "macro_f1"]["model"]), set(ms.CH_ARMS))
        prim = res[res.metric == "delta_macro_f1_vs_ch_knn_auto"].set_index("model")
        self.assertIn("ch_eu", prim.index)
        r = prim.loc["ch_eu"]
        self.assertLessEqual(r["ci_lo"], r["mean"] + 1e-9)
        self.assertGreaterEqual(r["ci_hi"], r["mean"] - 1e-9)
        # the delta is the difference of the arm rows (repeats = 1)
        m = res[res.metric == "macro_f1"].set_index("model")["mean"]
        self.assertAlmostEqual(r["mean"], m["ch_eu"] - m["ch_knn_auto"])
        self.assertIn("delta_macro_f1_vs_ch_menu", set(res.metric))
        # OOF: Swiss tickers only, every Swiss stock once per model
        o = pd.concat(oof, ignore_index=True)
        self.assertTrue(o["ticker"].isin(sd.tickers[:sd.n_ch]).all())
        self.assertEqual(set(o["region"]), {"CH"})
        self.assertEqual(len(o), len(ms.CH_ARMS) * sd.n_ch)
        self.assertFalse(o["ticker"].isin(self.table["ticker"]).any())
        # selection log: control uses a static_cv matrix, menu arms a menu entry
        s = pd.DataFrame(sels, columns=ms.SELECTION_COLUMNS)
        self.assertEqual(len(s), len(ms.CH_ARMS) * 5)
        self.assertTrue(s[s.arm == "ch_knn_auto"]["entry"].isin(sc.MATRIX_NAMES).all())
        self.assertTrue(s[s.arm == "ch_menu"]["entry"].isin(ms.MENU).all())
        # the control is the static_cv grouped run itself
        g = sd.groups[:sd.n_ch]
        direct = sc.run_sector_repeat(sd.ctrl_C, sd.y[:sd.n_ch], sd.ctrl_fe, len(sd.classes), 0, sd.ctrl_mats, None, g)
        np.testing.assert_array_equal(reps[0]["ch_knn_auto"], direct["knn_auto"])

    def test_evaluation_b_sector(self):
        rows, oof, sels = ms.evaluate_pooled_sector(self.sd, repeats=1, n_boot=20)
        res = pd.DataFrame(rows)
        self.assertEqual(set(res["eval"]), {"pooled"})
        self.assertEqual(set(res["region"]), {"CH", "EU", "ALL"})
        self.assertEqual(set(res["model"]), {"all_knn", "baseline"})
        self.assertEqual(set(res["metric"]), {"macro_f1", "weighted_f1", "accuracy"})
        a = res[(res.model == "all_knn") & (res.metric == "macro_f1") & (res.region == "ALL")]["mean"].iloc[0]
        b = res[(res.model == "baseline") & (res.metric == "macro_f1") & (res.region == "ALL")]["mean"].iloc[0]
        self.assertGreater(a, b)
        o = pd.concat(oof, ignore_index=True)
        self.assertEqual(set(o["region"]), {"CH", "EU"})
        n = res[(res.model == "all_knn") & (res.metric == "accuracy")].set_index("region")["n_stocks"]
        self.assertEqual(n["ALL"], n["CH"] + n["EU"])
        self.assertEqual(len(sels), 5)

    def test_share_gap_filter(self):
        tab = self.table.copy()
        eu = tab.index[tab.region == "EU"][:3]
        tab.loc[eu[0], "market_cap"] *= 2.0         # log gap 0.69 -> dropped
        tab.loc[eu[1], "market_cap"] *= 1.1         # log gap 0.10 -> kept
        kept, log = ms.share_gap_filter(self.panels, tab)
        self.assertNotIn(tab.loc[eu[0], "ticker"], kept)
        self.assertIn(tab.loc[eu[1], "ticker"], kept)
        self.assertEqual(len(kept), len(tab) - 1)                   # pence lines are scaled, nothing else drops
        self.assertEqual(int((~log["kept"]).sum()), 1)
        self.assertEqual(set(log["region"]), {"EU", "US"})          # the filter itself is region-generic

    def test_mcap_data_and_evaluation(self):
        tickers, region, y, X, groups, n_ch, log = ms.build_mcap_data(self.swiss, self.panels, self.table, self.universe)
        self.assertEqual(len(tickers), len(y))
        self.assertEqual(list(X.columns), ms.FE_BASE_FEATURES + ["reg_EU"])
        self.assertEqual(set(region), {"CH", "EU"})
        np.testing.assert_array_equal(X["reg_EU"].to_numpy() == 1, region == "EU")
        self.assertTrue(np.isfinite(y).all() and X[ms.FE_BASE_FEATURES].notna().all().all())
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(len(set(groups)), len(groups) - 2)
        rows, oofs = ms.evaluate_mcap(self.swiss, self.panels, self.table, self.universe, repeats=1, n_boot=20)
        res = pd.DataFrame(rows)
        a = res[res["eval"] == "ch_query"]
        self.assertEqual(set(a["model"]), {"ch_ridge", "ch_lgbm"})
        self.assertEqual(set(a["metric"]), {"r2", "mae"})
        b = res[res["eval"] == "pooled"]
        self.assertEqual(set(b["model"]), {"all_ridge", "all_lgbm", "baseline"})
        self.assertEqual(set(b["region"]), {"CH", "EU", "ALL"})
        o = pd.concat(oofs, ignore_index=True)
        ch = o[o["eval"] == "ch_query"]
        self.assertTrue(ch["ticker"].isin(self.sd.tickers[:self.sd.n_ch] + sc.mcap_inputs(self.swiss)[0]).all())
        self.assertEqual(set(ch["region"]), {"CH"})
        # the pooled ridge beats the pooled mean on a planted-factor panel
        r2 = b[(b.metric == "r2") & (b.region == "ALL")].set_index("model")["mean"]
        self.assertGreater(r2["all_ridge"], r2["baseline"])

    def test_run_writes_schema_and_reproduction_check(self):
        with tempfile.TemporaryDirectory() as out:
            res = ms.run(self.swiss, self.panels, self.table, self.universe, repeats=1, out_dir=out, n_boot=20,
                         repro=True, verbose=False)
            out = Path(out)
            self.assertEqual(list(res.columns), ms.COLUMNS)
            back = pd.read_csv(out / "multi_static.csv")
            self.assertEqual(list(back.columns), ms.COLUMNS)
            self.assertEqual(set(back["eval"]), {"ch_query", "pooled"})
            oof = pd.read_parquet(out / "multi_static_oof.parquet")
            self.assertEqual(list(oof.columns), ["eval", "target", "model", "repeat", "ticker", "region", "y_true",
                                                 "y_pred", "y_true_num", "y_pred_num"])
            ch = oof[oof["eval"] == "ch_query"]
            self.assertTrue(ch["ticker"].str.endswith(".SW").all())
            sel = pd.read_csv(out / "multi_static_selection.csv")
            self.assertEqual(list(sel.columns), ms.SELECTION_COLUMNS)
            self.assertEqual(set(sel["eval"]), {"ch_query", "pooled"})
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                             ["multi_static.csv", "multi_static_oof.parquet", "multi_static_selection.csv"])

    def test_reproduction_check(self):
        sd = self.sd
        st, sy, sX, _ = sc.mcap_inputs(self.swiss)
        ctrl = ms.control_rows(sd, (st, sy, sX), repeats=1, n_boot=20)
        self.assertEqual(set(ctrl["model"]), {"knn_auto", "ridge", "lgbm"})
        # the old-fold control is exactly the per-ticker static_cv computation
        direct = sc.run_sector_repeat(sd.ctrl_C, sd.y[:sd.n_ch], sd.ctrl_fe, len(sd.classes), 0, sd.ctrl_mats)
        nc = len(sd.classes)
        want = sc.class_metrics(sd.y[:sd.n_ch], direct["knn_auto"], nc)["macro_f1"]
        got = ctrl[(ctrl.model == "knn_auto") & (ctrl.metric == "macro_f1")]["mean"].iloc[0]
        self.assertAlmostEqual(got, want, places=12)
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "static_cv.csv"
            self.assertIsNone(ms.reproduction_check(ctrl, ref))
            ctrl.to_csv(ref, index=False)
            self.assertLessEqual(ms.reproduction_check(ctrl, ref), 1e-12)
            bad = ctrl.copy()
            bad.loc[bad.index[0], "mean"] += 1e-3
            bad.to_csv(ref, index=False)
            self.assertGreater(ms.reproduction_check(ctrl, ref), 1e-4)
            other = ctrl.assign(n_repeats=10)
            other.to_csv(ref, index=False)
            self.assertIsNone(ms.reproduction_check(ctrl, ref))


class CliTests(unittest.TestCase):
    def test_main_parses_and_forwards(self):
        with mock.patch.object(ms.mm, "load_markets", return_value=("s", {}, "t", "d")) as lm, \
                mock.patch.object(ms.data, "load_universe", return_value="u"), \
                mock.patch.object(ms, "run", return_value=pd.DataFrame({"metric": ["macro_f1"], "n_repeats": [2]})) as rn:
            ms.main(["--repeats", "2", "--max-foreign", "50", "--out-dir", "/x/y", "--skip-repro"])
        self.assertEqual(lm.call_args.args[2], ("EU",))
        args = rn.call_args
        self.assertEqual(args.args[4], 2)
        self.assertEqual(args.args[7], 50)
        self.assertFalse(args.kwargs["repro"])


if __name__ == "__main__":
    unittest.main()
