"""Synthetic tests for multi_market: FX to CHF (lag, ffill, GBp, pence override), foreign selection filters,
dedupe, eligibility, issuer groups, causality of features and CHF conversion, CHF-only panel identity."""
import torch  # noqa: F401  (macOS libomp: torch before lightgbm, which multi_market imports via static_cv)
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from equity_fingerprint import data
from equity_fingerprint import features as F
from equity_fingerprint import multi_market as mm

from _synth_markets import SECRET_TEXT, make_synth


def fx_series():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-07", "2020-01-08"])
    return pd.DataFrame({"EURCHF=X": [1.0, 1.1, 1.2, 1.3], "GBPCHF=X": [1.5, 1.5, 1.5, 1.5]}, index=dates)


class FxTests(unittest.TestCase):
    def test_load_fx_missing_and_env_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as cm:
                mm.load_fx(tmp)
            self.assertIn("--download-fx", str(cm.exception))
            (Path(tmp) / "raw").mkdir()
            long = fx_series().stack().rename("close").reset_index()
            long.columns = ["date", "ticker", "close"]
            long.to_parquet(Path(tmp) / "raw" / "fx.parquet", index=False)
            with mock.patch.dict(os.environ, {"EF_DATA_DIR": tmp}):
                fx = mm.load_fx()
        self.assertEqual(sorted(fx.columns), ["EURCHF=X", "GBPCHF=X"])
        self.assertTrue(fx.index.is_monotonic_increasing)

    def test_download_fx_uses_download_series(self):
        long = fx_series().stack().rename("close").reset_index()
        long.columns = ["date", "ticker", "close"]
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(data, "download_series", return_value=long) as ds:
                mm.download_fx(tmp)
            ds.assert_called_once_with(list(mm.FX_TICKERS.values()), "fx")
            self.assertEqual(len(mm.load_fx(tmp)), 4)

    def test_chf_rate_forward_fill_lag_and_gbp(self):
        fx = fx_series()
        cal = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"])
        r0 = mm.chf_rate(fx, "EUR", cal, lag=0)
        self.assertTrue(np.isnan(r0.iloc[0]))                      # no back fill
        self.assertEqual(r0.tolist()[1:], [1.0, 1.1, 1.1, 1.2, 1.3])  # forward fill only
        r1 = mm.chf_rate(fx, "EUR", cal)                             # default lag 1
        self.assertTrue(np.isnan(r1.iloc[:2]).all())
        self.assertEqual(r1.tolist()[2:], [1.0, 1.1, 1.1, 1.2])
        np.testing.assert_allclose(mm.chf_rate(fx, "GBp", cal, lag=0).iloc[1:], 0.015)
        np.testing.assert_allclose(mm.chf_rate(fx, "GBP", cal, lag=0).iloc[1:], 1.5)
        self.assertEqual(mm.chf_rate(fx, "CHF", cal).tolist(), [1.0] * 6)
        self.assertTrue(mm.chf_rate(fx, "CAD", cal).isna().all())

    def test_rate_never_uses_later_fx(self):
        fx = fx_series()
        cal = pd.bdate_range("2020-01-01", "2020-01-10")
        base = mm.chf_rate(fx, "EUR", cal, lag=1)
        bumped = fx.copy()
        bumped.loc["2020-01-07":, "EURCHF=X"] *= 5
        new = mm.chf_rate(bumped, "EUR", cal, lag=1)
        # the bump starts on 2020-01-07: with lag 1 the first affected calendar row is 2020-01-08
        pd.testing.assert_series_equal(base.loc[:"2020-01-07"], new.loc[:"2020-01-07"])
        self.assertNotEqual(base.loc["2020-01-08"], new.loc["2020-01-08"])

    def test_pence_override(self):
        tab = pd.DataFrame({"ticker": ["A.L", "B.L", "C.L", "D.DE"], "currency": ["GBP", "GBP", "GBp", "EUR"],
                            "last_close": [5000.0, 50.0, 5000.0, 5000.0],
                            "shares_outstanding": [1e6] * 4, "market_cap": [5e7, 5e7, 5e7, 5e7]})
        cur, log = mm.effective_currency(tab)
        self.assertEqual(cur.to_dict(), {"A.L": "GBp", "B.L": "GBP", "C.L": "GBp", "D.DE": "EUR"})
        self.assertEqual(log["ticker"].tolist(), ["A.L"])
        self.assertAlmostEqual(log["ratio"].iloc[0], 100.0)


class NameTests(unittest.TestCase):
    def test_region_of(self):
        self.assertEqual([mm.region_of(c) for c in ("CH", "Switzerland", "US", "United States", "DE", "France")],
                         ["CH", "CH", "US", "US", "EU", "EU"])

    def test_norm_name(self):
        self.assertEqual(mm.norm_name("Alphabet Inc. Class A"), "alphabet")
        self.assertEqual(mm.norm_name("ALPHABET INC - Series C"), "alphabet")
        self.assertEqual(mm.norm_name("Nestle S.A. ADR"), "nestle")
        self.assertEqual(mm.norm_name("Koninklijke Ahold Delhaize N.V."), "koninklijke ahold delhaize")
        self.assertEqual(mm.norm_name("Foo Holdings Group plc"), "foo")

    def test_issuer_groups_share_classes(self):
        uni = pd.DataFrame({"ticker": ["LISN.SW", "LISP.SW", "RO.SW", "ROP.SW", "NESN.SW"],
                            "company": ["Chocoladefabriken Lindt & Spruengli AG"] * 2 + ["Roche Holding AG"] * 2
                            + ["Nestle SA"]})
        names = {"GOOG": "Alphabet Inc. Class C", "GOOGL": "Alphabet Inc. Class A", "AAPL": "Apple Inc.",
                 "NESN.SW": "ignored foreign name"}
        tickers = ["LISN.SW", "LISP.SW", "RO.SW", "ROP.SW", "NESN.SW", "GOOG", "GOOGL", "AAPL", "MYSTERY"]
        g = mm.issuer_groups(tickers, uni, names)
        self.assertEqual(g[0], g[1])
        self.assertEqual(g[2], g[3])
        self.assertEqual(g[5], g[6])
        self.assertEqual(len(set(g)), 6)
        self.assertEqual(g.dtype.kind, "i")


class MarketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.s = make_synth(cls._tmp.name)
        cls.swiss, cls.panels, cls.table, cls.drops = mm.load_markets(cls.s["data_dir"], cls.s["pretrain_dir"],
                                                                      regions=("EU", "US"))  # generic code path
        cls.sp = cls.s["specials"]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # ---- selection
    def test_selection_rejects_filters(self):
        counts = self.table.attrs["counts"].set_index("filter")["all"]
        self.assertEqual(list(counts.index), ["universe", "in_info", "equity", "sector", "not_sw", "not_swiss_name",
                                              "domicile", "returns_504", "price_chf", "traded_value"])
        self.assertTrue((counts.diff().dropna() <= 0).all())
        for k in ("etf", "no_sector", "sw", "swiss_name", "non_domiciled", "short", "penny", "illiquid"):
            self.assertNotIn(self.sp[k], self.table["ticker"].tolist(), k)
        # each special is removed by its own filter
        uni = pd.read_csv(self.s["pretrain_dir"] / "universe.csv")
        info = pd.read_parquet(self.s["pretrain_dir"] / "raw" / "info.parquet")
        dom = mm.read_domicile(uni["ticker"], self.s["pretrain_dir"])
        fx = mm.load_fx(self.s["data_dir"])
        pr = mm.read_prices(self.s["pretrain_dir"] / "raw" / "prices.parquet", uni["ticker"])
        for k, drop_at in dict(etf="equity", no_sector="sector", sw="not_sw", swiss_name="not_swiss_name",
                               non_domiciled="domicile", short="returns_504", penny="price_chf",
                               illiquid="traded_value").items():
            sub = uni[uni["ticker"].isin([self.sp[k]] + self.s["us"][:3])]
            _, c = mm.select_foreign(sub, info, dom, pr, fx, ("EU", "US"))
            c = c.set_index("filter")["all"]
            first = c.index[(c.diff() < 0).to_numpy()][0]
            self.assertEqual(first, drop_at, k)
        for t in self.s["eu"] + self.s["us"]:
            self.assertIn(t, self.table["ticker"].tolist() + self.drops["ticker"].tolist())

    def test_region_floor_and_text_unused(self):
        self.assertTrue((self.table.loc[self.table.region == "US", "median_tv_chf"] >= 1e6).all())
        self.assertTrue((self.table.loc[self.table.region == "EU", "median_tv_chf"] >= 1e5).all())
        self.assertTrue((self.table["median_close_chf"] >= 5).all())
        self.assertNotIn(SECRET_TEXT, self.table.to_string())
        self.assertFalse(any(t.endswith(".SW") for t in self.table["ticker"]))
        self.assertTrue((self.table["domicile"].notna()).all())

    def test_pence_override_on_synthetic_line(self):
        cur = self.table.set_index("ticker")["currency"]
        pl = self.table.attrs["pence_log"]
        self.assertEqual(pl["ticker"].tolist(), [self.sp["pence_gbp"]])
        self.assertEqual(cur[self.sp["pence_gbp"]], "GBp")
        self.assertEqual(cur[self.sp["pounds"]], "GBP")
        self.assertEqual(cur[self.sp["pence_gbp_label"]], "GBp")
        med = self.table.set_index("ticker")["median_close_chf"]
        # pence line and pound line land in the same CHF range (GBP quotes of 2000-8000 pence = 22-90 CHF)
        self.assertTrue(10 < med[self.sp["pence_gbp"]] < 400)

    # ---- dedupe
    def test_dedupe_wedge_duplicate_dropped_similar_kept(self):
        d = self.drops.set_index("ticker")
        w = d.loc[self.sp["wedge_dup"]]
        self.assertEqual(w["partner"], self.sp["wedge_of"])
        self.assertEqual(w["reason"], "corr_swiss")
        self.assertGreaterEqual(w["corr"], 0.90)
        self.assertLess(w["corr"], 0.99)
        self.assertNotIn(self.sp["similar"], d.index)
        self.assertIn(self.sp["similar"], self.table["ticker"].tolist())
        # the kept similar line sits clearly below the threshold
        fx = mm.load_fx(self.s["data_dir"])
        pr = mm.read_prices(self.s["pretrain_dir"] / "raw" / "prices.parquet", [self.sp["similar"]])
        w_sim = mm.weekly_log_returns(pr, pd.Series({self.sp["similar"]: "EUR"}), fx)[self.sp["similar"]]
        ws = np.log(self.swiss.adj.resample("W-FRI").last()).diff()[self.sp["similar_of"]]
        c = pd.concat([w_sim, ws], axis=1).dropna().corr().iloc[0, 1]
        self.assertTrue(0.7 < c < 0.9, c)

    def test_dedupe_name_pass_and_us_eu_pair(self):
        d = self.drops.set_index("ticker")
        self.assertEqual(d.loc[self.sp["name_dup"], "reason"], "name_swiss")
        self.assertEqual(d.loc[self.sp["name_dup"], "partner"], "S05.SW")
        self.assertEqual(d.loc[self.sp["cross_us"], "partner"], self.sp["cross_eu"])
        self.assertEqual(d.loc[self.sp["cross_us"], "reason"], "corr")
        self.assertNotIn(self.sp["cross_eu"], d.index)
        kept = self.table["ticker"].tolist()
        self.assertNotIn(self.sp["cross_us"], kept)
        self.assertIn(self.sp["cross_eu"], kept)
        self.assertEqual(set(d.index), {self.sp["wedge_dup"], self.sp["name_dup"], self.sp["cross_us"]})

    def test_dedupe_unit_rules(self):
        tab = pd.DataFrame({"ticker": ["U1", "E1", "E2", "E3", "U2"],
                            "name": ["Alpha Inc", "Alpha SA", "Beta SA", "Beta Class B plc", "Gamma Inc"],
                            "region": ["US", "EU", "EU", "EU", "US"],
                            "median_tv_chf": [9e6, 2e5, 5e5, 3e5, 4e6]})
        w = pd.DataFrame(np.random.default_rng(0).normal(size=(120, 5)), columns=tab["ticker"].tolist())
        kept, log = mm.dedupe(tab, w, pd.Series({"SW.SW": "Delta AG"}))
        self.assertEqual(sorted(kept), ["E1", "E2", "U2"])         # US line of Alpha, lower-tv Beta line dropped
        lg = log.set_index("ticker")
        self.assertEqual(lg.loc["U1", "partner"], "E1")
        self.assertEqual(lg.loc["E3", "partner"], "E2")
        # an exact weekly copy of a Swiss stock goes, a < 104 week overlap never matches
        base = np.random.default_rng(1).normal(size=130)
        w2 = pd.DataFrame({"E1": base, "E2": np.random.default_rng(2).normal(size=130), "SW.SW": base,
                           "U2": np.r_[np.full(30, np.nan), base[30:] * 2][:130]})
        short = w2.copy()
        short.loc[short.index[:60], "U2"] = np.nan
        t2 = tab[tab["ticker"].isin(["E1", "E2", "U2"])].assign(name=["x1", "x2", "x3"])
        kept2, log2 = mm.dedupe(t2.reset_index(drop=True), short, pd.Series({"SW.SW": "Delta AG"}))
        self.assertEqual(sorted(kept2), ["E2", "U2"])
        self.assertEqual(log2["partner"].tolist(), ["SW.SW"])

    # ---- panels
    def test_panels_regions_and_markets(self):
        self.assertEqual(set(self.panels), {"^GDAXI", "^FTSE", "^FCHI", "^OMX", "^GSPC"})
        for k, p in self.panels.items():
            self.assertEqual(p.region, "US" if k == "^GSPC" else "EU")
            self.assertEqual(p.excluded_currency, {})
        self.assertEqual(sorted(sum((p.tickers for p in self.panels.values()), [])), sorted(self.table["ticker"]))

    def test_chf_only_panel_equals_build_panel(self):
        ref = F.build_panel(self.s["data_dir"])
        px, idx, info = data.load_prices(self.s["data_dir"]), data.load_index(self.s["data_dir"]), data.load_info(
            self.s["data_dir"])
        fx = mm.load_fx(self.s["data_dir"])
        rate = mm.rate_frame(fx, pd.Series("CHF", index=sorted(px["ticker"].unique())), sorted(px["date"].unique()))
        p = mm.build_market_panel(px, idx, info, "^SSHI", rate, "CH")
        self.assertEqual(p.tickers, ref.tickers)
        for k in ("calendar", "adj", "close", "open", "high", "low", "volume", "tv", "ret", "rows", "span"):
            a, b = getattr(p, k), getattr(ref, k)
            if isinstance(a, pd.DataFrame):
                pd.testing.assert_frame_equal(a, b)
            else:
                pd.testing.assert_index_equal(a, b)
        pd.testing.assert_series_equal(p.mkt, ref.mkt)
        pd.testing.assert_frame_equal(self.swiss.ret, ref.ret)
        pos = F.month_end_positions(ref)[-3:]
        pd.testing.assert_frame_equal(F.base_features(p, pos), F.base_features(ref, pos))
        self.assertEqual(p.region, "CH")

    def test_eu_panel_is_chf_and_local_returns(self):
        p = self.panels["^GDAXI"]
        px = mm.read_prices(self.s["pretrain_dir"] / "raw" / "prices.parquet", p.tickers, data.PRICE_COLS)
        t = p.tickers[0]
        loc = px[px.ticker == t].set_index("date")["close"]
        fx = mm.load_fx(self.s["data_dir"])
        rate = mm.chf_rate(fx, "EUR", p.calendar, lag=1)
        d = p.calendar[500]
        self.assertAlmostEqual(p.close.loc[d, t], loc.loc[d] * rate.loc[d], places=8)
        np.testing.assert_allclose(p.ret[t].dropna().to_numpy(), np.log(loc).diff().dropna().to_numpy(), atol=1e-12)

    def test_causality_of_features_and_chf_conversion(self):
        mk, p0 = "^GDAXI", self.panels["^GDAXI"]
        px = mm.read_prices(self.s["pretrain_dir"] / "raw" / "prices.parquet", p0.tickers, data.PRICE_COLS)
        idx = pd.read_parquet(self.s["pretrain_dir"] / "raw" / "index.parquet")
        idx["date"] = pd.to_datetime(idx["date"])
        idx = idx[idx.ticker == mk]
        info = pd.read_parquet(self.s["pretrain_dir"] / "raw" / "info.parquet")
        info = info[info.ticker.isin(p0.tickers)]
        fx = mm.load_fx(self.s["data_dir"])
        cur = self.table.set_index("ticker")["currency"].reindex(p0.tickers)
        D = p0.calendar[1000]
        rng = np.random.default_rng(5)

        def feats(px_, idx_, fx_, lag=1):
            rate = mm.rate_frame(fx_, cur, sorted(px_["date"].unique()), lag)
            p = mm.build_market_panel(px_, idx_, info, mk, rate, "EU")
            pos = np.arange(990, 1004)
            f = F.base_features(p, pos).replace([np.inf, -np.inf], np.nan)
            return p, f

        p, base = feats(px, idx, fx)
        px2 = px.copy()
        fut = px2["date"] > D
        for c in ("open", "high", "low", "close", "adj_close", "volume"):
            px2.loc[fut, c] = px2.loc[fut, c] * rng.uniform(0.5, 2.0, fut.sum())
        idx2 = idx.copy()
        idx2.loc[idx2["date"] > D, ["close", "adj_close"]] *= 3.0
        _, pert = feats(px2, idx2, fx)
        a = base[base["date"] <= D].reset_index(drop=True)
        b = pert[pert["date"] <= D].reset_index(drop=True)
        self.assertGreater(len(a), 10)
        pd.testing.assert_frame_equal(a, b)
        self.assertFalse(base[base["date"] > D].equals(
            pert[pert["date"] > D]))

        # FX perturbed from D on: with lag 1 nothing up to D changes, the next row does; with lag 0 D changes
        fxb = fx.copy()
        fxb.loc[fxb.index >= D, "EURCHF=X"] *= 1.5
        _, pf = feats(px, idx, fxb)
        pd.testing.assert_frame_equal(a, pf[pf["date"] <= D].reset_index(drop=True))
        nxt = p.calendar[1001]
        self.assertFalse(base[base["date"] == nxt].equals(
            pf[pf["date"] == nxt]))
        _, b0 = feats(px, idx, fx, lag=0)
        _, f0 = feats(px, idx, fxb, lag=0)
        self.assertFalse(b0[b0["date"] == D].equals(f0[f0["date"] == D]))

    def test_eligible_rows_ignores_future_traded_value(self):
        p = self.panels["^GSPC"]
        pos = np.array([400, 800, 1200])
        floor = float(p.tv.iloc[600:1201].median().median())
        e = mm.eligible_rows(p, pos, floor)
        self.assertEqual(list(e.index), list(p.calendar[pos]))
        self.assertEqual(list(e.columns), p.tickers)
        self.assertEqual(e.dtypes.unique().tolist(), [np.dtype(bool)])
        self.assertTrue(e.to_numpy().any() and not e.to_numpy().all())
        tv = p.tv.copy()
        tv.iloc[801:] = tv.iloc[801:] * 1000.0 + 1e9
        q = F.Panel(**{**p.__dict__, "tv": tv})
        e2 = mm.eligible_rows(q, pos, floor)
        pd.testing.assert_frame_equal(e.iloc[:2], e2.iloc[:2])
        self.assertTrue(e2.iloc[2].all())
        # trailing window: an early spike that left the 252-day window no longer counts
        tv3 = p.tv.copy()
        tv3.iloc[:800] = 1e12
        r = mm.eligible_rows(F.Panel(**{**p.__dict__, "tv": tv3}), np.array([1100]), 1e12)
        self.assertFalse(r.to_numpy().any())

    # ---- weekly residuals and stacking
    def test_weekly_resid_two_factors(self):
        w = pd.bdate_range("2021-01-04", periods=200)
        rng = np.random.default_rng(0)
        m1, ew_noise = rng.normal(0, 0.01, 200), rng.normal(0, 0.01, 200)

        def tiny(cols, mkt, scale):
            ret = pd.DataFrame({c: scale * mkt + 0.5 * ew_noise + rng.normal(0, 0.002, 200) for c in cols}, index=w)
            return F.Panel(ret=ret, mkt=pd.Series(mkt, index=w))

        a, b = tiny(["A", "B"], m1, 1.0), tiny(["C"], rng.normal(0, 0.01, 200), 1.0)
        a.ret.iloc[0:5, 0] = np.nan                       # week with < 3 valid days
        a.ret.iloc[5:9, 0] = np.nan                       # 1 valid day in that week
        r = mm.weekly_resid({"X": a, "Y": b})
        self.assertEqual(sorted(r.columns), ["A", "B", "C"])
        self.assertTrue(all(d.dayofweek == 4 for d in r.index))
        self.assertTrue(np.isnan(r["A"].iloc[0]) and np.isnan(r["A"].iloc[1]))
        self.assertLess(r["B"].std(), 0.012)             # removes the local factor (weekly raw std ~ 0.03)
        self.assertAlmostEqual(float(r["B"].mean()), 0.0, places=10)

    def test_stack_returns_union_calendar(self):
        s = mm.stack_returns({"a": self.panels["^GDAXI"], "b": self.panels["^GSPC"]})
        self.assertEqual(s.dtypes.unique().tolist(), [np.dtype("float32")])
        self.assertEqual(len(s.columns), len(self.panels["^GDAXI"].tickers) + len(self.panels["^GSPC"].tickers))
        self.assertTrue(s.index.equals(self.panels["^GDAXI"].calendar.union(self.panels["^GSPC"].calendar)))
        wk = mm.stack_returns({"a": self.panels["^GDAXI"]}, freq="W")
        self.assertTrue(all(d.dayofweek == 4 for d in wk.index))
        with self.assertRaises(ValueError):
            mm.stack_returns({"a": self.panels["^GDAXI"]}, freq="M")

    # ---- CLI
    def test_cli_summary_and_download(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mm.main(["--summary", "--data-dir", str(self.s["data_dir"]), "--pretrain-dir", str(self.s["pretrain_dir"])])
        out = buf.getvalue()
        for key in ("traded_value", "dedupe drops", "pence overrides", "Healthcare", "DUPW.DE", "GB04.L"):
            self.assertIn(key, out)
        long = fx_series().stack().rename("close").reset_index()
        long.columns = ["date", "ticker", "close"]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(data, "download_series", return_value=long):
            with contextlib.redirect_stdout(io.StringIO()):
                mm.main(["--download-fx", "--data-dir", tmp])
            self.assertTrue((Path(tmp) / "raw" / "fx.parquet").exists())

    def test_description_text_does_not_change_output(self):
        import json
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            pre = Path(tmp) / "pretrain"
            shutil.copytree(self.s["pretrain_dir"], pre)
            for f in (pre / "raw" / "info").glob("*.json"):
                j = json.loads(f.read_text())
                j["longBusinessSummary"] = "completely different text " * 20
                f.write_text(json.dumps(j))
            _, panels, tab, drops = mm.load_markets(self.s["data_dir"], pre, ("EU", "US"))
        pd.testing.assert_frame_equal(tab, self.table)
        pd.testing.assert_frame_equal(drops, self.drops)
        self.assertEqual(set(panels), set(self.panels))

    def test_regions_argument(self):
        self.assertEqual(mm.REGIONS, ("EU",))  # US stocks were removed from the v10 results (user decision)
        _, panels, tab, _ = mm.load_markets(self.s["data_dir"], self.s["pretrain_dir"])
        self.assertEqual(set(tab["region"]), {"EU"})
        self.assertNotIn("^GSPC", panels)


if __name__ == "__main__":
    unittest.main()
