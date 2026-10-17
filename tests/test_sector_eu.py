"""Synthetic tests for sector_eu: .SW rejection, EU stocks only references, cut date, schema."""
import torch  # noqa: F401  (macOS libomp: torch before lightgbm)
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from equity_fingerprint import sector_eu as eu
from equity_fingerprint import static_cv as sc

NC, PER, M_PER = 3, 12, 6
CLASSES = ["Energy", "Financials", "Health"]


def make_data(seed=0, T=700):
    rng = np.random.default_rng(seed)
    n, m = NC * PER, NC * M_PER
    y, y_eu = np.repeat(np.arange(NC), PER), np.repeat(np.arange(NC), M_PER)
    fac = rng.normal(size=(T, NC))
    idx = pd.bdate_range("2019-01-01", periods=T)
    ch = pd.DataFrame(fac[:, y] * 0.01 + rng.normal(size=(T, n)) * 0.01, index=idx, columns=[f"S{i}" for i in range(n)])
    r_eu = pd.DataFrame(fac[:, y_eu] * 0.01 + rng.normal(size=(T, m)) * 0.01, index=idx, columns=[f"E{i}.PA" for i in range(m)])
    return ch, r_eu, y, y_eu


class TestSectorEU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ch, cls.r_eu, cls.y, cls.y_eu = make_data()
        cls.cchch, cls.cheu = eu.cross_space(cls.ch, cls.r_eu)
        cls.cfull = sc.pairwise_corr(sc.resid_on(cls.ch, cls.r_eu.mean(axis=1).to_numpy())).to_numpy()

    def test_sw_ticker_raises(self):
        info = pd.DataFrame({"ticker": ["AAA.PA", "NESN.SW"], "sector": ["Energy", "Health"]})
        uni = pd.DataFrame({"ticker": ["AAA.PA", "NESN.SW"], "country": ["FR", "CH"]})
        with self.assertRaises(ValueError):
            eu.select_eu_universe(info, uni, ["AAA.PA", "NESN.SW"], CLASSES)
        with self.assertRaises(ValueError):
            eu.eu_returns(pd.DataFrame(), ["X.SW"], pd.DatetimeIndex([]))
        with self.assertRaises(ValueError):
            eu.run_sector_eu(self.cchch, self.cheu, self.cfull, self.y, self.y_eu, NC, 1, eu_tickers=["A.SW"])
        ok = eu.select_eu_universe(info.iloc[:1], uni.iloc[:1], ["AAA.PA"], CLASSES)
        self.assertEqual(list(ok.index), ["AAA.PA"])

    def test_universe_filters_us_unknown_sector_and_no_prices(self):
        info = pd.DataFrame({"ticker": list("ABCD"), "sector": ["Energy", "Energy", "Tech", "Health"]})
        uni = pd.DataFrame({"ticker": list("ABCD"), "country": ["DE", "US", "DE", "FR"]})
        sel = eu.select_eu_universe(info, uni, list("ABC"), CLASSES)
        self.assertEqual(list(sel.index), ["A"])

    def test_loading_from_env_dir_and_cut(self):
        ch, r_eu, _, _ = make_data(T=700)
        px = np.exp(r_eu.cumsum() * 0.5 + 3)
        px.iloc[0, 0] = np.nan
        long = px.stack().rename("adj_close").reset_index()
        long.columns = ["date", "ticker", "adj_close"]
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "raw").mkdir()
            long.to_parquet(root / "raw" / "prices.parquet")
            pd.DataFrame({"ticker": r_eu.columns, "sector": [CLASSES[i // M_PER] for i in range(len(r_eu.columns))]}
                         ).to_parquet(root / "raw" / "info.parquet")
            pd.DataFrame({"ticker": r_eu.columns, "country": "FR"}).to_csv(root / "universe.csv", index=False)
            old = os.environ.get("EF_PRETRAIN_DIR")
            os.environ["EF_PRETRAIN_DIR"] = t
            try:
                prices, info, uni = eu.load_eu_frames()
            finally:
                if old is None:
                    os.environ.pop("EF_PRETRAIN_DIR")
                else:
                    os.environ["EF_PRETRAIN_DIR"] = old
        sel = eu.select_eu_universe(info, uni, prices["ticker"].unique(), CLASSES)
        self.assertEqual(len(sel), NC * M_PER)
        cal = ch.index[ch.index <= pd.Timestamp("2020-12-31")]
        r = eu.eu_returns(prices, list(sel.index), cal, cut=pd.Timestamp("2020-06-30"), min_valid=100)
        self.assertTrue(r.index.equals(cal))
        self.assertTrue(r.loc[r.index > pd.Timestamp("2020-06-30")].isna().all().all())
        self.assertGreater(r.notna().sum().min(), 100)

    def test_eu_stocks_never_queries_and_schema(self):
        res = eu.run_sector_eu(self.cchch, self.cheu, self.cfull, self.y, self.y_eu, NC, repeats=2, n_boot=40)
        self.assertEqual(list(res.columns), eu.STATIC_COLUMNS + ["n_eu_refs"])
        self.assertTrue((res.n_stocks == len(self.y)).all())  # only Swiss stocks are scored
        self.assertTrue((res.n_eu_refs == len(self.y_eu)).all())
        self.assertEqual(set(res[~res.metric.str.startswith("delta_")].model), set(eu.EU_ARMS))
        d = res[res.metric.str.startswith("delta_")]
        self.assertEqual(set(d.metric), {f"delta_{m}_vs_ch_le2021" for m in eu.CLASS_METRICS})
        self.assertNotIn("ch_le2021", set(d.model))
        mf = res[(res.metric == "macro_f1")].set_index("model")["mean"]
        self.assertGreater(mf["ch_le2021"], 0.7)
        self.assertGreater(mf["eu_only"], 0.7)
        # fold output has one prediction per Swiss test stock, never per EU stock
        M = {"C_chch": self.cchch, "C_cheu": self.cheu, "C_full": self.cfull}
        te = np.arange(0, len(self.y), 4)
        tr = np.setdiff1d(np.arange(len(self.y)), te)
        out = eu.eu_fold(M, self.y, self.y_eu, tr, te, NC)
        for arm, p in out.items():
            self.assertEqual(len(p), len(te), arm)

    def test_swiss_test_labels_do_not_change_predictions(self):
        M = {"C_chch": self.cchch, "C_cheu": self.cheu, "C_full": self.cfull}
        te = np.arange(0, len(self.y), 4)
        tr = np.setdiff1d(np.arange(len(self.y)), te)
        y2 = self.y.copy()
        y2[te] = (y2[te] + 1) % NC
        a = eu.eu_fold(M, self.y, self.y_eu, tr, te, NC)
        b = eu.eu_fold(M, y2, self.y_eu, tr, te, NC)
        for arm in a:
            np.testing.assert_array_equal(a[arm], b[arm], err_msg=arm)

    def test_reference_rows_copied(self):
        df = pd.DataFrame([dict(zip(eu.STATIC_COLUMNS, ["sector", "knn_auto", "macro_f1", .5, .01, .4, .6, 10, 10, 3])),
                           dict(zip(eu.STATIC_COLUMNS, ["sector", "knn", "macro_f1", .4, .01, .3, .5, 10, 10, 3]))])
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "static_cv.csv"
            df.to_csv(f, index=False)
            ref = eu.reference_rows(f)
            self.assertIsNone(eu.reference_rows(Path(t) / "none.csv"))
        self.assertEqual(list(ref.model), ["knn_auto"])
        res = eu.run_sector_eu(self.cchch, self.cheu, self.cfull, self.y, self.y_eu, NC, 1, n_boot=20, ref_rows=ref)
        self.assertEqual(res.iloc[0].model, "knn_auto")
        self.assertTrue(np.isnan(res.iloc[0].n_eu_refs))

    def test_swiss_oof_paired_deltas(self):
        tickers = [f"S{i}" for i in range(len(self.y))]
        rows = [dict(target="sector", model=m, repeat=0, ticker=t, y_pred=CLASSES[v])
                for m in eu.SWISS_REF_MODELS for t, v in zip(tickers, self.y)]
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "oof.parquet"
            pd.DataFrame(rows).to_parquet(f, index=False)
            swiss = eu.swiss_oof(f, tickers, CLASSES, 1)
            self.assertIsNone(eu.swiss_oof(f, tickers, CLASSES, 2))   # missing repeat -> None
        np.testing.assert_array_equal(swiss["knn_auto"][0], self.y)
        res = eu.run_sector_eu(self.cchch, self.cheu, self.cfull, self.y, self.y_eu, NC, 1, n_boot=20, swiss_refs=swiss)
        d = res[res.metric == "delta_macro_f1_vs_knn_auto"]
        self.assertEqual(sorted(d.model), ["pooled_full", "pooled_full_prior"])
        self.assertTrue((d["mean"] <= 1e-12).all())   # the reference is the truth, so no arm can beat it


if __name__ == "__main__":
    unittest.main()
