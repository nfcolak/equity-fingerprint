"""run.py v5 helpers are additive: no v5 files -> exactly the v4 sets; v5 files add only the new sets."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from equity_fingerprint import features as F
from equity_fingerprint import run as R

KEYS = ["pos", "ticker"]
TICKERS = ["A0", "A1", "B0", "B1", "C0", "C1"]
SEC = {"A0": "Alpha", "A1": "Alpha", "B0": "Beta", "B1": "Beta", "C0": "Gamma", "C1": "Gamma"}


def _emb(path: Path, n_dim: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = [(p, t) for p in (0, 1) for t in TICKERS]
    df = pd.DataFrame(rows, columns=KEYS)
    df["date"] = pd.Timestamp("2020-01-31")
    for k in range(n_dim):
        df[f"fp_{k}"] = rng.normal(size=len(df))
    df.to_parquet(path, index=False)


def _comove(path: Path):
    rng = np.random.default_rng(3)
    dirs = {"Alpha": rng.normal(size=64), "Beta": rng.normal(size=64), "Gamma": rng.normal(size=64)}
    rows = []
    for p in (0, 1):
        for t in TICKERS:
            v = dirs[SEC[t]] + 0.1 * rng.normal(size=64)
            rows.append([p, t, pd.Timestamp("2020-01-31"), *(v / np.linalg.norm(v))])
    pd.DataFrame(rows, columns=KEYS + ["date"] + [f"fc_{k}" for k in range(64)]).to_parquet(path, index=False)


class LoadFpTest(unittest.TestCase):
    def test_no_v5_files_equal_v4(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _emb(d / "fp_embeddings_ssl.parquet", 3, 0)
            _emb(d / "fp_embeddings_ssl_aux.parquet", 3, 1)
            frames, cols = R.load_fp(d)
            self.assertEqual(len(frames), 2)
            self.assertEqual(cols, {"fp": ["fp_0", "fp_1", "fp_2"], "fpa": ["fpa_0", "fpa_1", "fpa_2"]})
            self.assertEqual([n for n, _ in R.fp_sets(cols)], ["fp", "fpa", "fefpa"])
            self.assertEqual(dict(R.fp_sets(cols))["fefpa"], F.FE_FEATURES + cols["fpa"])
            self.assertIsNone(R.load_comove(d))

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as td:
            frames, cols = R.load_fp(Path(td))
            self.assertEqual((frames, cols), ([], {"fp": [], "fpa": []}))
            self.assertEqual(R.fp_sets(cols), [])

    def test_v5_files_add_only_new_sets(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _emb(d / "fp_embeddings_ssl.parquet", 3, 0)
            _emb(d / "fp_embeddings_ssl_aux.parquet", 3, 1)
            base_sets = R.fp_sets(R.load_fp(d)[1])
            _emb(d / "fp_embeddings_ssl_far.parquet", 2, 2)
            _emb(d / "fp_embeddings_ssl_aux_far.parquet", 2, 3)
            frames, cols = R.load_fp(d)
            self.assertEqual(len(frames), 4)
            self.assertEqual(cols["fpf"], ["fpf_0", "fpf_1"])
            self.assertEqual(cols["fpfa"], ["fpfa_0", "fpfa_1"])
            sets = R.fp_sets(cols)
            self.assertEqual(sets[:len(base_sets)], base_sets)
            self.assertEqual([n for n, _ in sets[len(base_sets):]], ["fpf", "fpfa", "fefpfa"])
            self.assertEqual(dict(sets)["fefpfa"], F.FE_FEATURES + cols["fpfa"])

    def test_only_one_far_arm(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _emb(d / "fp_embeddings_ssl_far.parquet", 2, 2)
            _, cols = R.load_fp(d)
            self.assertEqual([n for n, _ in R.fp_sets(cols)], ["fpf"])


class ComoveTest(unittest.TestCase):
    def test_load_comove_prefers_aux_far(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _comove(d / "fp_comove_ssl_far.parquet")
            self.assertEqual(len(R.load_comove(d)), 12)
            c = pd.read_parquet(d / "fp_comove_ssl_far.parquet")
            c.iloc[:6].to_parquet(d / "fp_comove_ssl_aux_far.parquet", index=False)
            self.assertEqual(len(R.load_comove(d)), 6)

    def test_add_fp_profile_train_only_and_order(self):
        with tempfile.TemporaryDirectory() as td:
            _comove(Path(td) / "c.parquet")
            comove = pd.read_parquet(Path(td) / "c.parquet")
        sector = pd.DataFrame([(p, t) for p in (0, 1) for t in TICKERS], columns=KEYS)
        sector["split"] = sector["ticker"].map({"A0": "train", "B0": "train", "C0": "train",
                                                "A1": "val", "B1": "test", "C1": "train"})
        sector["sector"] = sector["ticker"].map(SEC)
        ds = {"sector": sector, "log_mcap": sector.copy()}
        out, cols = R.add_fp_profile(ds, comove, SEC)
        self.assertTrue(all(c.startswith("fpprof_") for c in cols))
        self.assertEqual(len(cols), 6)  # knn + vote for 3 classes
        self.assertIs(out["log_mcap"], ds["log_mcap"])
        self.assertEqual(list(out["sector"][KEYS].itertuples(index=False)), list(sector[KEYS].itertuples(index=False)))
        self.assertEqual(list(out["sector"].columns), list(sector.columns) + cols)
        self.assertNotIn(cols[0], sector.columns)  # input frame untouched
        # val/test tickers never act as references: Alpha references are only A0, so A1 (val) votes Alpha from A0
        a1 = out["sector"].query("ticker == 'A1' and pos == 0").iloc[0]
        knn_alpha = [c for c in cols if c.startswith("fpprof_knn_") and "alpha" in c.lower()]
        self.assertEqual(len(knn_alpha), 1)
        self.assertGreater(a1[knn_alpha[0]], 0.5)
        # a train ticker never references itself: Beta has the single train ref B0, so B0's own Beta kNN is NaN
        b0 = out["sector"].query("ticker == 'B0' and pos == 0").iloc[0]
        knn_beta = [c for c in cols if c.startswith("fpprof_knn_") and "beta" in c.lower()][0]
        self.assertTrue(np.isnan(b0[knn_beta]))

    def test_fp_profile_sets(self):
        sets = R.fp_profile_sets(["a", "b"], ["p", "q"])
        self.assertEqual(sets, [("fpprof", ["p", "q"]), ("fe3fpprof", ["a", "b", "p", "q"])])


if __name__ == "__main__":
    unittest.main()
