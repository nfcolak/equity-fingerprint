"""report.py v5 section: absent without v5 rows, present with them.

Synthetic metrics.csv / robustness.csv / run_info.json in a temp dir; build functions called directly.
Run: env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_report_v5.py' -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from equity_fingerprint import report

V14 = ["baseline", "linear_raw", "linear_fe", "linear_fe2", "linear_fe3", "lgbm_raw", "lgbm_fe", "lgbm_fe2", "lgbm_fe3",
       "linear_har", "lgbm_har", "linear_fp", "lgbm_fp", "linear_fpa", "lgbm_fpa", "lgbm_fefpa", "fpnet", "fpnet_a"]
V5_ALL = ["linear_fpf", "lgbm_fpf", "linear_fpfa", "lgbm_fpfa", "lgbm_fefpfa"]
V5_SECTOR = ["linear_fpprof", "lgbm_fpprof", "linear_fe3fpprof", "lgbm_fe3fpprof"]
FEATURES = {"baseline": "none", "linear_raw": "raw", "lgbm_raw": "raw", "linear_fe": "fe", "lgbm_fe": "fe",
            "linear_fe2": "fe2", "lgbm_fe2": "fe2", "linear_fe3": "fe3", "lgbm_fe3": "fe3", "linear_har": "har",
            "lgbm_har": "har", "linear_fp": "fp", "lgbm_fp": "fp", "linear_fpa": "fpa", "lgbm_fpa": "fpa",
            "lgbm_fefpa": "fefpa", "fpnet": "fp_e2e", "fpnet_a": "fpa_e2e", "linear_fpf": "fpf", "lgbm_fpf": "fpf",
            "linear_fpfa": "fpfa", "lgbm_fpfa": "fpfa", "lgbm_fefpfa": "fefpfa", "linear_fpprof": "fpprof",
            "lgbm_fpprof": "fpprof", "linear_fe3fpprof": "fe3fpprof", "lgbm_fe3fpprof": "fe3fpprof"}
METRICS = {"vol_1m": ["r2", "mae", "qlike"], "beta_1y": ["r2", "mae"], "log_mcap": ["r2", "mae"],
           "liquidity_1m": ["r2", "mae"],
           "sector": ["macro_f1", "accuracy", "ticker_macro_f1", "ticker_accuracy"]}
SEEDS = [1, 2, 3, 4, 42]


def make_metrics(rng, v5: bool) -> pd.DataFrame:
    rows = []
    for t, mets in METRICS.items():
        models = V14 + (V5_ALL + (V5_SECTOR if t == "sector" else []) if v5 else [])
        for mod in models:
            for split in ("val", "test"):
                for met in mets:
                    rows.append((t, mod, FEATURES[mod], split, met, float(rng.uniform(0.1, 0.9)), 100))
    return pd.DataFrame(rows, columns=["target", "model", "features", "split", "metric", "value", "n_obs"])


def make_robustness(rng, v5: bool) -> pd.DataFrame:
    rows = []
    for t, met in (("log_mcap", "r2"), ("sector", "macro_f1")):
        models = V14 + (V5_SECTOR if v5 and t == "sector" else [])
        for mod in models:
            for seed in SEEDS:
                for split in ("val", "test"):
                    rows.append((t, mod, FEATURES[mod], seed, split, met, float(rng.uniform(0.1, 0.9)), 33))
    return pd.DataFrame(rows, columns=["target", "model", "features", "seed", "split", "metric", "value", "n_tickers"])


def write_results(d: Path, v5: bool) -> None:
    rng = np.random.default_rng(0)
    make_metrics(rng, v5).to_csv(d / "metrics.csv", index=False)
    make_robustness(rng, v5).to_csv(d / "robustness.csv", index=False)
    arm = {"n_tickers": 11, "max_date": "2021-12-31", "steps": 7, "ckpt_sha256": "ab" * 32}
    info = {"run_date": "2026-01-01", "git_sha": "0123456789", "n_tickers": 5, "n_observations_total": 50,
            "observation_range": ["2010-01-31", "2025-12-31"], "runtime_seconds": 600.0,
            "n_obs": {t: {"train": 10, "val": 5, "test": 5} for t in report.TARGETS},
            "fingerprint": {"ssl": arm, "ssl_aux": arm}}
    if v5:
        info["fingerprint"].update({"ssl_far": arm, "ssl_aux_far": arm})
    (d / "run_info.json").write_text(json.dumps(info))


class ReportV5Test(unittest.TestCase):
    def test_without_v5_rows_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            write_results(d, v5=False)
            self.assertEqual(report.build_v5(d), [])
            block = report.build_block(d)
            self.assertNotIn("### v5", block)
            self.assertIn("### v4", block)

    def test_with_v5_rows_section_present(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            write_results(d, v5=True)
            text = "\n".join(report.build_v5(d))
            self.assertIn("### v5", text)
            for mod in V5_ALL + V5_SECTOR:
                self.assertIn(mod, text)
            self.assertIn("best v1-v4 by validation", text)
            self.assertIn("`sector` robustness", text)
            self.assertIn("seed 42", text)
            self.assertIn("Δ vs linear_fe3", text)
            self.assertIn("Arm `ssl_far`", text)
            self.assertIn("Arm `ssl_aux_far`", text)
            self.assertIn("the best v5 model against the best v1-v4 model", text)
            block = report.build_block(d)
            self.assertIn("### v4", block)
            self.assertLess(block.index("### v4"), block.index("### v5"))

    def test_old_best_excludes_v5(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            write_results(d, v5=True)
            m = pd.read_csv(d / "metrics.csv")
            for t in report.TARGETS:
                self.assertNotIn(report._best_old(m, t), report.V5_MODELS)
                self.assertNotIn(report._best_v1_v4(m, t), report.V5_MODELS)

    def test_v4_text_unchanged_by_v5_rows(self):
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as td2:
            a, b = Path(td), Path(td2)
            write_results(a, v5=False)
            write_results(b, v5=True)
            # v5 rows are appended after the v1-v4 rows; the rest of the data is the same draw sequence for v1-v4 only
            # when v5 rows do not interleave, so compare the v4 section on the v1-v4 subset of the v5 frame.
            m = pd.read_csv(b / "metrics.csv")
            rb = pd.read_csv(b / "robustness.csv")
            m[~m.model.isin(report.V5_MODELS)].to_csv(a / "metrics.csv", index=False)
            rb[~rb.model.isin(report.V5_MODELS)].to_csv(a / "robustness.csv", index=False)
            self.assertEqual(report.build_v4(a), report.build_v4(b))


if __name__ == "__main__":
    unittest.main()
