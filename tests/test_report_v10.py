"""report: issuer-grouped headline switch and the v10 section."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from equity_fingerprint import report


def _static(path, knn, ridge):
    rows = []
    for t, m, me, v in (("sector", "knn_auto", "macro_f1", knn), ("sector", "knn_auto", "weighted_f1", 0.6),
                        ("sector", "knn_auto", "accuracy", 0.6), ("sector", "knn_auto", "balanced_accuracy", 0.5),
                        ("log_mcap", "ridge", "r2", ridge), ("log_mcap", "ridge", "mae", 0.5)):
        rows.append(dict(target=t, model=m, metric=me, mean=v, sd=0.01, ci_lo=v - 0.05, ci_hi=v + 0.05,
                         n_stocks=200, n_repeats=10, n_classes=9 if t == "sector" else float("nan")))
    pd.DataFrame(rows).to_csv(path, index=False)


class ReportV10Test(unittest.TestCase):
    def test_grouped_headline_preferred(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _static(d / "static_cv.csv", 0.598, 0.892)
            text = "\n".join(report.build_v6(d))
            self.assertIn("Source: `results/static_cv.csv`", text)
            self.assertNotIn("issuer-grouped", text)
            _static(d / "static_cv_grouped.csv", 0.578, 0.891)
            text = "\n".join(report.build_v6(d))
            self.assertIn("Source: `results/static_cv_grouped.csv`", text)
            self.assertIn("issuer-grouped", text)
            self.assertIn("macro-F1 0.598 (now 0.578)", text)
            self.assertIn("| knn_auto | 0.578 ± 0.010", text)

    def test_v10_absent_and_sector_table(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.assertEqual(report.build_v10(d), [])
            rows = [dict(target="sector", model="ch_knn_auto", metric="macro_f1", mean=0.578, sd=0.017, ci_lo=0.505,
                         ci_hi=0.631, n_stocks=203, n_repeats=10, n_classes=9, eval="ch_query", region="CH"),
                    dict(target="sector", model="ch_eu", metric="macro_f1", mean=0.592, sd=0.014, ci_lo=0.507,
                         ci_hi=0.658, n_stocks=203, n_repeats=10, n_classes=9, eval="ch_query", region="CH"),
                    dict(target="sector", model="ch_eu", metric="delta_macro_f1_vs_ch_knn_auto", mean=0.015,
                         sd=0.016, ci_lo=-0.055, ci_hi=0.085, n_stocks=203, n_repeats=10, n_classes=9,
                         eval="ch_query", region="CH")]
            pd.DataFrame(rows).to_csv(d / "multi_static.csv", index=False)
            text = "\n".join(report.build_v10(d))
            self.assertIn("| ch_eu (main) | 0.592 ± 0.014 | [0.507, 0.658] | +0.015 [-0.055, +0.085] | – |", text)
            self.assertIn("### v10: European stocks, longer Swiss history", text)
            self.assertNotIn("multi_forward", text)

    def test_headline_beta_row_from_history_ext(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pd.DataFrame([dict(target="beta_1y", split="test", persistence_ols_r2=0.522, reliability=0.79)]
                         ).to_csv(d / "ceilings.csv", index=False)
            pd.DataFrame([dict(target="beta_1y", model="lgbm_fe", split=sp, metric="r2", value=v, n_obs=100)
                          for sp, v in (("val", 0.6), ("test", 0.63))]).to_csv(d / "metrics.csv", index=False)
            text = "\n".join(report.build_v6(d))
            self.assertNotIn("avg_long2011", text)
            h = [dict(target="beta_1y", model="avg_long2011", split="test", metric="r2", value=0.6504, ci_lo=None,
                      ci_hi=None),
                 dict(target="beta_1y", model="avg_long2011", split="test", metric="delta_r2_vs_ctrl", value=0.0157,
                      ci_lo=0.0058, ci_hi=0.0266)]
            pd.DataFrame(h).to_csv(d / "history_ext.csv", index=False)
            text = "\n".join(report.build_v6(d))
            self.assertIn("| `beta_1y` | lgbm_fe | 0.630 | 0.522 | 0.23 | 0.790 |", text)
            self.assertIn("| `beta_1y` | avg_long2011 (v10) | 0.650 | 0.522 | 0.27 | 0.790 |", text)
            self.assertIn("+0.016 [+0.006, +0.027]", text)


if __name__ == "__main__":
    unittest.main()
