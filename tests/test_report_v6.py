"""report.build_v6: headline section from static_cv.csv / ceilings.csv; absent files -> no section."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from equity_fingerprint import report


def _metrics(d: Path):
    rows = []
    for t in ("vol_1m", "beta_1y", "liquidity_1m"):
        for mod, v_val, v_test in (("baseline", 0.0, 0.0), ("linear_fe", 0.5, 0.6), ("lgbm_fe", 0.4, 0.7)):
            for split, v in (("val", v_val), ("test", v_test)):
                rows.append(dict(target=t, model=mod, features="fe", split=split, metric="r2", value=v, n_obs=10))
    pd.DataFrame(rows).to_csv(d / "metrics.csv", index=False)


class BuildV6Test(unittest.TestCase):
    def test_absent_files_give_no_section(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            _metrics(d)
            self.assertEqual(report.build_v6(d), [])

    def test_tables_from_files(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            _metrics(d)
            pd.DataFrame([dict(target="sector", model="knn", metric=m, mean=0.5, sd=0.02, ci_lo=0.4, ci_hi=0.6,
                               n_stocks=200, n_repeats=10, n_classes=9)
                          for m in ("macro_f1", "weighted_f1", "accuracy", "balanced_accuracy")]
                         + [dict(target="log_mcap", model="ridge", metric=m, mean=0.9, sd=0.01, ci_lo=0.85, ci_hi=0.92,
                                 n_stocks=204, n_repeats=10, n_classes=None) for m in ("r2", "mae")]
                         ).to_csv(d / "static_cv.csv", index=False)
            pd.DataFrame([dict(target=t, split="test", n_obs=10, split_half_r=0.6, reliability=0.8,
                               persistence_ols_r2=0.4, persistence_identity_r2=0.3, persistence_feature="x")
                          for t in ("vol_1m", "beta_1y", "liquidity_1m")]).to_csv(d / "ceilings.csv", index=False)
            text = "\n".join(report.build_v6(d))
            self.assertIn("| knn | 0.500 ± 0.020", text)
            self.assertIn("[0.400, 0.600]", text)
            self.assertIn("| ridge | 0.900 ± 0.010", text)
            # best by validation is linear_fe (val 0.5), test 0.6; skill (0.6-0.4)/0.6 = 0.33; reliability 0.8
            self.assertIn("| `vol_1m` | linear_fe | 0.600 | 0.400 | 0.33 | 0.800 |", text)
            self.assertNotIn("(v9)", text)   # no v9 file -> no v9 headline row
            v9 = [dict(target="log_mcap", model="ridge_md_ext", metric=m, mean=v, sd=0.003, ci_lo=lo, ci_hi=hi,
                       n_stocks=204, n_repeats=10, n_classes=None)
                  for m, v, lo, hi in (("r2", 0.924, 0.898, 0.942), ("mae", 0.437, 0.39, 0.48),
                                       ("delta_r2_vs_ridge", 0.031, 0.015, 0.047))]
            pd.DataFrame(v9).to_csv(d / "static_v9_mcap.csv", index=False)
            text = "\n".join(report.build_v6(d))
            self.assertIn("| ridge_md_ext (v9) | 0.924 ± 0.003 | 0.437 ± 0.003 | [0.898, 0.942] |", text)
            self.assertIn("+0.031 R² [+0.015, +0.047]", text)

    def test_forward_v7_table(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            _metrics(d)
            pd.DataFrame([dict(target="vol_1m", split="test", n_obs=10, split_half_r=0.6, reliability=0.8,
                               persistence_ols_r2=0.4, persistence_identity_r2=0.3, persistence_feature="x")]
                         ).to_csv(d / "ceilings.csv", index=False)
            rows = [dict(target="beta_1y", model="linear_fe", features="fe", split="val", metric="r2", value=0.66,
                         n_obs=5, ci_lo=None, ci_hi=None, selected=0),
                    dict(target="beta_1y", model="linear_fe", features="fe", split="test", metric="r2", value=0.63,
                         n_obs=5, ci_lo=None, ci_hi=None, selected=0),
                    dict(target="beta_1y", model="avg_fewelch", features="fewelch", split="val", metric="r2",
                         value=0.674, n_obs=5, ci_lo=None, ci_hi=None, selected=1),
                    dict(target="beta_1y", model="avg_fewelch", features="fewelch", split="test", metric="r2",
                         value=0.645, n_obs=5, ci_lo=None, ci_hi=None, selected=1),
                    dict(target="beta_1y", model="avg_fewelch", features="fewelch", split="test",
                         metric="delta_r2_vs_linear_fe", value=0.013, n_obs=5, ci_lo=-0.001, ci_hi=0.025, selected=1)]
            pd.DataFrame(rows).to_csv(d / "forward_v7.csv", index=False)
            text = "\n".join(report.build_v6(d))
            self.assertIn("| linear_fe | 0.660 | 0.630 | – | – |", text)
            self.assertIn("Source: `results/forward_v7.csv`", text)
            self.assertIn("| avg_fewelch ★ | 0.674 | 0.645 | +0.013 | [-0.001, +0.025] |", text)


if __name__ == "__main__":
    unittest.main()
