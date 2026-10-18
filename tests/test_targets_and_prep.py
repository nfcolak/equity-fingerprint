"""Synthetic tests for the core no-leakage guards: the headline time split (boundaries and the rule that a train /
validation row whose forward window crosses the split end is dropped), `models.Prep` (clip bounds and scales
from the fitted rows only, `transform` never refits), and run-twice determinism of the static CV repeats."""
import torch  # noqa: F401  (macOS libomp: torch before lightgbm, which models / static_cv import)
import unittest

import numpy as np
import pandas as pd

from equity_fingerprint import static_cv as sc
from equity_fingerprint import targets as T
from equity_fingerprint.models import Prep

D = pd.Timestamp


class TimeSplitTest(unittest.TestCase):
    def test_boundaries_and_crossing_windows(self):
        rows = [  # (feature date, forward-window end, expected split)
            ("2021-11-30", "2021-12-31", "train"),   # window ends on the last train day
            ("2021-11-30", "2022-01-03", None),      # crosses into validation -> dropped
            ("2021-12-31", "2022-01-31", None),      # last train date, window in validation -> dropped
            ("2022-01-31", "2022-02-28", "val"),
            ("2023-11-30", "2023-12-29", "val"),
            ("2023-11-30", "2024-01-02", None),      # crosses into test -> dropped
            ("2024-01-31", "2024-02-29", "test"),
            ("2024-01-31", None, None),              # no forward window yet -> no test row
            ("2026-09-30", "2026-10-07", "test"),
        ]
        date = pd.Series([D(a) for a, _, _ in rows])
        end = pd.Series([D(b) if b else pd.NaT for _, b, _ in rows])
        got = T.time_split(date, end).tolist()
        want = [w for _, _, w in rows]
        self.assertEqual([g if isinstance(g, str) else None for g in got], want)
        self.assertEqual((T.TRAIN_END, T.VAL_START, T.VAL_END, T.TEST_START),
                         (D("2021-12-31"), D("2022-01-01"), D("2023-12-31"), D("2024-01-01")))


class PrepTest(unittest.TestCase):
    def test_bounds_come_from_the_fitted_rows_only(self):
        rng = np.random.default_rng(0)
        train = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
        other = pd.DataFrame({"a": rng.normal(size=200) * 50 + 100, "b": rng.normal(size=200) * 50})
        p = Prep(["a", "b"], linear=True).fit(train)
        lo, hi = p.lo_.copy(), p.hi_.copy()
        pd.testing.assert_series_equal(lo, train.quantile(0.01))
        pd.testing.assert_series_equal(hi, train.quantile(0.99))
        out = p.transform(other)
        pd.testing.assert_series_equal(p.lo_, lo)            # transform does not refit
        pd.testing.assert_series_equal(p.hi_, hi)
        clipped_max = (hi - p.mu_) / p.sd_                   # outliers end at the train 99% bound
        np.testing.assert_allclose(out.max().to_numpy(), clipped_max.to_numpy())
        p2 = Prep(["a", "b"], linear=True).fit(train)        # the other frame never changes the fit
        pd.testing.assert_series_equal(p2.mu_, p.mu_)


class DeterminismTest(unittest.TestCase):
    def test_static_cv_repeats_are_bit_identical(self):
        rng = np.random.default_rng(0)
        nc, per, T_ = 4, 12, 400
        n = nc * per
        y = np.repeat(np.arange(nc), per)
        ret = pd.DataFrame(rng.normal(size=(T_, nc))[:, y] * 0.01 + rng.normal(size=(T_, n)) * 0.01,
                           columns=[f"S{i:02d}" for i in range(n)])
        fe = pd.DataFrame(rng.normal(size=(n, 3)) + y[:, None] * 0.5, index=ret.columns, columns=["a", "b", "c"])
        C = sc.pairwise_corr(ret).to_numpy()
        a, b = sc.run_sector_repeat(C, y, fe, nc, 3), sc.run_sector_repeat(C, y, fe, nc, 3)
        for m in a:
            np.testing.assert_array_equal(a[m], b[m], err_msg=m)
        X = pd.DataFrame(rng.normal(size=(60, 4)), columns=["a", "b", "c", "log_tv_21"])
        yy = X["a"].to_numpy() * 2 + rng.normal(size=60) * 0.1
        a, b = sc.run_mcap_repeat(X, yy, 3), sc.run_mcap_repeat(X, yy, 3)
        for m in a:
            np.testing.assert_array_equal(a[m], b[m], err_msg=m)


if __name__ == "__main__":
    unittest.main()
