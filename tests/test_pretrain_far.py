"""--positives far: sampler invariants, near-mode identity with the v4 sampler, heads in the checkpoint."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from equity_fingerprint.fingerprint import pretrain
from equity_fingerprint.fingerprint.model import load_comove_head, load_encoder
from equity_fingerprint.fingerprint.pretrain import FAR_MAX, FAR_MIN, MAX_SHIFT, Sampler, load_pretrain, window_ok

N_TICKERS, N_DAYS = 12, 1100


def make_pretrain_dir(pdir, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(end="2021-12-31", periods=N_DAYS)
    tickers = [f"T{i:02d}.L" for i in range(N_TICKERS)]
    rows = []
    for t in tickers:
        close = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, N_DAYS)))
        rows.append(pd.DataFrame({"date": days, "ticker": t, "open": close * 0.999, "high": close * 1.01,
                                  "low": close * 0.99, "close": close, "adj_close": close,
                                  "volume": rng.integers(1_000, 100_000, N_DAYS).astype(float)}))
    (pdir / "raw").mkdir(parents=True)
    pd.concat(rows).to_parquet(pdir / "raw" / "prices.parquet")
    idx = 1000 * np.exp(np.cumsum(rng.normal(0, 0.005, N_DAYS)))
    pd.DataFrame({"date": days, "ticker": "^TEST", "open": idx, "high": idx, "low": idx, "close": idx,
                  "adj_close": idx, "volume": 0.0}).to_parquet(pdir / "raw" / "index.parquet")
    pd.DataFrame({"ticker": tickers, "market_index": "^TEST"}).to_csv(pdir / "universe.csv", index=False)
    pd.DataFrame({"ticker": tickers, "sector": ["Tech", "Health", "Energy"] * (N_TICKERS // 3),
                  "shares_outstanding": 1e6, "currency": "GBp"}).to_parquet(pdir / "raw" / "info.parquet")


def v4_draw(s, rng):
    """Frozen copy of the v4 ``Sampler.draw``."""
    t = rng.choice(s.days, s.n_days, replace=len(s.days) < s.n_days)
    local = np.stack([rng.choice(np.flatnonzero(s.ok[d]), s.k, replace=False) for d in t])
    a = t - rng.integers(0, MAX_SHIFT + 1, s.n_days)
    b = t - rng.integers(0, MAX_SHIFT + 1, s.n_days)
    return np.repeat(a, s.k), np.repeat(b, s.k), s.pool[local].reshape(-1)


class PretrainFarTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.pdir = cls.root / "pretrain"
        make_pretrain_dir(cls.pdir)
        cls.data = load_pretrain(cls.pdir)
        cls.ok = window_ok(cls.data.ch)
        cls.pool = np.arange(len(cls.data.tickers) - 1)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_far_draws(self):
        s = Sampler(self.data, self.pool, self.ok, 64, positives="far")
        rng = np.random.default_rng(1)
        for _ in range(200):
            ta, tb, j = s.draw(rng)
            self.assertEqual(len(ta), s.n_days * s.k)
            gap = np.abs(ta - tb)
            self.assertTrue(((gap >= FAR_MIN) & (gap <= FAR_MAX)).all())
            self.assertTrue(self.ok[ta, j].all() and self.ok[tb, j].all())
            self.assertTrue((ta <= len(self.data.calendar) - 1).all() and (tb <= len(self.data.calendar) - 1).all())
            self.assertTrue((self.data.calendar[np.maximum(ta, tb)] <= pd.Timestamp("2021-12-31")).all())
            groups = j.reshape(s.n_days, s.k)
            self.assertTrue(all(len(set(g)) == s.k for g in groups))  # distinct tickers in a day group
            self.assertTrue((ta.reshape(s.n_days, s.k) == ta.reshape(s.n_days, s.k)[:, :1]).all())
        self.assertFalse(any(t.upper().endswith(".SW") for t in self.data.tickers))

    def test_near_matches_v4(self):
        s = Sampler(self.data, self.pool, self.ok, 64)
        self.assertEqual(s.positives, "near")
        for seed in (0, 7):
            got = s.draw(np.random.default_rng(seed))
            want = v4_draw(s, np.random.default_rng(seed))
            for g, w in zip(got, want):
                np.testing.assert_array_equal(g, w)

    def _run(self, out, *extra):
        pretrain.main(["--pretrain-dir", str(self.pdir), "--out", str(out), "--arm", "ssl", "--steps", "3",
                       "--batch", "32", "--device", "cpu", *extra])

    def test_far_checkpoint_has_heads(self):
        out = self.root / "out_far"
        self._run(out, "--positives", "far")
        path = out / "ckpt_ssl_far.pt"
        self.assertTrue(path.exists() and (out / "pretrain_log_ssl_far.json").exists())
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(ckpt["pretrain"]["positives"], "far")
        self.assertIn("comove.0.weight", ckpt["heads_state_dict"])
        self.assertFalse(any(n.startswith("encoder.") for n in ckpt["heads_state_dict"]))
        self.assertEqual(ckpt["heads_config"]["comove_out"], 64)
        head = load_comove_head(path, "cpu")
        self.assertIsNotNone(head)
        self.assertFalse(head.training)
        self.assertEqual(head(torch.zeros(2, ckpt["heads_config"]["d_model"])).shape, (2, 64))

        del ckpt["heads_state_dict"], ckpt["heads_config"]
        old = self.root / "ckpt_old.pt"
        torch.save(ckpt, old)
        self.assertIsNone(load_comove_head(old, "cpu"))
        enc, _ = load_encoder(old, "cpu")
        self.assertFalse(enc.training)

    def test_near_default_names(self):
        out = self.root / "out_near"
        self._run(out)
        self.assertTrue((out / "ckpt_ssl.pt").exists() and (out / "pretrain_log_ssl.json").exists())
        self.assertFalse((out / "ckpt_ssl_far.pt").exists())


if __name__ == "__main__":
    unittest.main()
