"""fp_profile_features (no self reference, class signal, train-only references) and the comove projection."""
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn

from equity_fingerprint.embed import fp_profile_features
from equity_fingerprint.fingerprint.embed import embed_windows
from equity_fingerprint.fingerprint.model import FingerprintEncoder
from equity_fingerprint.fingerprint.windows import WINDOW

FC = [f"fc_{k}" for k in range(64)]


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _panel(n_per=6, n_pos=3, seed=0):
    """Three class directions in 64-d; ticker vectors = direction + noise; the first 4 tickers per class are train."""
    rng = np.random.default_rng(seed)
    cls = ["Alpha", "Beta", "Gamma"]
    dirs = _unit(rng.normal(size=(3, 64)))
    sec, train, keys, vecs = {}, [], [], []
    for c, name in enumerate(cls):
        for k in range(n_per):
            t = f"{name[0]}{k}"
            sec[t] = name
            if k < 4:
                train.append(t)
            for pos in range(n_pos):
                keys.append((pos * 21 + 300, t))
                vecs.append(_unit(dirs[c] + 0.2 * rng.normal(size=64)))
    df = pd.concat([pd.DataFrame(keys, columns=["pos", "ticker"]), pd.DataFrame(np.stack(vecs), columns=FC)], axis=1)
    return df, train, sec, cls


class FpProfileTest(unittest.TestCase):
    def test_class_signal_held_out(self):
        df, train, sec, cls = _panel()
        out, names = fp_profile_features(df, train, sec, m=3, K=5)
        self.assertEqual(names, [f"fpprof_{k}_{c.lower()}" for c in cls for k in ("knn", "vote")])
        knn = out[[n for n in names if n.startswith("fpprof_knn_")]].to_numpy()
        vote = out[[n for n in names if n.startswith("fpprof_vote_")]].to_numpy()
        held = ~out["ticker"].isin(train).to_numpy()
        true = out["ticker"].map(sec).map({c: i for i, c in enumerate(cls)}).to_numpy()
        self.assertTrue(held.any())
        self.assertTrue((knn[held].argmax(1) == true[held]).all())
        self.assertTrue((vote[held].argmax(1) == true[held]).all())

    def test_no_self_reference(self):
        df, train, sec, _ = _panel()
        # twin: a different training ticker with A0's exact vectors
        twin = df[df["ticker"] == "A0"].assign(ticker="A0x")
        out, _ = fp_profile_features(pd.concat([df, twin], ignore_index=True), train + ["A0x"],
                                     {**sec, "A0x": "Alpha"}, m=1, K=1)
        self.assertTrue(np.allclose(out[out["ticker"] == "A0"]["fpprof_knn_alpha"], 1.0))  # via the twin
        out1, _ = fp_profile_features(df, train, sec, m=1, K=1)
        self.assertTrue((out1[out1["ticker"] == "A0"]["fpprof_knn_alpha"] < 0.999).all())  # not via itself

    def test_held_out_never_references(self):
        df, train, sec, _ = _panel()
        mod = df.copy()
        a0 = df[df["ticker"] == "A0"].set_index("pos")[FC]
        for t in (t for t in sec if t not in train):
            idx = mod["ticker"] == t
            mod.loc[idx, FC] = a0.loc[mod.loc[idx, "pos"]].to_numpy()  # held-out copies of A0's vector
        a, _ = fp_profile_features(df, train, sec, m=2, K=3)
        b, _ = fp_profile_features(mod, train, sec, m=2, K=3)
        tr = a["ticker"].isin(train).to_numpy()
        cols = [c for c in a.columns if c.startswith("fpprof_")]
        np.testing.assert_allclose(a.loc[tr, cols].to_numpy(), b.loc[tr, cols].to_numpy())

    def test_nan_when_class_has_no_reference_that_day(self):
        df, train, sec, _ = _panel()
        gone = df["pos"].eq(300) & df["ticker"].isin([t for t in train if sec[t] == "Beta"])
        out, _ = fp_profile_features(df[~gone], train, sec, m=3, K=5)
        d0 = out[out["pos"] == 300]
        self.assertTrue(d0["fpprof_knn_beta"].isna().all())
        self.assertTrue((d0["fpprof_vote_beta"] == 0).all())
        self.assertFalse(out[out["pos"] != 300]["fpprof_knn_beta"].isna().any())


class ComoveProjectionTest(unittest.TestCase):
    def test_unit_norm_64(self):
        torch.manual_seed(0)
        enc = FingerprintEncoder(d_model=16, n_layers=1, n_heads=2).eval()
        n_ch = enc.config["n_channels"]
        head = nn.Sequential(nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 64)).eval()
        rng = np.random.default_rng(0)
        ch = rng.normal(size=(WINDOW + 40, 3, n_ch - 1)).astype(np.float32)
        scaler = {"median": np.zeros(n_ch - 1), "iqr": np.ones(n_ch - 1)}
        t_end = np.array([WINDOW, WINDOW + 10, WINDOW + 30, WINDOW + 5, WINDOW + 20])
        j = np.array([0, 1, 2, 0, 1])
        emb, com = embed_windows(enc, ch, scaler, t_end, j, torch.device("cpu"), batch=2, head=head)
        self.assertEqual(emb.shape, (5, 16))
        self.assertEqual(com.shape, (5, 64))
        np.testing.assert_allclose(np.linalg.norm(com, axis=1), 1.0, atol=1e-5)
        _, none = embed_windows(enc, ch, scaler, t_end, j, torch.device("cpu"))
        self.assertIsNone(none)


if __name__ == "__main__":
    unittest.main()
