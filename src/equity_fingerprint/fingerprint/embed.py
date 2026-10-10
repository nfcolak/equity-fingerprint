"""Frozen fingerprint embeddings of every Swiss benchmark observation.

CLI: python -m equity_fingerprint.fingerprint.embed --ckpt PATH --arm {ssl,ssl_aux} --out-dir DIR [--device cpu]

For each month-end observation (pos, ticker) of the benchmark the 252-day window ending at ``pos``
(inclusive, past data only) goes through the frozen encoder in eval mode. Writes
``fp_embeddings_<arm>.parquet`` (``pos, ticker, date, fp_0 .. fp_<d-1>``) and ``fp_embeddings_<arm>.json``
(the checkpoint's ``pretrain`` dict plus ``ckpt_sha256``, ``arm``, ``n_obs``) to ``--out-dir``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import features as F
from .model import load_encoder
from .windows import swiss_channels, window_batch

BATCH = 512


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def observation_keys() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Benchmark observations (pos, ticker, date), the Swiss channel tensor and its ticker order."""
    panel = F.build_panel()
    pos = F.month_end_positions(panel)
    obs = F.base_features(panel, pos)[["pos", "ticker", "date"]]
    ch, tickers = swiss_channels(panel)
    return obs, ch, tickers


@torch.no_grad()
def embed_windows(enc, ch, scaler, t_end, j, device, batch: int = BATCH) -> np.ndarray:
    out = []
    for i in range(0, len(t_end), batch):
        x = torch.from_numpy(window_batch(ch, t_end[i:i + batch], j[i:i + batch], scaler)).to(device)
        out.append(enc(x)[0].float().cpu().numpy())
    return np.concatenate(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arm", choices=["ssl", "ssl_aux"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    enc, ckpt = load_encoder(args.ckpt, device)
    obs, ch, tickers = observation_keys()
    j = obs["ticker"].map({t: k for k, t in enumerate(tickers)}).to_numpy()
    emb = embed_windows(enc, ch, ckpt["scaler"], obs["pos"].to_numpy(), j, device)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([obs.reset_index(drop=True),
                    pd.DataFrame(emb, columns=[f"fp_{k}" for k in range(emb.shape[1])])], axis=1)
    df.to_parquet(out / f"fp_embeddings_{args.arm}.parquet", index=False)
    side = {**ckpt["pretrain"], "ckpt_sha256": sha256_file(args.ckpt), "arm": args.arm, "n_obs": int(len(df))}
    (out / f"fp_embeddings_{args.arm}.json").write_text(json.dumps(side, indent=2, default=str))
    print(f"[embed] {args.arm}: {len(df)} observations x {emb.shape[1]} dims -> {out}")


if __name__ == "__main__":
    main()
