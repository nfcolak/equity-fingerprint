"""Frozen fingerprint embeddings of every Swiss benchmark observation.

CLI: python -m equity_fingerprint.fingerprint.embed --ckpt PATH --arm {ssl,ssl_aux,ssl_far,ssl_aux_far} --out-dir DIR [--device cpu]

For each month-end observation (pos, ticker) of the benchmark the 252-day window ending at ``pos``
(inclusive, past data only) goes through the frozen encoder in eval mode. Writes
``fp_embeddings_<arm>.parquet`` (``pos, ticker, date, fp_0 .. fp_<d-1>``) and ``fp_embeddings_<arm>.json``
(the checkpoint's ``pretrain`` dict plus ``ckpt_sha256``, ``arm``, ``n_obs``) to ``--out-dir``. A checkpoint with a
comove head also gives ``fp_comove_<arm>.parquet`` (``pos, ticker, date, fc_0 .. fc_63``, unit-norm rows; the JSON
gets ``comove_dims``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F_

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
def project_comove(head, emb: torch.Tensor) -> np.ndarray:
    """L2-normalised comove head output for a batch of CLS embeddings [B, d] -> float32 [B, 64]."""
    return F_.normalize(head(emb.float()).float(), dim=-1).cpu().numpy()


@torch.no_grad()
def embed_windows(enc, ch, scaler, t_end, j, device, batch: int = BATCH, head=None):
    """CLS embeddings [N, d] and, when ``head`` is given, the unit-norm comove vectors [N, 64] (else None)."""
    out, com = [], []
    for i in range(0, len(t_end), batch):
        x = torch.from_numpy(window_batch(ch, t_end[i:i + batch], j[i:i + batch], scaler)).to(device)
        e = enc(x)[0]
        out.append(e.float().cpu().numpy())
        if head is not None:
            com.append(project_comove(head, e))
    return np.concatenate(out), (np.concatenate(com) if head is not None else None)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arm", choices=["ssl", "ssl_aux", "ssl_far", "ssl_aux_far"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    from .model import load_comove_head
    enc, ckpt = load_encoder(args.ckpt, device)
    head = load_comove_head(args.ckpt, device)
    obs, ch, tickers = observation_keys()
    j = obs["ticker"].map({t: k for k, t in enumerate(tickers)}).to_numpy()
    emb, com = embed_windows(enc, ch, ckpt["scaler"], obs["pos"].to_numpy(), j, device, head=head)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([obs.reset_index(drop=True),
                    pd.DataFrame(emb, columns=[f"fp_{k}" for k in range(emb.shape[1])])], axis=1)
    df.to_parquet(out / f"fp_embeddings_{args.arm}.parquet", index=False)
    side = {**ckpt["pretrain"], "ckpt_sha256": sha256_file(args.ckpt), "arm": args.arm, "n_obs": int(len(df))}
    if com is not None:
        cdf = pd.concat([obs.reset_index(drop=True),
                         pd.DataFrame(com, columns=[f"fc_{k}" for k in range(com.shape[1])])], axis=1)
        cdf.to_parquet(out / f"fp_comove_{args.arm}.parquet", index=False)
        side["comove_dims"] = int(com.shape[1])
    (out / f"fp_embeddings_{args.arm}.json").write_text(json.dumps(side, indent=2, default=str))
    print(f"[embed] {args.arm}: {len(df)} observations x {emb.shape[1]} dims -> {out}")
    if com is not None:
        print(f"[embed] {args.arm}: comove head -> fp_comove_{args.arm}.parquet ({com.shape[1]} dims)")


if __name__ == "__main__":
    main()
