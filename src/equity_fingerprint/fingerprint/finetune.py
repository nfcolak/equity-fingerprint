"""End-to-end fine-tuning of the fingerprint encoder on the Swiss benchmark targets.

CLI: python -m equity_fingerprint.fingerprint.finetune --ckpt PATH --name {fpnet,fpnet_a} --labels PATH
         --out-dir DIR [--device cuda] [--seeds 42,1,2,3,4] [--max-epochs N]

``--labels`` is the ``labels.parquet`` written by ``run.py`` with ``EF_DUMP_LABELS``. The encoder starts from the
pretraining checkpoint; one linear head per target sits on the CLS embedding (4 regression heads, one sector
classifier with the classes of that seed's train rows). Seed 42 trains all five targets jointly (each loss only
on that target's train rows, regression targets standardised with the train mean / sd); the other seeds train
``log_mcap`` and ``sector`` with that seed's ticker split. AdamW (encoder lr 1e-4, heads lr 1e-3), early stopping
per target on its validation loss: the predictions of each target are kept from its best epoch.
Writes ``<name>_preds_seed<s>.parquet`` (``target, pos, ticker, split, pred`` / ``proba_<class>``, val and test
rows) and ``<name>_log_seed<s>.json`` to ``--out-dir``.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
from torch import nn

from .embed import observation_keys, sha256_file
from .model import load_encoder
from .windows import window_batch

KEYS = ["pos", "ticker"]
CODE = {"train": 0, "val": 1, "test": 2}
ENC_LR, HEAD_LR, WEIGHT_DECAY = 1e-4, 1e-3, 0.01
BATCH, EVAL_BATCH, PATIENCE = 256, 1024, 5
GROUPED = ["log_mcap", "sector"]


# --------------------------------------------------------------------------- task
def build_task(lab: pd.DataFrame, tickers: list[str]) -> dict:
    """Union of the observations of all targets of one seed, with per-target labels and split codes."""
    ti = {t: j for j, t in enumerate(tickers)}
    rows = lab[KEYS].drop_duplicates().reset_index(drop=True)
    index = pd.MultiIndex.from_frame(rows)
    targets = list(dict.fromkeys(lab["target"]))
    n = len(rows)
    task = {"targets": targets, "pos": rows["pos"].to_numpy(), "col": rows["ticker"].map(ti).to_numpy(),
            "ticker": rows["ticker"].to_numpy(), "code": np.full((len(targets), n), -1, np.int8), "y": {}, "info": {}}
    for k, t in enumerate(targets):
        d = lab[lab["target"] == t]
        at = index.get_indexer(pd.MultiIndex.from_frame(d[KEYS]))
        task["code"][k, at] = d["split"].map(CODE).to_numpy()
        train = d["split"] == "train"
        if t == "sector":
            classes = sorted(d.loc[train, "y"].unique())
            lookup = {c: i for i, c in enumerate(classes)}
            y = np.full(n, -1, np.int64)
            y[at] = d["y"].map(lookup).fillna(-1).to_numpy(np.int64)  # class unseen in train -> ignored
            task["info"][t] = {"classes": classes}
        else:
            v = d["y"].astype(float).to_numpy()
            mu, sd = float(v[train.to_numpy()].mean()), float(v[train.to_numpy()].std() or 1.0)
            y = np.full(n, np.nan)
            y[at] = (v - mu) / sd
            task["info"][t] = {"mu": mu, "sd": sd}
        task["y"][t] = y
    return task


class Net(nn.Module):
    def __init__(self, enc, task):
        super().__init__()
        self.enc = enc
        d = enc.config["d_model"]
        self.heads = nn.ModuleDict({t: nn.Linear(d, len(task["info"][t]["classes"]) if t == "sector" else 1)
                                    for t in task["targets"]})

    def forward(self, x, amp: bool):
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=amp):
            emb = self.enc(x)[0]
        emb = emb.float()
        return {t: (h(emb) if t == "sector" else h(emb)[:, 0]) for t, h in self.heads.items()}


def make_x(ch, scaler, task, idx, device):
    return torch.from_numpy(window_batch(ch, task["pos"][idx], task["col"][idx], scaler)).to(device)


@torch.no_grad()
def predict(net, ch, scaler, task, idx, device, amp) -> dict:
    net.eval()
    out = {t: [] for t in task["targets"]}
    for i in range(0, len(idx), EVAL_BATCH):
        o = net(make_x(ch, scaler, task, idx[i:i + EVAL_BATCH], device), amp)
        for t, v in o.items():
            out[t].append(v.cpu().numpy())
    return {t: np.concatenate(v) for t, v in out.items()}


def target_loss(t, out, y):
    if t == "sector":
        return Fn.cross_entropy(out, y, ignore_index=-1)
    return Fn.mse_loss(out, y)


# --------------------------------------------------------------------------- one seed
def finetune_seed(args, lab, ch, tickers, device, seed: int) -> tuple[pd.DataFrame, dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    enc, ckpt = load_encoder(args.ckpt, device)
    scaler = ckpt["scaler"]
    task = build_task(lab, tickers)
    targets, code = task["targets"], task["code"]
    net = Net(enc, task).to(device)
    opt = torch.optim.AdamW([{"params": net.enc.parameters(), "lr": ENC_LR},
                             {"params": net.heads.parameters(), "lr": HEAD_LR}], weight_decay=WEIGHT_DECAY)
    amp = device.type == "cuda"
    gscaler = torch.amp.GradScaler("cuda", enabled=amp)
    y_t = {t: torch.from_numpy(task["y"][t]).to(device) for t in targets}
    code_t = torch.from_numpy(code.astype(np.int64)).to(device)

    train_idx = np.flatnonzero((code == CODE["train"]).any(0))
    eval_idx = np.flatnonzero(((code == CODE["val"]) | (code == CODE["test"])).any(0))
    best = {t: {"val_loss": np.inf, "epoch": 0, "out": None} for t in targets}
    epoch_seconds = []
    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        net.train()
        perm = rng.permutation(train_idx)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            out = net(make_x(ch, scaler, task, b, device), amp)
            bt = torch.from_numpy(b).to(device)
            loss = 0.0
            for k, t in enumerate(targets):
                m = code_t[k, bt] == CODE["train"]
                if m.any():
                    loss = loss + target_loss(t, out[t][m], y_t[t][bt][m])
            opt.zero_grad(set_to_none=True)
            gscaler.scale(loss).backward()
            gscaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            gscaler.step(opt)
            gscaler.update()
        out = predict(net, ch, scaler, task, eval_idx, device, amp)
        msg = []
        for k, t in enumerate(targets):
            m = code[k, eval_idx] == CODE["val"]
            vl = float(target_loss(t, torch.from_numpy(out[t][m]), torch.from_numpy(task["y"][t][eval_idx][m])))
            msg.append(f"{t} {vl:.4f}")
            if vl < best[t]["val_loss"]:
                best[t] = {"val_loss": vl, "epoch": epoch, "out": out[t]}
        epoch_seconds.append(round(time.time() - t0, 1))
        print(f"[{args.name} seed {seed}] epoch {epoch} ({epoch_seconds[-1]}s) val loss: " + ", ".join(msg), flush=True)
        if all(epoch - best[t]["epoch"] >= PATIENCE for t in targets):
            break

    frames = []
    for k, t in enumerate(targets):
        sel = np.flatnonzero(np.isin(code[k, eval_idx], [CODE["val"], CODE["test"]]))
        rows = eval_idx[sel]
        df = pd.DataFrame({"target": t, "pos": task["pos"][rows], "ticker": task["ticker"][rows],
                           "split": np.where(code[k, rows] == CODE["val"], "val", "test")})
        o = best[t]["out"][sel]
        if t == "sector":
            p = torch.softmax(torch.from_numpy(o), dim=1).numpy()
            for c, cls in enumerate(task["info"][t]["classes"]):
                df[f"proba_{cls}"] = p[:, c]
        else:
            df["pred"] = o * task["info"][t]["sd"] + task["info"][t]["mu"]
        frames.append(df)
    log = {"name": args.name, "seed": seed, "ckpt": str(args.ckpt), "ckpt_sha256": sha256_file(args.ckpt),
           "device": str(device), "epochs_run": len(epoch_seconds), "epoch_seconds": epoch_seconds,
           "targets": {t: {"best_epoch": best[t]["epoch"], "best_val_loss": best[t]["val_loss"]} for t in targets}}
    return pd.concat(frames, ignore_index=True), log


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", choices=["fpnet", "fpnet_a"], required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--seeds", default="42,1,2,3,4")
    ap.add_argument("--max-epochs", type=int, default=30)
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = pd.read_parquet(args.labels)
    _, ch, tickers = observation_keys()
    for seed in [int(s) for s in args.seeds.split(",")]:
        lab = labels[labels["seed"] == seed]
        lab = lab if seed == 42 else lab[lab["target"].isin(GROUPED)]  # robustness seeds: grouped targets only
        preds, log = finetune_seed(args, lab, ch, tickers, device, seed)
        preds.to_parquet(out / f"{args.name}_preds_seed{seed}.parquet", index=False)
        (out / f"{args.name}_log_seed{seed}.json").write_text(json.dumps(log, indent=2))
        print(f"[{args.name}] seed {seed}: {len(preds)} prediction rows, best epochs "
              + ", ".join(f"{t} {v['best_epoch']}" for t, v in log["targets"].items()))


if __name__ == "__main__":
    main()
