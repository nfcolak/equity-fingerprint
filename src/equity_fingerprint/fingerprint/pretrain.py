"""Self-supervised pretraining of the fingerprint encoder on NON-Swiss stocks (dates <= 2021-12-31).

CLI: python -m equity_fingerprint.fingerprint.pretrain --pretrain-dir DIR --out DIR --arm {ssl,ssl_aux}
         --device {cpu,cuda,mps} --steps N --batch 512 --lr 3e-4 --seed 0 [--max-tickers N]
         [--positives {near,far}]

Writes ``ckpt_<arm>.pt`` (every 2,000 steps and at the end) and ``pretrain_log_<arm>.json`` to ``--out``
(``ckpt_<arm>_far.pt`` / ``pretrain_log_<arm>_far.json`` with ``--positives far``).

Losses (the "fingerprint" objective):
  mpm      masked patch modelling: 30% of the patches of view B are masked, scaled channels reconstructed
  nce      InfoNCE (T=0.1) between two augmented views of the same ticker; view B is the masked one.
           near: same end day (views shifted back by 0..5 days); far: end days FAR_MIN..FAR_MAX days apart
  comove   cosine of projected embeddings of two tickers on the same end day -> their realized return correlation
  sector, mcap (arm ssl_aux only): auxiliary heads on the embedding with non-Swiss yfinance labels
Batches are n_days x k tickers (same end day per group), so the co-movement term has pairs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from ..data import PRICE_COLS
from ..features import TRAIN_END
from .model import FingerprintEncoder, mlp
from .windows import CHANNELS, PATCH, WINDOW, channels_from_wide, window_batch

N_DATA = len(CHANNELS)
MIN_ROWS = 500          # rows (<= cutoff) a ticker needs
MIN_WINDOW_DAYS = 200   # valid days a (ticker, end day) window needs
MAX_SHIFT = 5           # view augmentation: end day moved back by 0..5 trading days
FAR_MIN, FAR_MAX = 252, 756  # --positives far: trading days between the end days of the two views
MIN_CORR_DAYS = 100     # common valid days for a realized correlation
MASK_FRAC, TEMP, P_DROP, JITTER = 0.3, 0.1, 0.1, (0.9, 1.1)
JITTER_CH = [CHANNELS.index(c) for c in ("ret", "gap", "mkt_ret")]
WEIGHTS = {"mpm": 1.0, "nce": 1.0, "comove": 5.0, "sector": 0.5, "mcap": 0.5}
VAL_FRAC, VAL_BATCHES = 0.05, 4
LOG_EVERY, SAVE_EVERY, WARMUP, WEIGHT_DECAY = 100, 2000, 500, 0.05
UNIT = {"GBp": 0.01, "GBX": 0.01, "ILA": 0.01}  # sub-unit quotes -> main currency (no FX otherwise)


# --------------------------------------------------------------------------- data
class Data:
    """Channels and labels of the pretraining universe (all dates <= cutoff, no .SW ticker)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _pretrain_dir(path=None) -> Path:
    path = path or os.environ.get("EF_PRETRAIN_DIR")
    return Path(path) if path else Path(__file__).resolve().parents[3] / "data" / "pretrain"


def load_pretrain(pdir, max_tickers=None, seed=0) -> Data:
    pdir = Path(pdir)
    cutoff = pd.Timestamp(TRAIN_END)
    uni = pd.read_csv(pdir / "universe.csv").drop_duplicates("ticker").set_index("ticker")
    prices = pd.read_parquet(pdir / "raw" / "prices.parquet")[PRICE_COLS]
    index = pd.read_parquet(pdir / "raw" / "index.parquet")[PRICE_COLS]
    info = pd.read_parquet(pdir / "raw" / "info.parquet").drop_duplicates("ticker").set_index("ticker")
    assert not any(str(t).upper().endswith(".SW") for t in set(uni.index) | set(prices["ticker"])), \
        "a .SW ticker is in the pretraining data"

    for df in (prices, index):
        df["date"] = pd.to_datetime(df["date"])
    prices = prices[prices["date"] <= cutoff].drop_duplicates(["date", "ticker"])
    index = index[index["date"] <= cutoff].drop_duplicates(["date", "ticker"])
    assert prices["date"].max() <= cutoff and index["date"].max() <= cutoff

    rows = prices.groupby("ticker").size()
    has_idx = uni["market_index"].isin(set(index["ticker"]))
    keep = sorted(t for t in rows.index[rows >= MIN_ROWS] if t in uni.index and has_idx[t])
    if max_tickers and len(keep) > max_tickers:
        keep = sorted(np.random.default_rng(seed).choice(keep, max_tickers, replace=False))
    prices = prices[prices["ticker"].isin(keep)]

    def wide(col):
        return prices.pivot(index="date", columns="ticker", values=col).sort_index()[keep]

    adj, close = wide("adj_close"), wide("close")
    cal = adj.index
    idx = index.pivot(index="date", columns="ticker", values="close").sort_index()
    idx = idx.reindex(cal.union(idx.index)).ffill().reindex(cal)
    mkt = pd.DataFrame(idx[uni.loc[keep, "market_index"].to_numpy()].to_numpy(), index=cal, columns=keep)
    ch = channels_from_wide(adj, close, wide("open"), wide("high"), wide("low"), wide("volume"), mkt)

    unit = np.array([UNIT.get(info["currency"].get(t), 1.0) for t in keep])
    shares = info["shares_outstanding"].reindex(keep).to_numpy(float) * unit
    shares[~(shares > 0)] = np.nan
    manifest = pdir / "manifest.json"
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else ""
    return Data(tickers=keep, calendar=cal, ch=ch, close=close.ffill(limit=5).to_numpy(np.float32),
                log_shares=np.log(shares), sector=info["sector"].reindex(keep).to_numpy(object),
                max_date=str(cal.max().date()), manifest_sha256=sha)


def fit_scaler(ch, cols, seed=0, n=2_000_000) -> dict:
    """Per-channel median / IQR over a random sample of valid cells of the training tickers."""
    rng = np.random.default_rng(seed)
    cells = ch[rng.integers(0, ch.shape[0], n), rng.choice(cols, n)]
    med, iqr = [], []
    for c in range(cells.shape[1]):
        v = cells[:, c][np.isfinite(cells[:, c])]
        q1, m, q3 = np.percentile(v, [25, 50, 75])
        spread = q3 - q1
        if spread < 1e-8:  # e.g. zero_vol is almost always 0
            spread = v.std() if v.std() > 1e-8 else 1.0
        med.append(float(m))
        iqr.append(float(spread))
    return {"median": med, "iqr": iqr}


def window_ok(ch) -> np.ndarray:
    """[T, N] True where the window ending that day has >= MIN_WINDOW_DAYS valid days (and room to shift)."""
    avail = np.isfinite(ch[..., :2]).any(-1).cumsum(0, dtype=np.int32)
    cnt = avail.copy()
    cnt[WINDOW:] -= avail[:-WINDOW]
    ok = cnt >= MIN_WINDOW_DAYS
    ok[:WINDOW - 1 + MAX_SHIFT] = False
    return ok


def realized_corr(ret, d, k):
    """Pairwise-complete correlation of raw returns [d*k, W] within each group of k tickers."""
    r = ret.reshape(d, k, -1).astype(np.float64)
    v = np.isfinite(r).astype(np.float64)
    x = np.nan_to_num(r)
    e = lambda a, b: np.einsum("dit,djt->dij", a, b)  # noqa: E731
    n, sx, sy = e(v, v), e(x, v), e(v, x)
    cov = e(x, x) - sx * sy / np.maximum(n, 1)
    vx = e(x * x, v) - sx ** 2 / np.maximum(n, 1)
    vy = e(v, x * x) - sy ** 2 / np.maximum(n, 1)
    good = (n >= MIN_CORR_DAYS) & (vx > 1e-12) & (vy > 1e-12)
    corr = np.where(good, cov / np.sqrt(np.maximum(vx * vy, 1e-24)), 0.0)
    return corr.astype(np.float32), good & ~np.eye(k, dtype=bool)[None]


class Sampler:
    """Batches of ``n_days`` end days x ``k`` tickers drawn from the tickers ``pool``."""

    def __init__(self, data, pool, ok, batch, k=8, positives="near"):
        self.d, self.pool, self.ok, self.positives = data, np.asarray(pool), ok[:, pool], positives
        self.k = min(k, len(self.pool))
        self.days = np.flatnonzero(self.ok.sum(1) >= self.k)
        if positives == "far":
            # sorted valid days of every ticker as one key array (column * span + day + pad) for vectorised searchsorted
            self.pad, self.span = FAR_MAX + 1, ok.shape[0] + 2 * FAR_MAX + 2
            col, day = np.nonzero(self.ok.T)
            self.vday, self.vkey = day, col * self.span + day + self.pad
            # [T, P] once: valid on that day and has a far partner (per-batch searchsorted over all tickers is ~90 ms)
            self.elig = np.zeros_like(self.ok)
            for lo in range(0, ok.shape[0], 256):
                rows = np.arange(lo, min(lo + 256, ok.shape[0]))
                _, n1, _, n2 = self._spans(rows[:, None], np.arange(len(self.pool))[None])
                self.elig[rows] = self.ok[rows] & (n1 + n2 > 0)
            self.days = self.days[self._eligible(self.days, np.zeros(len(self.days), int)).sum(1) >= self.k]
        assert len(self.days), "no end day with enough tickers"
        self.n_days = max(1, batch // self.k)

    def _spans(self, a, c):
        """Positions in the key array of the valid days of ticker ``c`` in [a-FAR_MAX, a-FAR_MIN] and
        [a+FAR_MIN, a+FAR_MAX] -> (lo_before, n_before, lo_after, n_after)."""
        key = lambda x: np.searchsorted(self.vkey, c * self.span + x + self.pad, side="left")  # noqa: E731
        lo1, hi1 = key(a - FAR_MAX), key(a - FAR_MIN + 1)
        lo2, hi2 = key(a + FAR_MIN), key(a + FAR_MAX + 1)
        return lo1, hi1 - lo1, lo2, hi2 - lo2

    def _eligible(self, t, shift):
        """[len(t), P] bool: ticker valid on day ``t - shift`` and has a valid day FAR_MIN..FAR_MAX away from it."""
        return self.elig[t - shift]

    def _draw_far(self, rng):
        t = rng.choice(self.days, self.n_days, replace=len(self.days) < self.n_days)
        a = t - rng.integers(0, MAX_SHIFT + 1, self.n_days)
        elig = self._eligible(t, t - a)
        short = elig.sum(1) < self.k  # shifted day leaves too few tickers: use the unshifted day (always enough)
        a[short] = t[short]
        elig[short] = self._eligible(t[short], np.zeros(int(short.sum()), int))
        score = np.where(elig, rng.random(elig.shape), -1.0)  # k random eligible tickers per day group
        local = np.argpartition(-score, self.k - 1, axis=1)[:, :self.k].reshape(-1)
        a_r = np.repeat(a, self.k)
        lo1, n1, lo2, n2 = self._spans(a_r, local)
        r = np.minimum((rng.random(len(local)) * (n1 + n2)).astype(np.int64), n1 + n2 - 1)  # uniform over both sides
        b = self.vday[np.where(r < n1, lo1 + r, lo2 + r - n1)]
        return a_r, b, self.pool[local]

    def draw(self, rng):
        if self.positives == "far":
            return self._draw_far(rng)
        t = rng.choice(self.days, self.n_days, replace=len(self.days) < self.n_days)
        local = np.stack([rng.choice(np.flatnonzero(self.ok[d]), self.k, replace=False) for d in t])
        a = t - rng.integers(0, MAX_SHIFT + 1, self.n_days)
        b = t - rng.integers(0, MAX_SHIFT + 1, self.n_days)
        return np.repeat(a, self.k), np.repeat(b, self.k), self.pool[local].reshape(-1)

    def batch(self, rng, scaler, labels):
        """-> dict of numpy arrays (windows of both views, correlation target, labels)."""
        ta, tb, j = self.draw(rng)
        d = self.d
        raw = d.ch[(ta[:, None] + np.arange(-WINDOW + 1, 1)), j[:, None], 0]
        corr, good = realized_corr(raw, self.n_days, self.k)
        mc = d.log_shares[j] + np.log(d.close[ta, j])
        return dict(xa=window_batch(d.ch, ta, j, scaler), xb=window_batch(d.ch, tb, j, scaler), corr=corr,
                    good=good, tid=j, sector=labels["sector"][j],
                    mcap=((mc - labels["mc_mean"]) / labels["mc_std"]).astype(np.float32))


def make_labels(data, train, min_sector, seed=0):
    """Sector class ids (classes with >= min_sector training tickers, else -100) and log mcap moments."""
    s = pd.Series(data.sector[train]).dropna()
    counts = s.value_counts()
    classes = sorted(counts.index[counts >= min_sector])
    sector = np.array([classes.index(x) if x in classes else -100 for x in data.sector], dtype=np.int64)
    rng = np.random.default_rng(seed)
    t, j = rng.integers(WINDOW, data.ch.shape[0], 50_000), rng.choice(train, 50_000)
    mc = data.log_shares[j] + np.log(data.close[t, j])
    mc = mc[np.isfinite(mc)]
    return dict(sector=sector, classes=classes, mc_mean=float(mc.mean()) if len(mc) else 0.0,
                mc_std=float(mc.std()) if len(mc) > 1 and mc.std() > 0 else 1.0)


# --------------------------------------------------------------------------- model + losses
class PretrainModel(nn.Module):
    def __init__(self, encoder, arm, n_classes):
        super().__init__()
        d = encoder.config["d_model"]
        self.encoder = encoder
        self.recon = nn.Linear(d, PATCH * N_DATA)
        self.proj = mlp(d, d, 64)     # contrastive head
        self.comove = mlp(d, d, 64)   # separate head: InfoNCE pushes tickers apart, comove wants graded cosines
        self.aux = arm == "ssl_aux"
        self.sector = nn.Linear(d, n_classes) if self.aux and n_classes else None
        self.mcap = nn.Linear(d, 1) if self.aux else None


def augment(x, g):
    """Channel dropout (p=0.1) and magnitude jitter (0.9-1.1) on the return channels; ``g`` is a CPU generator."""
    b, dev = x.shape[0], x.device
    x = x.clone()
    drop = torch.rand(b, 1, N_DATA, generator=g) < P_DROP
    x[..., :N_DATA] = x[..., :N_DATA] * (~drop).to(dev, x.dtype)
    jit = JITTER[0] + (JITTER[1] - JITTER[0]) * torch.rand(b, 1, len(JITTER_CH), generator=g)
    x[..., JITTER_CH] = x[..., JITTER_CH] * jit.to(dev, x.dtype)
    return x


def losses(model, bt, g, n_days, k):
    xa, xb = bt["xa"], bt["xb"]
    n_p = model.encoder.n_patches
    n_masked = max(1, round(MASK_FRAC * n_p))
    idx = torch.rand(xb.shape[0], n_p, generator=g).argsort(1)[:, :n_masked]
    pmask = torch.zeros(xb.shape[0], n_p, dtype=torch.bool).scatter_(1, idx, True).to(xb.device)

    emb_a, _ = model.encoder(augment(xa, g))
    emb_b, tok_b = model.encoder(augment(xb, g), pmask)

    pred = model.recon(tok_b).float().reshape(xb.shape[0], WINDOW, N_DATA)
    w = (pmask.repeat_interleave(PATCH, 1) & (xb[..., N_DATA] > 0)).float()
    out = {"mpm": (((pred - xb[..., :N_DATA]) ** 2).mean(-1) * w).sum() / w.sum().clamp(min=1)}

    # Autocast would downcast the matmuls below to float16 (-1e9 fill overflows, exp/CE lose range): keep them float32.
    za, zb = F.normalize(model.proj(emb_a).float(), dim=-1), F.normalize(model.proj(emb_b).float(), dim=-1)
    zc = F.normalize(model.comove(emb_a).float(), dim=-1)
    with torch.autocast(xb.device.type, enabled=False):
        logits = (za @ zb.T / TEMP).float()
        same = bt["tid"][:, None] == bt["tid"][None, :]
        logits = logits.masked_fill(same & ~torch.eye(len(same), dtype=torch.bool, device=same.device), -1e9)
        tgt = torch.arange(len(logits), device=logits.device)
        out["nce"] = (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt)) / 2

        c = zc.reshape(n_days, k, -1)
        cos = c @ c.transpose(1, 2)
        out["comove"] = ((cos - bt["corr"].float()) ** 2)[bt["good"]].mean() if bt["good"].any() else cos.sum() * 0

    if model.aux:
        sec = bt["sector"]
        out["sector"] = (F.cross_entropy(model.sector(emb_a).float(), sec, ignore_index=-100)
                         if model.sector is not None and (sec >= 0).any() else emb_a.sum() * 0)
        m = torch.isfinite(bt["mcap"])
        err = (model.mcap(emb_a).float().squeeze(-1) - torch.nan_to_num(bt["mcap"])) ** 2
        out["mcap"] = (err * m).sum() / m.sum().clamp(min=1)
    out["total"] = sum(WEIGHTS[n] * v for n, v in out.items())
    return out


def to_device(bt, device):
    kinds = {"good": torch.bool, "tid": torch.long, "sector": torch.long}
    return {n: torch.as_tensor(v, dtype=kinds.get(n, torch.float32)).to(device) for n, v in bt.items()}


@torch.no_grad()
def evaluate(model, val_batches, sampler, seed):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    tot = {}
    for bt in val_batches:
        for n, v in losses(model, bt, g, sampler.n_days, sampler.k).items():
            tot[n] = tot.get(n, 0.0) + float(v) / len(val_batches)
    model.train()
    return tot


# --------------------------------------------------------------------------- training
def save_ckpt(path, model, args, scaler, data, step, final):
    ckpt = {"state_dict": {n: v.detach().cpu() for n, v in model.encoder.state_dict().items()},
            "config": dict(model.encoder.config), "scaler": scaler,
            "heads_state_dict": {n: v.detach().cpu() for n, v in model.state_dict().items()
                                 if not n.startswith("encoder.")},
            "heads_config": {"d_model": model.encoder.config["d_model"], "comove_out": 64,
                             "n_sector_classes": model.sector.out_features if model.sector is not None else 0},
            "pretrain": {"arm": args.arm, "positives": args.positives, "n_tickers": len(data.tickers), "max_date": data.max_date,
                         "steps": step, "seed": args.seed, "manifest_sha256": data.manifest_sha256,
                         "final_losses": final}}
    tmp = Path(str(path) + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def lr_at(step, steps, lr):
    warm = min(WARMUP, max(1, steps // 10))
    if step <= warm:
        return lr * step / warm
    return lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm))))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pretrain-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=["ssl", "ssl_aux"], required=True)
    ap.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--positives", choices=["near", "far"], default="near",
                    help="contrastive views: same end day (near) or same ticker 252-756 days apart (far)")
    ap.add_argument("--max-tickers", type=int, default=None)
    ap.add_argument("--min-sector-tickers", type=int, default=20, help="arm ssl_aux: smallest sector class")
    args = ap.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    data = load_pretrain(_pretrain_dir(args.pretrain_dir), args.max_tickers, args.seed)
    n = len(data.tickers)
    perm = np.random.default_rng(args.seed).permutation(n)
    n_val = max(1, round(VAL_FRAC * n))
    val, train = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    scaler = fit_scaler(data.ch, train, args.seed)
    ok = window_ok(data.ch)
    labels = make_labels(data, train, args.min_sector_tickers, args.seed)
    tr_s = Sampler(data, train, ok, args.batch, positives=args.positives)
    va_s = Sampler(data, val, ok, args.batch, positives=args.positives)
    print(f"data: {n} tickers ({len(train)} train / {len(val)} val), {data.ch.shape[0]} days, "
          f"to {data.max_date}, {int(ok[:, train].sum())} train windows, "
          f"{len(labels['classes'])} sector classes, loaded in {time.time() - t0:.0f}s", flush=True)

    model = PretrainModel(FingerprintEncoder(), args.arm, len(labels["classes"])).to(device)
    n_params = sum(p.numel() for p in model.encoder.parameters())
    decay = [p for p in model.parameters() if p.ndim >= 2]
    other = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY},
                             {"params": other, "weight_decay": 0.0}], lr=args.lr)
    amp = device.type == "cuda"
    gscaler = torch.amp.GradScaler("cuda", enabled=amp)

    rng_val = np.random.default_rng(args.seed + 1)
    val_batches = [to_device(va_s.batch(rng_val, scaler, labels), device) for _ in range(VAL_BATCHES)]
    rng, g = np.random.default_rng(args.seed + 2), torch.Generator().manual_seed(args.seed + 3)

    rng_gap = np.random.default_rng(args.seed + 4)  # own stream: logging must not move the training draws
    gaps = [float(np.abs(ta - tb).mean()) for ta, tb, _ in (tr_s.draw(rng_gap) for _ in range(20))]
    print(f"positives {args.positives}: mean |ta - tb| {np.mean(gaps):.1f} days", flush=True)

    log = {"args": vars(args), "positives": args.positives, "mean_abs_dt": float(np.mean(gaps)), "n_encoder_params": n_params, "n_train_tickers": int(len(train)),
           "n_val_tickers": int(len(val)), "weights": WEIGHTS, "records": []}
    tag = args.arm + ("_far" if args.positives == "far" else "")
    log_path, ckpt_path = out / f"pretrain_log_{tag}.json", out / f"ckpt_{tag}.pt"
    log["records"].append({"step": 0, "positives": args.positives, "val": evaluate(model, val_batches, va_s, args.seed)})
    print(f"step 0 val {log['records'][0]['val']}", flush=True)

    acc, t_win, last = {}, time.time(), {}
    for step in range(1, args.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step, args.steps, args.lr)
        bt = to_device(tr_s.batch(rng, scaler, labels), device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            ls = losses(model, bt, g, tr_s.n_days, tr_s.k)
        opt.zero_grad(set_to_none=True)
        gscaler.scale(ls["total"]).backward()
        gscaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        gscaler.step(opt)
        gscaler.update()
        for name, v in ls.items():
            acc[name] = acc.get(name, 0.0) + v.detach()

        if step % LOG_EVERY == 0 or step == args.steps:
            cnt = LOG_EVERY if step % LOG_EVERY == 0 else step % LOG_EVERY
            train_l = {nm: float(v) / cnt for nm, v in acc.items()}  # float() syncs the device
            rec = {"step": step, "positives": args.positives, "train": train_l, "val": evaluate(model, val_batches, va_s, args.seed),
                   "steps_per_s": cnt / (time.time() - t_win), "lr": opt.param_groups[0]["lr"]}
            log["records"].append(rec)
            log_path.write_text(json.dumps(log, indent=1))
            last = {**{f"train_{k_}": v for k_, v in train_l.items()}, **{f"val_{k_}": v for k_, v in rec["val"].items()}}
            print(f"step {step} train {train_l['total']:.4f} val {rec['val']['total']:.4f} "
                  f"({rec['steps_per_s']:.2f} steps/s)", flush=True)
            acc, t_win = {}, time.time()
        if step % SAVE_EVERY == 0 and step < args.steps:
            save_ckpt(ckpt_path, model, args, scaler, data, step, last)

    save_ckpt(ckpt_path, model, args, scaler, data, args.steps, last)
    log_path.write_text(json.dumps(log, indent=1))
    print(f"saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
