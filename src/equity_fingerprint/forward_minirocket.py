"""v8 forward experiment: MiniRocket features of causal 252-day windows for vol_1m and beta_1y.

Protocol (selection on validation only, test scored once, per target):

1. Dataset: built exactly as ``run.main`` does for the time targets (build_panel, month_end_positions,
   base_features, pc_loadings(pc_factors(panel, all tickers)), forward_targets, time_split; no fingerprint
   embeddings). For vol_1m the windows with zero realised vol (``vol_1m_floored``) are dropped as in run.py.
   The rows (hence val/test ``n_obs``) equal those of results/metrics.csv for the same target.
2. Input per observation: the 252 trading days ending at the feature date (inclusive), three channels from
   the daily panel: stock log return (``panel.ret``), market return (``panel.mkt``, SPI) and the stock's log
   traded value ``log1p(close * volume)`` (``panel.tv``). Missing values (before a stock's first price,
   missing market day) are zero-filled; the panel already forward-fills prices so gaps are 0 returns and
   volume 0. Scaling: both return channels are divided by one pooled standard deviation computed on the
   training windows; the log-traded-value channel is z-scored inside each window (it keeps only the shape;
   the level is in the FE features). Nothing after the feature date is used (causal windows).
3. MiniRocket (``minirocket.py``, ~5000 features, seed 42): biases are fitted on the training windows of the
   target only (random sample of them), then val/test are transformed. PPV features.
4. Models (alpha etc. chosen on validation MSE; the reference is the same ridge as ``models.run_target``):
   * ``linear_fe``     : Prep(FE_FEATURES, linear) + ridge (reproduces results/metrics.csv).
   * ``ridge_mr``      : ridge on the standardised MiniRocket features (train mean/sd); alpha from a log grid.
   * ``ridge_mrfe``    : ridge on [MiniRocket, s * FE features (Prep linear)]; alpha from the log grid and the
                         FE up-weight s in {1, 5, 25, 125} (a larger s penalises the FE coefficients less).
   * ``lgbm_femrpca``  : LightGBM (forward_v7 grid, early stopping on validation, n_jobs=1) on FE_FEATURES
                         + the top 20 principal components of the standardised MiniRocket features
                         (PCA from the train Gram matrix).
   * ``avg_mrfe``      : w * ridge_mrfe + (1 - w) * linear_fe with w in {0, .25, .5, .75, 1} on validation R2.
5. The arm (incl. ``linear_fe``) with the highest validation R2 is ``selected = 1``. Test is computed once for
   every arm. Test delta vs ``linear_fe`` of the same target: ticker-block bootstrap of the R2 difference
   (1000 draws, seed 42), 2.5/97.5 percentiles.

Output ``results/forward_minirocket.csv``: target, model, features, split, metric, value, n_obs, ci_lo, ci_hi,
selected. ``ci_lo/ci_hi`` are filled on the test rows with metric ``delta_r2_vs_linear_fe``. Extra val rows
hold hyper-parameters (``alpha``, ``fe_scale``, ``weight_mrfe``, ``n_mr_features``).

Run: env -u PYTHONPATH PYTHONPATH=src python -m equity_fingerprint.forward_minirocket
"""
from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from . import targets as T
from .forward_v7 import KEYS, ROOT, _lgbm, bootstrap_delta_r2, choose_weight, r2
from .minirocket import MiniRocket
from .models import Prep, fit_ridge, regression_metrics

TARGETS = ("vol_1m", "beta_1y")
WINDOW = 252
NUM_FEATURES = 5000
SEED = 42
ALPHAS = np.logspace(0, 6, 13)
FE_SCALES = (1.0, 5.0, 25.0, 125.0)
N_PCA = 20
REF = "linear_fe"
COLS = ["target", "model", "features", "split", "metric", "value", "n_obs", "ci_lo", "ci_hi", "selected"]


# --------------------------------------------------------------------------- data
def build_time_frame(panel) -> pd.DataFrame:
    """Observation frame of run.main (features + PC loadings + forward targets), no fingerprint columns."""
    pos = F.month_end_positions(panel)
    base = F.base_features(panel, pos)
    tickers = sorted(base["ticker"].unique())
    time_df = base.merge(F.pc_loadings(panel, F.pc_factors(panel, tickers), pos), on=KEYS, how="left")
    return pd.concat([time_df, T.forward_targets(panel, time_df)], axis=1)


def target_frame(time_df: pd.DataFrame, target: str) -> pd.DataFrame:
    d = time_df.copy()
    d["split"] = T.time_split(d["date"], d[f"end_{target}"])
    d = d[d[target].notna() & d["split"].notna()]
    if target == "vol_1m":
        d = d[~d["vol_1m_floored"]]
    return d


def windows(panel, d: pd.DataFrame) -> np.ndarray:
    """(n, 3, 252) raw windows ending at each observation's feature date (inclusive): ret, mkt, log1p(tv)."""
    p = d["pos"].to_numpy()
    c = panel.ret.columns.get_indexer(d["ticker"])
    idx = p[:, None] + np.arange(-WINDOW + 1, 1)[None, :]
    if idx.min() < 0:
        raise ValueError("observation without a full 252-day window")
    ret = np.nan_to_num(panel.ret.to_numpy(dtype=np.float64))
    mkt = np.nan_to_num(panel.mkt.to_numpy(dtype=np.float64))
    ltv = np.log1p(np.nan_to_num(panel.tv.to_numpy(dtype=np.float64)))
    X = np.empty((len(d), 3, WINDOW), dtype=np.float64)
    X[:, 0] = ret[idx, c[:, None]]
    X[:, 1] = mkt[idx]
    X[:, 2] = ltv[idx, c[:, None]]
    return X


def scale_windows(X: np.ndarray, ret_sd: float) -> np.ndarray:
    """Pooled train sd for the two return channels; per-window z-score for log traded value."""
    X = X.copy()
    X[:, :2] /= ret_sd
    t = X[:, 2]
    X[:, 2] = (t - t.mean(1, keepdims=True)) / np.where(t.std(1, keepdims=True) > 0, t.std(1, keepdims=True), 1.0)
    return X.astype(np.float32)


# --------------------------------------------------------------------------- ridge by eigendecomposition
def standardise(ftr, *others):
    mu = ftr.mean(0, dtype=np.float64)
    sd = ftr.std(0, dtype=np.float64)
    sd[sd < 1e-8] = 1.0
    f = lambda a: ((a - mu) / sd).astype(np.float32)  # noqa: E731
    return (f(ftr),) + tuple(f(o) for o in others)


def gram(Z: np.ndarray, chunk: int = 4096) -> np.ndarray:
    G = np.zeros((Z.shape[1],) * 2)
    for s in range(0, len(Z), chunk):
        z = Z[s:s + chunk].astype(np.float64)
        G += z.T @ z
    return G


class RidgePath:
    """Ridge with intercept for many alphas from one eigendecomposition of the train Gram matrix."""

    def __init__(self, Ztr, ytr):
        self.ymean = float(np.mean(ytr))
        self.lam, self.V = np.linalg.eigh(gram(Ztr))
        self.lam = np.clip(self.lam, 0, None)
        b = Ztr.T.astype(np.float64) @ (np.asarray(ytr, float) - self.ymean)
        self.c = self.V.T @ b
        self.Ztr = Ztr

    def project(self, Z):
        out = np.empty((len(Z), len(self.lam)))
        for s in range(0, len(Z), 4096):
            out[s:s + 4096] = Z[s:s + 4096].astype(np.float64) @ self.V
        return out

    def predict(self, P, alpha):
        return self.ymean + P @ (self.c / (self.lam + alpha))


def best_alpha(path: RidgePath, Pva, yva):
    scores = [float(np.mean((yva - path.predict(Pva, a)) ** 2)) for a in ALPHAS]
    i = int(np.argmin(scores))
    return float(ALPHAS[i]), scores[i]


# --------------------------------------------------------------------------- one target
def run_target(panel, time_df, target: str, log=print, preds_out=None):
    """One target. ``preds_out`` (dict, optional) receives ``arm -> (val_pred, test_pred, val_index, test_index)``."""
    d = target_frame(time_df, target).reset_index(drop=True)
    tr, va, te = (d[d["split"] == s] for s in ("train", "val", "test"))
    ytr, yva, yte = (x[target].to_numpy() for x in (tr, va, te))
    t0 = time.time()
    Xraw = windows(panel, d)
    itr = (d["split"] == "train").to_numpy()
    ret_sd = float(Xraw[itr, :2].std())
    X = scale_windows(Xraw, ret_sd)
    del Xraw
    Xtr, Xva, Xte = X[itr], X[(d["split"] == "val").to_numpy()], X[(d["split"] == "test").to_numpy()]
    mr = MiniRocket(NUM_FEATURES, SEED).fit(Xtr)  # biases from training windows only
    ftr, fva, fte = (mr.transform(x) for x in (Xtr, Xva, Xte))
    log(f"[{target}] windows {X.shape}, {mr.n_features_} MiniRocket features, transform {time.time() - t0:.0f}s")
    del X, Xtr, Xva, Xte
    Ztr, Zva, Zte = standardise(ftr, fva, fte)
    del ftr, fva, fte

    preds, info = {}, {}  # model -> (val, test)
    # reference
    pl = Prep(F.FE_FEATURES, linear=True).fit(tr)
    Atr, Ava, Ate = (pl.transform(x).to_numpy(np.float64) for x in (tr, va, te))
    m, hp = fit_ridge(Atr, ytr, Ava, yva)
    preds[REF] = (m.predict(Ava), m.predict(Ate))
    info[REF] = hp
    # ridge_mr
    t0 = time.time()
    rp = RidgePath(Ztr, ytr)
    Pva, Pte = rp.project(Zva), rp.project(Zte)
    a, _ = best_alpha(rp, Pva, yva)
    preds["ridge_mr"] = (rp.predict(Pva, a), rp.predict(Pte, a))
    info["ridge_mr"] = {"alpha": a}
    # PCA components of the MR features (top eigenvectors of the train Gram matrix)
    k = slice(-N_PCA, None)
    pc_tr, pc_va, pc_te = (Z.astype(np.float64) @ rp.V[:, k] for Z in (Ztr, Zva, Zte))
    del rp, Pva, Pte
    # ridge_mrfe: FE columns up-weighted by s
    best = None
    for s in FE_SCALES:
        tr_aug = np.hstack([Ztr, (s * Atr).astype(np.float32)])
        va_aug = np.hstack([Zva, (s * Ava).astype(np.float32)])
        te_aug = np.hstack([Zte, (s * Ate).astype(np.float32)])
        path = RidgePath(tr_aug, ytr)
        pv, pt = path.project(va_aug), path.project(te_aug)
        a, mse = best_alpha(path, pv, yva)
        if best is None or mse < best[0]:
            best = (mse, s, a, path.predict(pv, a), path.predict(pt, a))
        del tr_aug, va_aug, te_aug, path, pv, pt
    preds["ridge_mrfe"] = (best[3], best[4])
    info["ridge_mrfe"] = {"alpha": best[2], "fe_scale": best[1]}
    for key, v in (("ridge_mr", info["ridge_mr"]["alpha"]), ("ridge_mrfe", best[2])):
        if v in (ALPHAS[0], ALPHAS[-1]):
            log(f"  WARNING {target}/{key}: alpha {v} at the edge of the grid")
    log(f"  ridge paths {time.time() - t0:.0f}s  mr={info['ridge_mr']} mrfe={info['ridge_mrfe']}")
    # lgbm on FE + MR principal components
    pt_ = Prep(F.FE_FEATURES, linear=False).fit(tr)
    cols = [f"mrpc{i + 1}" for i in range(N_PCA)]
    mats = []
    for part, pc in ((tr, pc_tr), (va, pc_va), (te, pc_te)):
        x = pt_.transform(part).reset_index(drop=True)
        mats.append(pd.concat([x, pd.DataFrame(pc[:, ::-1], columns=cols)], axis=1))
    g = _lgbm(mats[0], ytr, mats[1], yva)
    preds["lgbm_femrpca"] = (g.predict(mats[1]), g.predict(mats[2]))
    info["lgbm_femrpca"] = {"best_iteration": int(g.best_iteration_)}
    # average with the reference
    w = choose_weight(yva, preds["ridge_mrfe"][0], preds[REF][0])  # validation only
    preds["avg_mrfe"] = tuple(w * preds["ridge_mrfe"][i] + (1 - w) * preds[REF][i] for i in (0, 1))
    info["avg_mrfe"] = {"weight_mrfe": w}
    if preds_out is not None:
        for m_, (pv, pt) in preds.items():
            preds_out[m_] = (pv, pt, va.index, te.index)

    feats = {REF: "fe", "ridge_mr": "mr", "ridge_mrfe": "mrfe", "lgbm_femrpca": "femrpca", "avg_mrfe": "mrfe"}
    val_r2 = {m_: r2(yva, p[0]) for m_, p in preds.items()}
    selected = max(val_r2, key=val_r2.get)
    qlike = target == "vol_1m"
    rows = []
    for m_, (pv, pt) in preds.items():
        base = dict(target=target, model=m_, features=feats[m_], ci_lo=np.nan, ci_hi=np.nan,
                    selected=int(m_ == selected))
        rv = float(np.var(yva - pv)) if qlike else 0.0
        for split, y, p in (("val", yva, pv), ("test", yte, pt)):
            for k_, v in regression_metrics(y, p, qlike, rv).items():
                rows.append({**base, "split": split, "metric": k_, "value": float(v),
                             "n_obs": len(y)})
        for k_, v in info[m_].items():
            rows.append({**base, "split": "val", "metric": k_, "value": float(v), "n_obs": len(yva)})
        if m_ in ("ridge_mr", "ridge_mrfe"):
            rows.append({**base, "split": "val", "metric": "n_mr_features", "value": float(mr.n_features_),
                         "n_obs": len(yva)})
        if m_ != REF:
            lo, hi = bootstrap_delta_r2(yte, pt, preds[REF][1], te["ticker"])
            rows.append({**base, "split": "test", "metric": "delta_r2_vs_linear_fe",
                         "value": r2(yte, pt) - r2(yte, preds[REF][1]), "n_obs": len(yte), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)[COLS], {"selected": selected, "info": info, "n_features": mr.n_features_}


def reproduction_check(out: pd.DataFrame, metrics_csv: Path) -> pd.DataFrame:
    """linear_fe rows vs the same target's linear_fe rows in results/metrics.csv (all shared metrics)."""
    ref = pd.read_csv(metrics_csv)
    ref = ref[ref["target"].isin(TARGETS) & (ref["model"] == REF) & ref["metric"].isin(["r2", "mae", "rmse", "qlike"])]
    new = out[(out["model"] == REF) & out["metric"].isin(["r2", "mae", "rmse", "qlike"])]
    k = ["target", "split", "metric"]
    mg = ref.merge(new, on=k, suffixes=("_metrics", "_mr"))
    mg["abs_diff"] = (mg["value_metrics"] - mg["value_mr"]).abs()
    mg["n_equal"] = mg["n_obs_metrics"] == mg["n_obs_mr"]
    return mg[k + ["value_metrics", "value_mr", "abs_diff", "n_obs_metrics", "n_obs_mr", "n_equal"]]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    a = ap.parse_args(argv)
    t0 = time.time()
    panel = F.build_panel()
    time_df = build_time_frame(panel)
    frames, sel = [], {}
    for target in TARGETS:
        o, info = run_target(panel, time_df, target)
        frames.append(o)
        sel[target] = info
    out = pd.concat(frames, ignore_index=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out_dir / "forward_minirocket.csv", index=False)
    print("[reproduction vs results/metrics.csv]\n",
          reproduction_check(out, ROOT / "results" / "metrics.csv").to_string(index=False))
    for target in TARGETS:
        print(f"[{target}] selected on val: {sel[target]['selected']}  hyper-parameters: {sel[target]['info']}")
        o = out[(out["target"] == target) & out["metric"].isin(["r2", "delta_r2_vs_linear_fe"])]
        print(o.pivot_table(index="model", columns=["split", "metric"], values="value").round(4).to_string())
        print(o[o["metric"] == "delta_r2_vs_linear_fe"][["model", "value", "ci_lo", "ci_hi"]].round(4).to_string(index=False))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30  # bytes on macOS -> GiB
    print(f"[runtime] {time.time() - t0:.0f}s, peak RSS {peak:.2f} GiB")


if __name__ == "__main__":
    main()
