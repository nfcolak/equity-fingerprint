"""Noise ceilings and persistence baselines for the forward (time) targets.

For vol_1m, beta_1y and liquidity_1m the target is a statistic of the next H trading days. Even a perfect
forecast of the *true* future value cannot beat the sampling noise of that statistic. We estimate this noise
by split-half reliability: the forward window's days are split into odd and even positions, the target
statistic is computed on each half exactly as in `targets.forward_targets`, and the correlation r of the two
halves across observations is Spearman-Brown corrected: reliability = 2r / (1 + r).

Reliability is an UPPER BOUND on the R² a model can reach (it assumes a perfect forecast of the true future
value). Forward-filled zero-return days of illiquid stocks can make the two halves look more alike than
independent noise would, so the bound may be slightly high for those stocks.

The persistence baseline is a one-feature OLS fitted on TRAIN rows only (vol_1m <- log(max(vol_21, floor)),
beta_1y <- beta_252, liquidity_1m <- log_tv_21), scored by R² on val and test; the no-fit identity
(prediction = feature) is reported next to it.

Observations and splits are identical to the benchmark in run.py. Output: results/ceilings.csv.
Run: python -m equity_fingerprint.ceilings [--out-dir results]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import targets as T
from .features import ANN

FEATURE_NAME = {"vol_1m": "log_vol_21", "beta_1y": "beta_252", "liquidity_1m": "log_tv_21"}


# --------------------------------------------------------------------------- pure core
def _stat(kind: str, x: np.ndarray, m: np.ndarray | None) -> np.ndarray:
    """Target statistic per column of x (days x stocks), as in targets.forward_targets."""
    if kind == "vol_1m":
        v = np.std(x, axis=0, ddof=1) * np.sqrt(ANN)
        return np.log(np.maximum(v, T.VOL_FLOOR))
    if kind == "beta_1y":
        xm, mm = x - x.mean(axis=0), m - m.mean()
        return (xm * mm[:, None]).sum(axis=0) / (len(m) - 1) / mm.var(ddof=1)
    if kind == "liquidity_1m":
        return np.log1p(x.mean(axis=0))
    raise ValueError(kind)


def half_statistics(kind: str, window: np.ndarray, mkt: np.ndarray | None = None):
    """window: (H days x stocks) of returns (vol, beta) or traded value (liquidity); mkt: (H,) market returns.

    Stocks with any non-finite value in the window (or market) are NaN, as the target itself needs full windows.
    Returns (odd_half, even_half, full) statistics, each (stocks,)."""
    window = np.asarray(window, float)
    ok = np.isfinite(window).all(axis=0)
    if mkt is not None:
        mkt = np.asarray(mkt, float)
        if not np.isfinite(mkt).all():
            ok[:] = False
    w = np.where(ok, window, 0.0)
    out = []
    for sl in (slice(0, None, 2), slice(1, None, 2), slice(None)):
        s = _stat(kind, w[sl], None if mkt is None else np.nan_to_num(mkt)[sl])
        out.append(np.where(ok, s, np.nan))
    return tuple(out)


def split_half_reliability(a: np.ndarray, b: np.ndarray) -> dict:
    """Split-half correlation and Spearman-Brown reliability; non-finite pairs excluded."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    g = np.isfinite(a) & np.isfinite(b)
    n = int(g.sum())
    if n < 3:
        return {"n_obs": n, "split_half_r": np.nan, "reliability": np.nan}
    r = float(np.corrcoef(a[g], b[g])[0, 1])
    return {"n_obs": n, "split_half_r": r, "reliability": 2 * r / (1 + r)}


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    sst = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - pred) ** 2).sum() / sst) if sst > 0 else np.nan


def persistence_r2(x_train, y_train, evals: dict) -> dict:
    """One-feature OLS fitted on the train rows only. evals: name -> (x, y). Returns coef and R² per eval set
    (OLS and identity prediction = x). Rows with non-finite x or y are excluded."""
    x, y = np.asarray(x_train, float), np.asarray(y_train, float)
    g = np.isfinite(x) & np.isfinite(y)
    slope, intercept = np.polyfit(x[g], y[g], 1)
    res = {"slope": float(slope), "intercept": float(intercept)}
    for name, (xe, ye) in evals.items():
        xe, ye = np.asarray(xe, float), np.asarray(ye, float)
        g = np.isfinite(xe) & np.isfinite(ye)
        res[name] = {"ols_r2": r2_score(ye[g], intercept + slope * xe[g]), "identity_r2": r2_score(ye[g], xe[g])}
    return res


# --------------------------------------------------------------------------- real data
def benchmark_observations(panel, base: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per time target the benchmark dataset (same rows as run.py) with a `split` column."""
    ft = T.forward_targets(panel, base)
    df = pd.concat([base, ft], axis=1)
    out = {}
    for t in T.TIME_TARGETS:
        d = df.copy()
        d["split"] = T.time_split(d["date"], d[f"end_{t}"])
        d = d[d[t].notna() & d["split"].notna()]
        if t == "vol_1m":
            d = d[~d["vol_1m_floored"]]
        out[t] = d
    return out


def _forward_halves(panel, d: pd.DataFrame, kind: str) -> pd.DataFrame:
    H = T.HORIZON[kind]
    ti = {t: j for j, t in enumerate(panel.tickers)}
    src = {"vol_1m": panel.ret, "beta_1y": panel.ret, "liquidity_1m": panel.tv}[kind].to_numpy()
    mkt = panel.mkt.to_numpy() if kind == "beta_1y" else None
    a = np.full(len(d), np.nan)
    b = np.full(len(d), np.nan)
    pos, cols = d["pos"].to_numpy(), d["ticker"].map(ti).to_numpy()
    for p in np.unique(pos):
        idx = np.flatnonzero(pos == p)
        lo = p + 1
        w = src[lo:lo + H][:, cols[idx]]
        ha, hb, _ = half_statistics(kind, w, None if mkt is None else mkt[lo:lo + H])
        a[idx], b[idx] = ha, hb
    return pd.DataFrame({"a": a, "b": b}, index=d.index)


def compute(panel=None) -> pd.DataFrame:
    from . import features as F
    panel = panel or F.build_panel()
    pos = F.month_end_positions(panel)
    base = F.base_features(panel, pos)
    obs = benchmark_observations(panel, base)
    rows = []
    for t in T.TIME_TARGETS:
        d = obs[t]
        halves = _forward_halves(panel, d, t)
        if t == "vol_1m":
            feat = np.log(np.maximum(d["vol_21"].to_numpy(), T.VOL_FLOOR))
        else:
            feat = d["beta_252" if t == "beta_1y" else "log_tv_21"].to_numpy()
        y = d[t].to_numpy()
        sp = d["split"].to_numpy()
        ev = {s: (feat[sp == s], y[sp == s]) for s in ("val", "test")}
        pers = persistence_r2(feat[sp == "train"], y[sp == "train"], ev)
        for s in ("val", "test", "train"):
            m = sp == s
            rel = split_half_reliability(halves["a"].to_numpy()[m], halves["b"].to_numpy()[m])
            rel["n_obs"] = int(m.sum())  # benchmark observation count (pairs used: see n_pairs)
            rows.append({"target": t, "split": s, **rel,
                         "persistence_ols_r2": pers[s]["ols_r2"] if s in pers else np.nan,
                         "persistence_identity_r2": pers[s]["identity_r2"] if s in pers else np.nan,
                         "persistence_feature": FEATURE_NAME[t]})
            rows[-1]["n_pairs"] = split_half_reliability(halves["a"].to_numpy()[m], halves["b"].to_numpy()[m])["n_obs"]
    cols = ["target", "split", "n_obs", "split_half_r", "reliability", "persistence_ols_r2",
            "persistence_identity_r2", "persistence_feature", "n_pairs"]
    return pd.DataFrame(rows)[cols]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = compute()
    df.to_csv(out / "ceilings.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
