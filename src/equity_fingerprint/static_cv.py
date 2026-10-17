"""v6 static cross-validation: sector and log_mcap, one row per Swiss stock.

Both targets are static per stock (a label snapshot / contemporaneous market cap), so a time split adds
nothing. Every eligible stock is one sample; evaluation is repeated stratified 5-fold CV (10 repeats,
fold seed = repeat index 0..9) and metrics are computed on the pooled out-of-fold predictions of each
repeat, then reported as mean and sd over repeats. Hyper-parameters are chosen by inner 5-fold CV on the
outer training fold only; labels of outer test stocks are never used for fitting, selection, references
or scaling (the fold-level functions only read y[train]).

sector
  eligible: stocks with a label in targets.sector_map (classes < 5 tickers dropped) and >= 252 valid
  daily returns. Correlations use the full available daily history (sector is a static snapshot), pairwise
  complete, NaN below 252 common days.
  models: baseline (train majority), knn (k in {3,5,7,10} by inner CV, neighbours weighted by
  clipped-positive correlation, majority fallback), prof_logit (per class mean of the top-3 correlations
  to train stocks of that class + top-10 weighted vote shares; self excluded, leave-one-out for training
  stocks), femean_logit (per-stock mean of FE_BASE_FEATURES over all month-end observations through
  Prep(linear=True) fitted on the train fold), prof_femean_logit (both). C in LOGIT_CS by inner CV.
  CI: 2.5/97.5 percentiles of 1000 stock-bootstrap resamples of repeat 0 (macro-F1).
log_mcap
  target = log(close x shares-outstanding snapshot) at the last month-end; features = FE_BASE_FEATURES
  at that date (no PC loadings). KFold(5) x 10 repeats. models: mean, ridge (alpha by inner CV), lgbm
  (300 trees, lr 0.05, num_leaves {4,8} x min_child_samples {10,20} by inner CV). Metrics r2, mae.

Outputs (results/ by default): static_cv.csv, static_cv_oof.parquet (y_true / y_pred stored as strings so
sector class names and log_mcap values share one column).
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error,
                             r2_score)
from sklearn.model_selection import KFold, StratifiedKFold

from .features import FE_BASE_FEATURES, base_features, build_panel, month_end_positions
from .models import LOGIT_CS, RIDGE_ALPHAS, Prep
from .targets import log_mcap_target, sector_map

MIN_DAYS = 252
KNN_KS = [3, 5, 7, 10]
PROF_M, PROF_K = 3, 10
LGBM_GRID = [dict(num_leaves=nl, min_child_samples=mcs) for nl in (4, 8) for mcs in (10, 20)]
INNER_SEED = 0
N_BOOT = 1000
BOOT_SEED = 12345
SECTOR_MODELS = ["baseline", "knn", "prof_logit", "femean_logit", "prof_femean_logit"]
MCAP_MODELS = ["baseline", "ridge", "lgbm"]


# --------------------------------------------------------------------------- correlations
def pairwise_corr(ret: pd.DataFrame, min_common: int = MIN_DAYS) -> pd.DataFrame:
    """Pairwise-complete Pearson correlation of a (date x ticker) returns frame; NaN below min_common days."""
    X = ret.to_numpy(float)
    V = np.isfinite(X).astype(float)
    X0 = np.where(V > 0, X, 0.0)
    n = V.T @ V
    sx = X0.T @ V                      # sx[i, j] = sum of x_i over days where both valid
    sxx = (X0 ** 2).T @ V
    sxy = X0.T @ X0
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = sxy - sx * sx.T / n
        vi = sxx - sx ** 2 / n
        c = cov / np.sqrt(vi * vi.T)
    c = np.where(n >= min_common, c, np.nan)
    c = np.clip(c, -1.0, 1.0).copy()
    np.fill_diagonal(c, np.nan)
    return pd.DataFrame(c, index=ret.columns, columns=ret.columns)


# --------------------------------------------------------------------------- sector building blocks
def _masked_sims(C: np.ndarray, ref: np.ndarray, query: np.ndarray) -> np.ndarray:
    """(len(query), len(ref)) correlations; a query's own entry in ref is set to NaN (self excluded)."""
    S = C[np.ix_(query, ref)].copy()
    S[query[:, None] == ref[None, :]] = np.nan
    return S


def knn_predict(S: np.ndarray, y_ref: np.ndarray, k: int, n_classes: int, majority: int) -> np.ndarray:
    """Distance-weighted kNN on a similarity block; weights = clipped-positive correlation."""
    out = np.full(S.shape[0], majority, dtype=int)
    Sf = np.where(np.isfinite(S), S, -np.inf)
    for i in range(S.shape[0]):
        idx = np.argsort(-Sf[i], kind="stable")[:k]
        w = np.clip(Sf[i, idx], 0.0, None)
        w[~np.isfinite(w)] = 0.0
        if w.sum() <= 0:
            continue
        votes = np.bincount(y_ref[idx], weights=w, minlength=n_classes)
        out[i] = int(np.argmax(votes))
    return out


def profile_from_sims(S: np.ndarray, y_ref: np.ndarray, n_classes: int,
                      m: int = PROF_M, K: int = PROF_K) -> np.ndarray:
    """Per class mean of top-m correlations, then top-K weighted vote shares. Missing -> 0."""
    n = S.shape[0]
    top = np.zeros((n, n_classes))
    for c in range(n_classes):
        Sc = S[:, y_ref == c]
        if Sc.shape[1] == 0:
            continue
        Sc = np.sort(np.where(np.isfinite(Sc), Sc, -np.inf), axis=1)[:, ::-1][:, :m]
        valid = np.isfinite(Sc)
        cnt = valid.sum(1)
        tot = np.where(valid, Sc, 0.0).sum(1)
        top[:, c] = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
    share = np.zeros((n, n_classes))
    Sf = np.where(np.isfinite(S), S, -np.inf)
    for i in range(n):
        idx = np.argsort(-Sf[i], kind="stable")[:K]
        w = np.clip(Sf[i, idx], 0.0, None)
        w[~np.isfinite(w)] = 0.0
        if w.sum() > 0:
            share[i] = np.bincount(y_ref[idx], weights=w, minlength=n_classes) / w.sum()
    return np.hstack([top, share])


def profile_features(C: np.ndarray, y: np.ndarray, ref: np.ndarray, query: np.ndarray,
                     n_classes: int) -> np.ndarray:
    """Profile of `query` stocks against `ref` stocks (labels y[ref]); own correlation excluded."""
    return profile_from_sims(_masked_sims(C, ref, query), y[ref], n_classes)


def _fit_logit(X: np.ndarray, y: np.ndarray, C: float) -> LogisticRegression:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LogisticRegression(C=C, max_iter=5000, class_weight=None).fit(X, y)


def _build_blocks(C, y, fe, fit_idx, ev_idx, n_classes, with_fe=True):
    """Feature blocks for logit models: fit rows (LOO profile) and eval rows (profile vs fit rows)."""
    P_fit = profile_features(C, y, fit_idx, fit_idx, n_classes)
    P_ev = profile_features(C, y, fit_idx, ev_idx, n_classes)
    prep = Prep(list(fe.columns), linear=True).fit(fe.iloc[fit_idx])
    F_fit = prep.transform(fe.iloc[fit_idx]).to_numpy()
    F_ev = prep.transform(fe.iloc[ev_idx]).to_numpy()
    return {"prof_logit": (P_fit, P_ev), "femean_logit": (F_fit, F_ev),
            "prof_femean_logit": (np.hstack([P_fit, F_fit]), np.hstack([P_ev, F_ev]))}


def _macro(y_true, y_pred, n_classes):
    return f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0)


def _inner_splits(y_tr: np.ndarray, seed: int = INNER_SEED):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y_tr)), y_tr))


def sector_fold(C: np.ndarray, y: np.ndarray, fe: pd.DataFrame, train: np.ndarray, test: np.ndarray,
                n_classes: int) -> dict[str, np.ndarray]:
    """Predict class indices of `test` stocks for every sector model. Reads y[train] only.

    C: (n, n) correlation matrix with NaN diagonal; y: class indices (entries outside train ignored);
    fe: per-stock feature means (rows aligned with C).
    """
    train, test = np.asarray(train), np.asarray(test)
    y_tr = y[train]
    majority = int(np.bincount(y_tr, minlength=n_classes).argmax())

    # ---- inner CV: k for kNN, C for the three logits
    knn_score = {k: [] for k in KNN_KS}
    logit_score = {(m, c): [] for m in ("prof_logit", "femean_logit", "prof_femean_logit") for c in LOGIT_CS}
    for itr, iva in _inner_splits(y_tr):
        a, b = train[itr], train[iva]
        maj = int(np.bincount(y[a], minlength=n_classes).argmax())
        S = _masked_sims(C, a, b)
        for k in KNN_KS:
            knn_score[k].append(_macro(y[b], knn_predict(S, y[a], k, n_classes, maj), n_classes))
        blocks = _build_blocks(C, y, fe, a, b, n_classes)
        for m, (Xa, Xb) in blocks.items():
            for c in LOGIT_CS:
                logit_score[(m, c)].append(_macro(y[b], _fit_logit(Xa, y[a], c).predict(Xb), n_classes))
    best_k = max(KNN_KS, key=lambda k: (np.mean(knn_score[k]), -k))
    best_c = {m: max(LOGIT_CS, key=lambda c: (np.mean(logit_score[(m, c)]), -c))
              for m in ("prof_logit", "femean_logit", "prof_femean_logit")}

    # ---- refit on the whole outer train fold, predict test
    out = {"baseline": np.full(len(test), majority, dtype=int),
           "knn": knn_predict(_masked_sims(C, train, test), y_tr, best_k, n_classes, majority)}
    for m, (Xa, Xb) in _build_blocks(C, y, fe, train, test, n_classes).items():
        out[m] = _fit_logit(Xa, y_tr, best_c[m]).predict(Xb)
    return out


def class_metrics(y_true, y_pred, n_classes) -> dict[str, float]:
    lab = list(range(n_classes))
    return {"macro_f1": f1_score(y_true, y_pred, labels=lab, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, labels=lab, average="weighted", zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)}


def run_sector_repeat(C, y, fe, n_classes, repeat) -> dict[str, np.ndarray]:
    """One outer repeat: pooled out-of-fold class predictions (every stock exactly once)."""
    oof = {m: np.full(len(y), -1, dtype=int) for m in SECTOR_MODELS}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        folds = list(StratifiedKFold(5, shuffle=True, random_state=repeat).split(np.zeros(len(y)), y))
    for tr, te in folds:
        pred = sector_fold(C, y, fe, tr, te, n_classes)
        for m in SECTOR_MODELS:
            oof[m][te] = pred[m]
    return oof


# --------------------------------------------------------------------------- log_mcap
def mcap_fold(X: pd.DataFrame, y: np.ndarray, train: np.ndarray, test: np.ndarray) -> dict[str, np.ndarray]:
    """Predict log_mcap of `test` stocks for every model. Reads y[train] only."""
    train, test = np.asarray(train), np.asarray(test)
    y_tr = y[train]
    cols = list(X.columns)
    inner = list(KFold(5, shuffle=True, random_state=INNER_SEED).split(train))
    ridge_s = {a: [] for a in RIDGE_ALPHAS}
    lgbm_s = {i: [] for i in range(len(LGBM_GRID))}
    for itr, iva in inner:
        a, b = train[itr], train[iva]
        lin = Prep(cols, linear=True).fit(X.iloc[a])
        Xa, Xb = lin.transform(X.iloc[a]).to_numpy(), lin.transform(X.iloc[b]).to_numpy()
        raw = Prep(cols, linear=False).fit(X.iloc[a])
        Ra, Rb = raw.transform(X.iloc[a]), raw.transform(X.iloc[b])
        for al in RIDGE_ALPHAS:
            ridge_s[al].append(np.mean((y[b] - Ridge(alpha=al).fit(Xa, y[a]).predict(Xb)) ** 2))
        for i, g in enumerate(LGBM_GRID):
            mdl = _lgbm(g).fit(Ra, y[a])
            lgbm_s[i].append(np.mean((y[b] - mdl.predict(Rb)) ** 2))
    best_a = min(RIDGE_ALPHAS, key=lambda a: (np.mean(ridge_s[a]), a))
    best_g = LGBM_GRID[min(lgbm_s, key=lambda i: (np.mean(lgbm_s[i]), i))]
    lin = Prep(cols, linear=True).fit(X.iloc[train])
    raw = Prep(cols, linear=False).fit(X.iloc[train])
    return {"baseline": np.full(len(test), float(y_tr.mean())),
            "ridge": Ridge(alpha=best_a).fit(lin.transform(X.iloc[train]).to_numpy(), y_tr)
            .predict(lin.transform(X.iloc[test]).to_numpy()),
            "lgbm": _lgbm(best_g).fit(raw.transform(X.iloc[train]), y_tr).predict(raw.transform(X.iloc[test]))}


def _lgbm(g: dict):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=300, learning_rate=0.05, verbose=-1, random_state=42, n_jobs=1, **g)  # 1 thread: ~200 rows; avoids a libomp deadlock when torch is loaded in the same process (unittest discover)


def run_mcap_repeat(X: pd.DataFrame, y: np.ndarray, repeat: int) -> dict[str, np.ndarray]:
    oof = {m: np.full(len(y), np.nan) for m in MCAP_MODELS}
    for tr, te in KFold(5, shuffle=True, random_state=repeat).split(np.zeros(len(y))):
        pred = mcap_fold(X, y, tr, te)
        for m in MCAP_MODELS:
            oof[m][te] = pred[m]
    return oof


def reg_metrics(y_true, y_pred) -> dict[str, float]:
    return {"r2": r2_score(y_true, y_pred), "mae": mean_absolute_error(y_true, y_pred)}


# --------------------------------------------------------------------------- aggregation
def bootstrap_ci(y_true, y_pred, stat, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        vals.append(stat(y_true[i], y_pred[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarize(target, models, metric_fn, ci_metric, ci_stat, truth, preds_by_rep, n_stocks, n_classes):
    """preds_by_rep[repeat][model] -> array. Returns summary rows (list of dicts)."""
    rows = []
    for m in models:
        per = pd.DataFrame([metric_fn(truth, preds_by_rep[r][m]) for r in range(len(preds_by_rep))])
        lo, hi = bootstrap_ci(truth, preds_by_rep[0][m], ci_stat)
        for met in per.columns:
            rows.append(dict(target=target, model=m, metric=met, mean=per[met].mean(),
                             sd=per[met].std(ddof=1) if len(per) > 1 else np.nan,
                             ci_lo=lo if met == ci_metric else np.nan, ci_hi=hi if met == ci_metric else np.nan,
                             n_stocks=n_stocks, n_repeats=len(per), n_classes=n_classes))
    return rows


# --------------------------------------------------------------------------- data
def sector_inputs(p):
    pos = month_end_positions(p)
    base = base_features(p, pos)
    smap, dropped = sector_map(p, sorted(base["ticker"].unique()))
    nvalid = p.ret.notna().sum()
    tickers = sorted(t for t in smap if t in p.ret.columns and nvalid[t] >= MIN_DAYS)
    fe = base[base["ticker"].isin(tickers)].replace([np.inf, -np.inf], np.nan)
    fe = fe.groupby("ticker")[FE_BASE_FEATURES].mean().reindex(tickers)
    C = pairwise_corr(p.ret[tickers]).to_numpy()
    classes = sorted({smap[t] for t in tickers})
    y = np.array([classes.index(smap[t]) for t in tickers])
    return tickers, classes, y, C, fe, dropped


def mcap_inputs(p):
    pos = month_end_positions(p)[-1:]
    base = base_features(p, pos)
    y = log_mcap_target(p, base)
    X = base[FE_BASE_FEATURES].replace([np.inf, -np.inf], np.nan)
    ok = np.isfinite(y.to_numpy()) & X.notna().all(axis=1).to_numpy()
    return base.loc[ok, "ticker"].tolist(), y[ok].to_numpy(), X[ok].reset_index(drop=True), base.loc[ok, "date"].iloc[0]


def _oof_rows(target, models, tickers, truth, preds_by_rep, fmt):
    parts = []
    for r, preds in enumerate(preds_by_rep):
        for m in models:
            parts.append(pd.DataFrame({"target": target, "model": m, "repeat": r, "ticker": tickers,
                                       "y_true": [fmt(v) for v in truth], "y_pred": [fmt(v) for v in preds[m]]}))
    return parts


def run(repeats: int = 10, out_dir="results", panel=None) -> pd.DataFrame:
    t0 = time.time()
    p = panel or build_panel()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, oofs = [], []

    tickers, classes, y, C, fe, dropped = sector_inputs(p)
    nc = len(classes)
    print(f"sector: {len(tickers)} stocks, {nc} classes {classes}; dropped {dropped}", flush=True)
    reps = []
    for r in range(repeats):
        reps.append(run_sector_repeat(C, y, fe, nc, r))
        print(f"  sector repeat {r} done ({time.time() - t0:.0f}s)", flush=True)
    rows += summarize("sector", SECTOR_MODELS, lambda t, q: class_metrics(t, q, nc), "macro_f1",
                      lambda t, q: _macro(t, q, nc), y, reps, len(tickers), nc)
    oofs += _oof_rows("sector", SECTOR_MODELS, tickers, [classes[v] for v in y],
                      [{m: [classes[v] for v in pr[m]] for m in SECTOR_MODELS} for pr in reps], str)

    mt, my, mX, mdate = mcap_inputs(p)
    print(f"log_mcap: {len(mt)} stocks at {pd.Timestamp(mdate).date()}", flush=True)
    reps = []
    for r in range(repeats):
        reps.append(run_mcap_repeat(mX, my, r))
        print(f"  log_mcap repeat {r} done ({time.time() - t0:.0f}s)", flush=True)
    rows += summarize("log_mcap", MCAP_MODELS, reg_metrics, "r2", r2_score, my, reps, len(mt), np.nan)
    oofs += _oof_rows("log_mcap", MCAP_MODELS, mt, my, reps, repr)

    res = pd.DataFrame(rows)
    res.to_csv(out / "static_cv.csv", index=False)
    pd.concat(oofs, ignore_index=True).to_parquet(out / "static_cv_oof.parquet", index=False)
    print(f"done in {time.time() - t0:.0f}s")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()
    res = run(a.repeats, a.out_dir)
    show = res[res["metric"].isin(["macro_f1", "weighted_f1", "accuracy", "r2", "mae"])]
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(show.drop(columns=["n_repeats"]).to_string(index=False))


if __name__ == "__main__":
    main()
