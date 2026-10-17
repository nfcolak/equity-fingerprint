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
  knn_resid: knn on correlations of SPI-residualised returns (OLS of each stock on panel.mkt over its valid
  days). knn_auto: inner CV on the train fold picks (matrix, k) over six label-free matrices (raw, resid,
  spearman_resid, resid_w4, pc1_resid, spearman_pc1_resid) x KNN_KS (tie -> earlier matrix, smaller k).
  Matrices are built once from unlabeled returns of all stocks.
  CI (every metric row): 2.5/97.5 percentiles of 1000 stock-bootstrap resamples of the repeat-averaged
  metric (same resampled stocks in every repeat).
log_mcap
  target = log(close x shares-outstanding snapshot) at the last month-end; features = FE_BASE_FEATURES
  at that date (no PC loadings). KFold(5) x 10 repeats. models: mean, ridge (alpha by inner CV), ridge_logtv21 (ridge on log_tv_21 only), lgbm
  (300 trees, lr 0.05, num_leaves {4,8} x min_child_samples {10,20} by inner CV). Metrics r2, mae.

Outputs (results/ by default): static_cv.csv, static_cv_oof.parquet (sector class names in y_true / y_pred,
log_mcap values as floats in y_true_num / y_pred_num), static_cv_knn_auto_selection.csv.
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
SECTOR_MODELS_V7 = SECTOR_MODELS + ["knn_resid", "knn_auto"]
MCAP_MODELS = ["baseline", "ridge", "ridge_logtv21", "lgbm"]
MATRIX_NAMES = ["raw", "resid", "spearman_resid", "resid_w4", "pc1_resid", "spearman_pc1_resid"]
WINSOR_SD = 4.0
TV_FEATURE = "log_tv_21"


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


# --------------------------------------------------------------------------- market-adjusted matrices (label-free)
def resid_on(ret: pd.DataFrame, factors, min_obs: int = 30) -> pd.DataFrame:
    """OLS (with intercept) of each column on factor column(s) over days where both are finite; residual
    r - a - b*f there, NaN elsewhere."""
    X = ret.to_numpy(float)
    F = np.asarray(factors, float).reshape(len(X), -1)
    out = np.full_like(X, np.nan)
    for j in range(X.shape[1]):
        ok = np.isfinite(X[:, j]) & np.isfinite(F).all(1)
        if ok.sum() < min_obs:
            continue
        A = np.column_stack([np.ones(ok.sum()), F[ok]])
        coef = np.linalg.lstsq(A, X[ok, j], rcond=None)[0]
        out[ok, j] = X[ok, j] - A @ coef
    return pd.DataFrame(out, index=ret.index, columns=ret.columns)


def pcs(ret: pd.DataFrame, m: int) -> np.ndarray:
    """First m principal-component scores of standardised returns (NaN filled with 0 for the SVD)."""
    X = ret.to_numpy(float)
    z = (X - np.nanmean(X, 0)) / np.nanstd(X, 0)
    z = np.where(np.isfinite(z), z, 0.0)
    U, s, _ = np.linalg.svd(z, full_matrices=False)
    return U[:, :m] * s[:m]


def spearman_like(ret: pd.DataFrame) -> pd.DataFrame:
    """Rank each stock's series over its own valid days (NaN stays NaN); Pearson on ranks ~ Spearman."""
    return ret.rank(axis=0)


def winsorise_sd(ret: pd.DataFrame, k: float = WINSOR_SD) -> pd.DataFrame:
    """Clip each stock's series at +-k of its own sd."""
    sd = ret.std(axis=0, skipna=True)
    return ret.clip(lower=-k * sd, upper=k * sd, axis=1)


def build_matrices(ret: pd.DataFrame, mkt) -> dict[str, np.ndarray]:
    """The six label-free similarity matrices (NaN diagonal) used by knn_resid / knn_auto."""
    m = np.asarray(pd.Series(mkt).reindex(ret.index), float)
    res = resid_on(ret, m)
    res_pc = resid_on(ret, pcs(ret, 1))
    return {"raw": pairwise_corr(ret).to_numpy(),
            "resid": pairwise_corr(res).to_numpy(),
            "spearman_resid": pairwise_corr(spearman_like(res)).to_numpy(),
            "resid_w4": pairwise_corr(winsorise_sd(res)).to_numpy(),
            "pc1_resid": pairwise_corr(res_pc).to_numpy(),
            "spearman_pc1_resid": pairwise_corr(spearman_like(res_pc)).to_numpy()}


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
                n_classes: int, mats: dict | None = None, selections: list | None = None) -> dict[str, np.ndarray]:
    """Predict class indices of `test` stocks for every sector model. Reads y[train] only.

    C: (n, n) correlation matrix with NaN diagonal; y: class indices (entries outside train ignored);
    fe: per-stock feature means (rows aligned with C). mats (optional, name -> matrix, MATRIX_NAMES) adds
    knn_resid and knn_auto; the chosen (matrix, k) of knn_auto is appended to `selections` if given.
    """
    train, test = np.asarray(train), np.asarray(test)
    y_tr = y[train]
    majority = int(np.bincount(y_tr, minlength=n_classes).argmax())

    # ---- inner CV: k for kNN, C for the three logits
    knn_score = {k: [] for k in KNN_KS}
    auto_score = {(mn, k): [] for mn in MATRIX_NAMES for k in KNN_KS} if mats is not None else {}
    logit_score = {(m, c): [] for m in ("prof_logit", "femean_logit", "prof_femean_logit") for c in LOGIT_CS}
    for itr, iva in _inner_splits(y_tr):
        a, b = train[itr], train[iva]
        maj = int(np.bincount(y[a], minlength=n_classes).argmax())
        S = _masked_sims(C, a, b)
        for k in KNN_KS:
            knn_score[k].append(_macro(y[b], knn_predict(S, y[a], k, n_classes, maj), n_classes))
        for mn in (MATRIX_NAMES if mats is not None else []):
            Sm = _masked_sims(mats[mn], a, b)
            for k in KNN_KS:
                auto_score[(mn, k)].append(_macro(y[b], knn_predict(Sm, y[a], k, n_classes, maj), n_classes))
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
    if mats is not None:
        best_kr = max(KNN_KS, key=lambda k: (np.mean(auto_score[("resid", k)]), -k))
        out["knn_resid"] = knn_predict(_masked_sims(mats["resid"], train, test), y_tr, best_kr, n_classes, majority)
        order = [(mn, k) for mn in MATRIX_NAMES for k in KNN_KS]   # first max wins: earlier matrix, smaller k
        best_mn, best_ka = max(order, key=lambda t: (np.mean(auto_score[t]), -order.index(t)))
        out["knn_auto"] = knn_predict(_masked_sims(mats[best_mn], train, test), y_tr, best_ka, n_classes, majority)
        if selections is not None:
            selections.append((best_mn, best_ka))
    return out


def class_metrics(y_true, y_pred, n_classes) -> dict[str, float]:
    lab = list(range(n_classes))
    return {"macro_f1": f1_score(y_true, y_pred, labels=lab, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, labels=lab, average="weighted", zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)}


def run_sector_repeat(C, y, fe, n_classes, repeat, mats=None, selections=None) -> dict[str, np.ndarray]:
    """One outer repeat: pooled out-of-fold class predictions (every stock exactly once)."""
    models = SECTOR_MODELS_V7 if mats is not None else SECTOR_MODELS
    oof = {m: np.full(len(y), -1, dtype=int) for m in models}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        folds = list(StratifiedKFold(5, shuffle=True, random_state=repeat).split(np.zeros(len(y)), y))
    for tr, te in folds:
        pred = sector_fold(C, y, fe, tr, te, n_classes, mats, selections)
        for m in models:
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
    tv_s = {a: [] for a in RIDGE_ALPHAS}
    lgbm_s = {i: [] for i in range(len(LGBM_GRID))}
    for itr, iva in inner:
        a, b = train[itr], train[iva]
        lin = Prep(cols, linear=True).fit(X.iloc[a])
        Xa, Xb = lin.transform(X.iloc[a]).to_numpy(), lin.transform(X.iloc[b]).to_numpy()
        raw = Prep(cols, linear=False).fit(X.iloc[a])
        Ra, Rb = raw.transform(X.iloc[a]), raw.transform(X.iloc[b])
        tv = Prep([TV_FEATURE], linear=True).fit(X.iloc[a][[TV_FEATURE]])
        Ta, Tb = tv.transform(X.iloc[a][[TV_FEATURE]]).to_numpy(), tv.transform(X.iloc[b][[TV_FEATURE]]).to_numpy()
        for al in RIDGE_ALPHAS:
            ridge_s[al].append(np.mean((y[b] - Ridge(alpha=al).fit(Xa, y[a]).predict(Xb)) ** 2))
            tv_s[al].append(np.mean((y[b] - Ridge(alpha=al).fit(Ta, y[a]).predict(Tb)) ** 2))
        for i, g in enumerate(LGBM_GRID):
            mdl = _lgbm(g).fit(Ra, y[a])
            lgbm_s[i].append(np.mean((y[b] - mdl.predict(Rb)) ** 2))
    best_a = min(RIDGE_ALPHAS, key=lambda a: (np.mean(ridge_s[a]), a))
    best_t = min(RIDGE_ALPHAS, key=lambda a: (np.mean(tv_s[a]), a))
    best_g = LGBM_GRID[min(lgbm_s, key=lambda i: (np.mean(lgbm_s[i]), i))]
    lin = Prep(cols, linear=True).fit(X.iloc[train])
    raw = Prep(cols, linear=False).fit(X.iloc[train])
    tvp = Prep([TV_FEATURE], linear=True).fit(X.iloc[train][[TV_FEATURE]])
    return {"baseline": np.full(len(test), float(y_tr.mean())),
            "ridge": Ridge(alpha=best_a).fit(lin.transform(X.iloc[train]).to_numpy(), y_tr)
            .predict(lin.transform(X.iloc[test]).to_numpy()),
            "ridge_logtv21": Ridge(alpha=best_t).fit(tvp.transform(X.iloc[train][[TV_FEATURE]]).to_numpy(), y_tr)
            .predict(tvp.transform(X.iloc[test][[TV_FEATURE]]).to_numpy()),
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
def fast_class_metrics(y_true, y_pred, n_classes) -> dict[str, float]:
    """Same four metrics as class_metrics, from a confusion matrix (fast enough for the bootstrap)."""
    cm = np.bincount(np.asarray(y_true) * n_classes + np.asarray(y_pred),
                     minlength=n_classes * n_classes).reshape(n_classes, n_classes).astype(float)
    tp, sup, pred = np.diag(cm), cm.sum(1), cm.sum(0)
    den = sup + pred
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    rec = tp[sup > 0] / sup[sup > 0]
    return {"macro_f1": f1.mean(), "weighted_f1": float((f1 * sup).sum() / sup.sum()),
            "accuracy": tp.sum() / cm.sum(), "balanced_accuracy": rec.mean()}


def fast_reg_metrics(y_true, y_pred) -> dict[str, float]:
    e = y_true - y_pred
    sst = ((y_true - y_true.mean()) ** 2).sum()
    return {"r2": 1.0 - (e ** 2).sum() / sst if sst > 0 else np.nan, "mae": np.abs(e).mean()}


def bootstrap_ci_mean(truth, preds, metric_fn, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict[str, tuple]:
    """Stock bootstrap of the repeat-averaged metric: each resample of stock indices is applied to every
    repeat's out-of-fold predictions, the metric is averaged over repeats, 2.5/97.5 percentiles over resamples.
    preds: list (one array per repeat). Returns {metric: (lo, hi)}."""
    rng = np.random.default_rng(seed)
    truth = np.asarray(truth)
    n = len(truth)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        per = [metric_fn(truth[i], np.asarray(p)[i]) for p in preds]
        vals.append({k: np.mean([d[k] for d in per]) for k in per[0]})
    df = pd.DataFrame(vals)
    return {k: (float(np.percentile(df[k].dropna(), 2.5)), float(np.percentile(df[k].dropna(), 97.5))) for k in df}


def summarize(target, models, metric_fn, boot_fn, truth, preds_by_rep, n_stocks, n_classes):
    """preds_by_rep[repeat][model] -> array. boot_fn(truth, pred_list) -> {metric: (lo, hi)}. Rows as dicts."""
    rows = []
    for m in models:
        per = pd.DataFrame([metric_fn(truth, preds_by_rep[r][m]) for r in range(len(preds_by_rep))])
        ci = boot_fn(truth, [pr[m] for pr in preds_by_rep])
        for met in per.columns:
            lo, hi = ci.get(met, (np.nan, np.nan))
            rows.append(dict(target=target, model=m, metric=met, mean=per[met].mean(),
                             sd=per[met].std(ddof=1) if len(per) > 1 else np.nan,
                             ci_lo=lo, ci_hi=hi, n_stocks=n_stocks, n_repeats=len(per), n_classes=n_classes))
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


def sector_matrices(p, tickers) -> dict[str, np.ndarray]:
    return build_matrices(p.ret[tickers], p.mkt)


def mcap_inputs(p):
    pos = month_end_positions(p)[-1:]
    base = base_features(p, pos)
    y = log_mcap_target(p, base)
    X = base[FE_BASE_FEATURES].replace([np.inf, -np.inf], np.nan)
    ok = np.isfinite(y.to_numpy()) & X.notna().all(axis=1).to_numpy()
    return base.loc[ok, "ticker"].tolist(), y[ok].to_numpy(), X[ok].reset_index(drop=True), base.loc[ok, "date"].iloc[0]


def _oof_rows(target, models, tickers, truth, preds_by_rep, numeric):
    """sector: class names in y_true / y_pred (numeric columns NaN); log_mcap: floats in y_true_num / y_pred_num."""
    parts = []
    for r, preds in enumerate(preds_by_rep):
        for m in models:
            d = {"target": target, "model": m, "repeat": r, "ticker": tickers}
            if numeric:
                d.update(y_true=None, y_pred=None, y_true_num=np.asarray(truth, float), y_pred_num=np.asarray(preds[m], float))
            else:
                d.update(y_true=list(truth), y_pred=list(preds[m]), y_true_num=np.nan, y_pred_num=np.nan)
            parts.append(pd.DataFrame(d))
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
    mats = sector_matrices(p, tickers)
    np.testing.assert_allclose(mats["raw"], C, equal_nan=True)
    print(f"  matrices built ({time.time() - t0:.0f}s)", flush=True)
    reps, sels = [], []
    for r in range(repeats):
        reps.append(run_sector_repeat(C, y, fe, nc, r, mats, sels))
        print(f"  sector repeat {r} done ({time.time() - t0:.0f}s)", flush=True)
    rows += summarize("sector", SECTOR_MODELS_V7, lambda t, q: class_metrics(t, q, nc),
                      lambda t, ps: bootstrap_ci_mean(t, ps, lambda a, b: fast_class_metrics(a, b, nc)),
                      y, reps, len(tickers), nc)
    oofs += _oof_rows("sector", SECTOR_MODELS_V7, tickers, [classes[v] for v in y],
                      [{m: [classes[v] for v in pr[m]] for m in SECTOR_MODELS_V7} for pr in reps], False)
    sel = pd.DataFrame(sels, columns=["matrix", "k"]).value_counts().rename("count").reset_index()
    sel.to_csv(out / "static_cv_knn_auto_selection.csv", index=False)
    print("knn_auto selections (matrix, k, count over outer folds):")
    print(sel.to_string(index=False), flush=True)

    mt, my, mX, mdate = mcap_inputs(p)
    print(f"log_mcap: {len(mt)} stocks at {pd.Timestamp(mdate).date()}", flush=True)
    reps = []
    for r in range(repeats):
        reps.append(run_mcap_repeat(mX, my, r))
        print(f"  log_mcap repeat {r} done ({time.time() - t0:.0f}s)", flush=True)
    rows += summarize("log_mcap", MCAP_MODELS, reg_metrics, lambda t, ps: bootstrap_ci_mean(t, ps, fast_reg_metrics),
                      my, reps, len(mt), np.nan)
    oofs += _oof_rows("log_mcap", MCAP_MODELS, mt, my, reps, True)

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
