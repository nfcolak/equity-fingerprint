"""v9 log_mcap benchmark: extended features, ridge/lgbm blend, multi-date training, optional TabPFN.

Same protocol as ``static_cv`` (log_mcap block): one row per stock, eligibility from ``mcap_inputs``
(latest month-end, 204 stocks), KFold(5, shuffle=True, random_state=repeat) for repeats 0..9, inner 5-fold CV
(INNER_SEED) on the outer training fold only, ``summarize`` + stock-bootstrap intervals of the repeat-averaged
metric. Labels of outer test stocks are never read (fits, inner CV, blend weights and multi-date targets use
y[train] only). Feature definitions (log_rebal_spike, VOLSTRUCT, SMI, FE_EXT) are documented in
``tabpfn_mcap`` (lightgbm-free; this module imports them from there and never imports torch/tabpfn).

Arms, in this order
  ridge, lgbm            static_cv models on FE_BASE_FEATURES (ridge reproduces static_cv.csv per repeat)
  avg_ridge_lgbm         w x ridge + (1 - w) x lgbm, w in {0, 0.1, .., 1} by inner-CV MSE of the inner
                         out-of-fold predictions of the already selected ridge / lgbm configurations
  ridge_ext, lgbm_ext    the same two models on FE_EXT (NaNs filled by Prep on the training fold)
  ridge_md               ridge on multi-date rows with FE_BASE_FEATURES, ridge_md_ext on FE_EXT
  tabpfn, tabpfn_ext     TabPFN on FE_BASE / FE_EXT, only if <out-dir>/static_v9_tabpfn_oof.parquet exists
                         (written by ``python -m equity_fingerprint.tabpfn_mcap``), else skipped with a note
  avg_ridge_md_ext_tabpfn_ext   0.5 x ridge_md_ext + 0.5 x tabpfn_ext. The weight is FIXED at 0.5: choosing it by
                         inner CV would need TabPFN fits inside every inner fold.
Multi-date training (md): the rows of the outer training stocks at the last K month-ends taken every ``step``
months (positions pos[-1 - step*j], j < K); row target y_t = y_now - (log close_now - log close_t) (identical
to log_mcap_target at t, shares are a snapshot); a test stock's prediction is the mean over its dates of
(model output + log close_now - log close_t). A stock lacking a date has fewer rows. (K, step, ridge alpha)
come from inner CV over stock folds (MSE of the date-averaged prediction of the validation stocks vs y_now)
from MD_GRID x RIDGE_ALPHAS; ties go to the smaller K, then smaller step, then smaller alpha.
Paired rows ``delta_r2_vs_ridge`` / ``delta_mae_vs_ridge`` for every arm but ridge: mean / sd over repeats of
metric(arm) - metric(ridge), CI = 2.5/97.5 percentiles of the stock-bootstrap (same resampled stocks for both
arms) repeat-averaged difference.

Issuer-grouped variant (v10, ``--grouped`` / ``run(grouped=True)``): ``groups`` (issuer id per stock, from
``static_cv.issuer_ids``) turns the outer folds into GroupKFold(5, shuffle, random_state=repeat) and every inner
KFold (pair_fold, md_fold) into GroupKFold on the training stocks (INNER_SEED); multi-date rows keep their stock's
group because folds are over stocks. Writes static_v9_mcap_grouped.csv, static_v9_mcap_grouped_oof.parquet and
static_v9_mcap_grouped_md_selection.csv; the TabPFN merge is not run (not in the headline) and the reproduction
check compares the ridge arm with static_cv_grouped.csv.

Outputs (out-dir, default results/): static_v9_mcap.csv (columns as static_cv.csv), static_v9_mcap_oof.parquet
(static_cv_oof layout), static_v9_mcap_md_selection.csv. The CLI prints a reproduction check of the ridge arm
against static_cv.csv (mean and sd of r2 / mae to 1e-9) and exits non-zero on a mismatch.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from .features import FE_BASE_FEATURES, base_features, build_panel, month_end_positions
from .models import RIDGE_ALPHAS, Prep
from .static_cv import (BOOT_SEED, INNER_SEED, LGBM_GRID, N_BOOT, _group_kfold, _lgbm, _oof_rows,
                        bootstrap_ci_mean, fast_reg_metrics, issuer_ids, reg_metrics, summarize)
from .tabpfn_mcap import (FE_EXT, TABPFN_FILE, ext_features, mcap_inputs_v9, outer_folds)

MD_GRID = [(1, 1), (3, 1), (6, 1), (12, 1), (6, 3)]
MD_MAX_LAG = max(step * (k - 1) for k, step in MD_GRID)
W_GRID = np.round(np.linspace(0.0, 1.0, 11), 1)
ARMS_STATIC = ["ridge", "lgbm", "avg_ridge_lgbm", "ridge_ext", "lgbm_ext", "ridge_md", "ridge_md_ext"]
ARMS_TABPFN = ["tabpfn", "tabpfn_ext", "avg_ridge_md_ext_tabpfn_ext"]
REF = "ridge"
TOL = 1e-9


def _lin(prep, X) -> np.ndarray:
    return np.nan_to_num(prep.transform(X).to_numpy())


# --------------------------------------------------------------------------- ridge / lgbm / blend on one row per stock
def _inner_folds(train: np.ndarray, groups) -> list:
    """Inner 5-fold splits of positions in `train`: per-stock KFold, or GroupKFold on groups[train]."""
    if groups is None:
        return list(KFold(5, shuffle=True, random_state=INNER_SEED).split(train))
    return _group_kfold(len(train), np.asarray(groups)[train], INNER_SEED)


def pair_fold(X: pd.DataFrame, y: np.ndarray, train: np.ndarray, test: np.ndarray,
              groups: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """ridge (alpha by inner CV), lgbm (grid by inner CV) and their blend for `test` stocks. Reads y[train] only.
    ridge / lgbm are exactly ``static_cv.mcap_fold``'s models (same inner folds, same tie rules)."""
    train, test = np.asarray(train), np.asarray(test)
    y_tr = y[train]
    cols = list(X.columns)
    inner = _inner_folds(train, groups)
    ridge_s = {a: [] for a in RIDGE_ALPHAS}
    lgbm_s = {i: [] for i in range(len(LGBM_GRID))}
    ridge_p = {a: np.zeros(len(train)) for a in RIDGE_ALPHAS}
    lgbm_p = {i: np.zeros(len(train)) for i in range(len(LGBM_GRID))}
    for itr, iva in inner:
        a, b = train[itr], train[iva]
        lin = Prep(cols, linear=True).fit(X.iloc[a])
        Xa, Xb = _lin(lin, X.iloc[a]), _lin(lin, X.iloc[b])
        raw = Prep(cols, linear=False).fit(X.iloc[a])
        Ra, Rb = raw.transform(X.iloc[a]), raw.transform(X.iloc[b])
        for al in RIDGE_ALPHAS:
            ridge_p[al][iva] = Ridge(alpha=al).fit(Xa, y[a]).predict(Xb)
            ridge_s[al].append(np.mean((y[b] - ridge_p[al][iva]) ** 2))
        for i, g in enumerate(LGBM_GRID):
            lgbm_p[i][iva] = _lgbm(g).fit(Ra, y[a]).predict(Rb)
            lgbm_s[i].append(np.mean((y[b] - lgbm_p[i][iva]) ** 2))
    best_a = min(RIDGE_ALPHAS, key=lambda a: (np.mean(ridge_s[a]), a))
    best_i = min(lgbm_s, key=lambda i: (np.mean(lgbm_s[i]), i))
    w = min(W_GRID, key=lambda w: (np.mean((y_tr - (w * ridge_p[best_a] + (1 - w) * lgbm_p[best_i])) ** 2), w))
    lin = Prep(cols, linear=True).fit(X.iloc[train])
    raw = Prep(cols, linear=False).fit(X.iloc[train])
    ridge = Ridge(alpha=best_a).fit(_lin(lin, X.iloc[train]), y_tr).predict(_lin(lin, X.iloc[test]))
    lgbm = _lgbm(LGBM_GRID[best_i]).fit(raw.transform(X.iloc[train]), y_tr).predict(raw.transform(X.iloc[test]))
    return {"ridge": ridge, "lgbm": lgbm, "avg": w * ridge + (1 - w) * lgbm}


# --------------------------------------------------------------------------- multi-date rows
class MDRows:
    """Rows of the eligible stocks at the last MD_MAX_LAG + 1 month-ends. g: stock index (into `tickers`),
    lag: months back from the latest month-end (position pos[-1 - lag]), X: FE_EXT, shift: log close_now -
    log close_t. Only rows with complete FE_BASE_FEATURES are kept (like the single-date eligibility)."""

    def __init__(self, g, lag, X, shift):
        self.g, self.lag, self.X, self.shift = g, lag, X, shift


def build_md_rows(p, tickers: list[str]) -> MDRows:
    allpos = month_end_positions(p)
    lags = [k for k in range(MD_MAX_LAG + 1) if k < len(allpos)]
    pos = np.array(sorted(allpos[-1 - k] for k in lags))
    base = base_features(p, pos)
    idx = {t: i for i, t in enumerate(tickers)}
    base = base[base["ticker"].isin(idx)].copy()
    X = pd.concat([base[FE_BASE_FEATURES].replace([np.inf, -np.inf], np.nan),
                   ext_features(p, base)], axis=1)[FE_EXT]
    keep = X[FE_BASE_FEATURES].notna().all(axis=1).to_numpy()
    base, X = base[keep], X[keep].reset_index(drop=True)
    g = base["ticker"].map(idx).to_numpy()
    lag = (len(allpos) - 1 - np.searchsorted(allpos, base["pos"].to_numpy())).astype(int)
    lc_now = np.full(len(tickers), np.nan)
    now = lag == 0
    lc_now[g[now]] = base["log_close"].to_numpy()[now]
    shift = lc_now[g] - base["log_close"].to_numpy()
    ok = np.isfinite(shift)
    return MDRows(g[ok], lag[ok], X[ok].reset_index(drop=True), shift[ok])


def _md_predict(rows: MDRows, y, fit_stocks, eval_stocks, cols, K, step) -> dict[float, np.ndarray]:
    """Per ridge alpha: date-averaged implied log mcap now for `eval_stocks`. Fit on rows of `fit_stocks`
    at the (K, step) dates with y[stock] - shift (only y[fit_stocks] is read)."""
    sel = np.isin(rows.lag, [step * j for j in range(K)])
    m_fit = sel & np.isin(rows.g, fit_stocks)
    m_ev = sel & np.isin(rows.g, eval_stocks)
    Xf, Xe = rows.X.loc[m_fit, cols], rows.X.loc[m_ev, cols]
    yf = y[rows.g[m_fit]] - rows.shift[m_fit]
    prep = Prep(cols, linear=True).fit(Xf)
    A, B = _lin(prep, Xf), _lin(prep, Xe)
    where = np.full(len(y), -1)
    where[np.asarray(eval_stocks)] = np.arange(len(eval_stocks))
    loc = where[rows.g[m_ev]]
    cnt = np.bincount(loc, minlength=len(eval_stocks))
    out = {}
    for al in RIDGE_ALPHAS:
        imp = Ridge(alpha=al).fit(A, yf).predict(B) + rows.shift[m_ev]
        out[al] = np.bincount(loc, weights=imp, minlength=len(eval_stocks)) / np.maximum(cnt, 1)
    return out


def md_fold(rows: MDRows, y: np.ndarray, train: np.ndarray, test: np.ndarray, cols: list[str], groups=None):
    """Multi-date ridge for `test` stocks; returns (predictions, (K, step, alpha)). Reads y[train] only."""
    train, test = np.asarray(train), np.asarray(test)
    score = {(k, s, a): [] for k, s in MD_GRID for a in RIDGE_ALPHAS}
    for itr, iva in _inner_folds(train, groups):
        a, b = train[itr], train[iva]
        for k, s in MD_GRID:
            for al, pr in _md_predict(rows, y, a, b, cols, k, s).items():
                score[(k, s, al)].append(np.mean((y[b] - pr) ** 2))
    best = min(score, key=lambda t: (np.mean(score[t]), t))
    k, s, al = best
    return _md_predict(rows, y, train, test, cols, k, s)[al], best


# --------------------------------------------------------------------------- one outer repeat
def run_repeat(Xb, Xe, rows, y, repeat, selections=None, groups=None) -> dict[str, np.ndarray]:
    """Pooled out-of-fold predictions of ARMS_STATIC for one repeat (every stock exactly once). groups (issuer
    ids) makes the outer and inner folds issuer-grouped; None is the per-stock KFold of ``outer_folds``."""
    oof = {m: np.full(len(y), np.nan) for m in ARMS_STATIC}
    folds = outer_folds(len(y), repeat) if groups is None else _group_kfold(len(y), np.asarray(groups), repeat)
    for f, (tr, te) in enumerate(folds):
        pb, pe = pair_fold(Xb, y, tr, te, groups), pair_fold(Xe, y, tr, te, groups)
        mb, sb = md_fold(rows, y, tr, te, FE_BASE_FEATURES, groups)
        me, se = md_fold(rows, y, tr, te, FE_EXT, groups)
        for name, v in (("ridge", pb["ridge"]), ("lgbm", pb["lgbm"]), ("avg_ridge_lgbm", pb["avg"]),
                        ("ridge_ext", pe["ridge"]), ("lgbm_ext", pe["lgbm"]), ("ridge_md", mb),
                        ("ridge_md_ext", me)):
            oof[name][te] = v
        if selections is not None:
            selections += [dict(repeat=repeat, fold=f, feature_set="base", K=sb[0], step=sb[1], alpha=sb[2]),
                           dict(repeat=repeat, fold=f, feature_set="ext", K=se[0], step=se[1], alpha=se[2])]
    return oof


# --------------------------------------------------------------------------- TabPFN merge
def merge_tabpfn(reps: list[dict], tickers: list[str], path) -> bool:
    """Add tabpfn, tabpfn_ext and the fixed 0.5 blend to every repeat dict in place. False (and a printed
    note) if the OOF file is missing."""
    path = Path(path)
    if not path.exists():
        print(f"note: {path} not found; tabpfn arms skipped (run python -m equity_fingerprint.tabpfn_mcap)")
        return False
    df = pd.read_parquet(path)
    for r, rep in enumerate(reps):
        for fs, arm in (("base", "tabpfn"), ("ext", "tabpfn_ext")):
            d = df[(df["feature_set"] == fs) & (df["repeat"] == r)].set_index("ticker")["y_pred"]
            if len(d) != len(tickers) or set(d.index) != set(tickers):
                raise ValueError(f"{path}: feature_set={fs} repeat={r} does not cover the {len(tickers)} stocks")
            rep[arm] = d.reindex(tickers).to_numpy(float)
        rep["avg_ridge_md_ext_tabpfn_ext"] = 0.5 * rep["ridge_md_ext"] + 0.5 * rep["tabpfn_ext"]
    return True


# --------------------------------------------------------------------------- paired deltas
def bootstrap_ci_delta(truth, preds_arm, preds_ref, metric_fn=fast_reg_metrics, n_boot: int = N_BOOT,
                       seed: int = BOOT_SEED) -> dict[str, tuple]:
    """Stock bootstrap of the repeat-averaged difference metric(arm) - metric(ref): the same resampled stocks
    are used for both arms and every repeat. preds_*: lists (one array per repeat). {metric: (lo, hi)}."""
    rng = np.random.default_rng(seed)
    truth = np.asarray(truth)
    n = len(truth)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        per = [{k: a[k] - b[k] for k in a} for a, b in
               ((metric_fn(truth[i], np.asarray(pa)[i]), metric_fn(truth[i], np.asarray(pb)[i]))
                for pa, pb in zip(preds_arm, preds_ref))]
        vals.append({k: np.mean([d[k] for d in per]) for k in per[0]})
    df = pd.DataFrame(vals)
    return {k: (float(np.percentile(df[k].dropna(), 2.5)), float(np.percentile(df[k].dropna(), 97.5))) for k in df}


def delta_rows(target, arm, ref, truth, reps, n_stocks, n_boot=N_BOOT) -> list[dict]:
    pa, pr = [r[arm] for r in reps], [r[ref] for r in reps]
    per = pd.DataFrame([{k: a[k] - b[k] for k in a} for a, b in
                        ((reg_metrics(truth, x), reg_metrics(truth, z)) for x, z in zip(pa, pr))])
    ci = bootstrap_ci_delta(truth, pa, pr, n_boot=n_boot)
    return [dict(target=target, model=arm, metric=f"delta_{m}_vs_{ref}", mean=per[m].mean(),
                 sd=per[m].std(ddof=1) if len(per) > 1 else np.nan, ci_lo=ci[m][0], ci_hi=ci[m][1],
                 n_stocks=n_stocks, n_repeats=len(per), n_classes=np.nan) for m in per.columns]


# --------------------------------------------------------------------------- driver
def reproduction_check(res: pd.DataFrame, ref_csv) -> bool | None:
    """Compare the ridge rows (mean, sd of r2 / mae) with the committed static_cv.csv; None if not comparable."""
    ref_csv = Path(ref_csv)
    if not ref_csv.exists():
        print(f"reproduction check: {ref_csv} not found, skipped")
        return None
    ref = pd.read_csv(ref_csv)
    ref = ref[(ref["target"] == "log_mcap") & (ref["model"] == REF) & ref["metric"].isin(["r2", "mae"])]
    new = res[(res["model"] == REF) & res["metric"].isin(["r2", "mae"])]
    if len(ref) == 0 or int(ref["n_repeats"].iloc[0]) != int(new["n_repeats"].iloc[0]):
        print("reproduction check: repeat count differs from the committed file, skipped")
        return None
    m = ref.merge(new, on="metric", suffixes=("_ref", "_new"))
    def gap(c):   # both NaN (sd with one repeat) counts as equal, one NaN as infinite
        a, b = m[f"{c}_ref"].to_numpy(float), m[f"{c}_new"].to_numpy(float)
        return np.where(np.isnan(a) & np.isnan(b), 0.0, np.abs(a - b)).max()
    diff = float(max(np.nan_to_num(gap(c), nan=np.inf) for c in ("mean", "sd")))
    ok = diff <= TOL
    print(f"reproduction check ridge vs {ref_csv.name}: max |diff| of mean/sd r2, mae = {diff:.2e} "
          f"({'OK' if ok else 'MISMATCH'}, tolerance {TOL:g})")
    return ok


def run(repeats: int = 10, out_dir="results", panel=None, ref_csv=None, n_boot: int = N_BOOT,
        grouped: bool = False, universe_csv=None):
    """Returns (summary frame, reproduction ok flag). grouped=True: issuer-grouped folds, `_grouped` file
    stem, no TabPFN merge, reproduction check against static_cv_grouped.csv."""
    t0 = time.time()
    stem = "static_v9_mcap_grouped" if grouped else "static_v9_mcap"
    p = panel or build_panel()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tickers, y, Xb, Xe, date = mcap_inputs_v9(p)
    rows = build_md_rows(p, tickers)
    mg = issuer_ids(tickers, universe_csv) if grouped else None
    print(f"log_mcap v9: {len(tickers)} stocks at {pd.Timestamp(date).date()}, {len(rows.g)} multi-date rows", flush=True)
    reps, sels = [], []
    for r in range(repeats):
        reps.append(run_repeat(Xb, Xe, rows, y, r, sels, mg))
        print(f"  repeat {r} done ({time.time() - t0:.0f}s)", flush=True)
    arms = list(ARMS_STATIC)
    if not grouped and merge_tabpfn(reps, tickers, out / TABPFN_FILE):
        arms += ARMS_TABPFN
    table = summarize("log_mcap", arms, reg_metrics,
                      lambda t, ps: bootstrap_ci_mean(t, ps, fast_reg_metrics, n_boot=n_boot),
                      y, reps, len(tickers), np.nan)
    by_arm = {a: [d for d in table if d["model"] == a] for a in arms}
    rows_out = list(by_arm[REF])
    for a in arms[1:]:
        rows_out += by_arm[a] + delta_rows("log_mcap", a, REF, y, reps, len(tickers), n_boot)
    res = pd.DataFrame(rows_out)
    res.to_csv(out / f"{stem}.csv", index=False)
    pd.concat(_oof_rows("log_mcap", arms, tickers, y, reps, True), ignore_index=True).to_parquet(
        out / f"{stem}_oof.parquet", index=False)
    sel = pd.DataFrame(sels)
    sel.to_csv(out / f"{stem}_md_selection.csv", index=False)
    print("multi-date selections (feature_set, K, step, count over outer folds):")
    print(sel.groupby(["feature_set", "K", "step"]).size().rename("count").reset_index().to_string(index=False))
    ok = reproduction_check(res, ref_csv or ("results/static_cv_grouped.csv" if grouped else "results/static_cv.csv"))
    print(f"done in {time.time() - t0:.0f}s")
    return res, ok


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--grouped", action="store_true", help="issuer-grouped folds -> static_v9_mcap_grouped*.{csv,parquet}")
    a = ap.parse_args(argv)
    res, ok = run(a.repeats, a.out_dir, grouped=a.grouped)
    show = res[res["metric"].str.contains("r2|mae")]
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(show.drop(columns=["n_repeats", "n_classes"]).to_string(index=False))
    if ok is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
