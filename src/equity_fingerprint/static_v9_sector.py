"""v9 static sector: exponentially weighted correlation matrices and a prior-corrected kNN vote.

Same protocol as static_cv (eligibility, StratifiedKFold(5, shuffle, random_state=repeat), inner 5-fold CV
with INNER_SEED on the outer train fold only, summarize + stock-bootstrap intervals). Only Swiss data.

matrices: the six static_cv.MATRIX_NAMES plus three exponentially weighted SPI-residual matrices (rows
multiplied by sqrt(0.5^((n-1-i)/hl)) before pairwise_corr): resid_hl2y (hl 504), resid_hl5y (hl 1260),
resid_w4_hl5y (winsorised residuals, hl 1260).
vote: knn_predict_prior divides the class votes by prior_c^a (prior = class shares of the reference labels;
a = 0 is exactly static_cv.knn_predict; classes without reference stocks get no votes).
arms (inner CV macro-F1, first max over matrix -> k -> a): knn_auto (6 matrices, a=0, reproduces static_cv),
knn_auto_prior (6 x k x a), knn_auto_ew (9 matrices, a=0), knn_auto_v9 (9 x k x a).
Outputs: results/static_v9_sector.csv (static_cv columns; extra rows metric = delta_<m>_vs_knn_auto with a
stock-bootstrap interval of the repeat-averaged difference) and results/static_v9_sector_selection.csv.
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .features import build_panel
from .static_cv import (BOOT_SEED, KNN_KS, MATRIX_NAMES, N_BOOT, _inner_splits, _macro, _masked_sims,
                        build_matrices, bootstrap_ci_mean, class_metrics, fast_class_metrics, pairwise_corr,
                        resid_on, sector_inputs, summarize, winsorise_sd)

EW_HALF_LIVES = {"resid_hl2y": 504, "resid_hl5y": 1260, "resid_w4_hl5y": 1260}
EW_NAMES = list(EW_HALF_LIVES)
MATRIX_V9 = MATRIX_NAMES + EW_NAMES
A_GRID = (0, 0.25, 0.5, 0.75, 1.0)
ARMS = ["knn_auto", "knn_auto_prior", "knn_auto_ew", "knn_auto_v9"]
# arm -> (matrices searched, whether a is searched)
ARM_SPEC = {"knn_auto": (MATRIX_NAMES, False), "knn_auto_prior": (MATRIX_NAMES, True),
            "knn_auto_ew": (MATRIX_V9, False), "knn_auto_v9": (MATRIX_V9, True)}
CLASS_METRICS = ["macro_f1", "weighted_f1", "accuracy", "balanced_accuracy"]
REF_ARM = "knn_auto"


# --------------------------------------------------------------------------- matrices
def ew_weights(n: int, hl: float) -> np.ndarray:
    """Row weights sqrt(0.5^((n-1-i)/hl)) as a column vector (newest row weight 1)."""
    return np.sqrt(0.5 ** ((n - 1 - np.arange(n)) / hl))[:, None]


def ew_matrices(res: pd.DataFrame) -> dict[str, np.ndarray]:
    """The three exponentially weighted matrices from SPI-residual returns `res` (date x ticker)."""
    n = len(res)
    out = {}
    for name, hl in EW_HALF_LIVES.items():
        base = winsorise_sd(res) if "w4" in name else res
        out[name] = pairwise_corr(base * ew_weights(n, hl)).to_numpy()
    return out


def build_matrices_v9(ret: pd.DataFrame, mkt) -> dict[str, np.ndarray]:
    mats = build_matrices(ret, mkt)
    res = resid_on(ret, np.asarray(pd.Series(mkt).reindex(ret.index), float))
    mats.update(ew_matrices(res))
    return mats


# --------------------------------------------------------------------------- prior-corrected kNN
def _neighbours(S: np.ndarray, kmax: int):
    """Top-kmax neighbours per row (stable order as in knn_predict): indices and clipped-positive weights."""
    Sf = np.where(np.isfinite(S), S, -np.inf)
    idx = np.argsort(-Sf, axis=1, kind="stable")[:, :kmax]
    w = np.clip(np.take_along_axis(Sf, idx, axis=1), 0.0, None)
    w[~np.isfinite(w)] = 0.0
    return idx, w


def _votes_by_k(idx, w, y_ref, ks, n_classes):
    """Class votes (rows x classes) for each k, accumulated neighbour by neighbour (same order as bincount)."""
    m = idx.shape[0]
    rows = np.arange(m)
    votes = np.zeros((m, n_classes))
    out, kset = {}, set(ks)
    for j in range(idx.shape[1]):
        votes[rows, y_ref[idx[:, j]]] += w[:, j]
        if j + 1 in kset:
            out[j + 1] = votes.copy()
    for k in kset:  # fewer references than k: all of them are used
        out.setdefault(k, votes.copy())
    return out


def scan_predictions(S, y_ref, n_classes, majority, ks=KNN_KS, a_grid=A_GRID) -> dict:
    """{(k, a): predictions} for a similarity block, sharing the neighbour search between all (k, a)."""
    y_ref = np.asarray(y_ref)
    prior = np.bincount(y_ref, minlength=n_classes) / len(y_ref)
    idx, w = _neighbours(S, max(ks))
    votes = _votes_by_k(idx, w, y_ref, ks, n_classes)
    out = {}
    for k in ks:
        any_w = w[:, :k].sum(1) > 0
        for a in a_grid:
            den = np.where(prior > 0, prior ** a, np.inf)
            pred = np.argmax(votes[k] / den, axis=1).astype(int)
            out[(k, a)] = np.where(any_w, pred, majority)
    return out


def knn_predict_prior(S: np.ndarray, y_ref: np.ndarray, k: int, n_classes: int, majority: int,
                      prior: np.ndarray, a: float) -> np.ndarray:
    """knn_predict with class votes divided by prior_c^a; a = 0 reproduces static_cv.knn_predict exactly;
    classes with zero prior get no votes."""
    y_ref = np.asarray(y_ref)
    idx, w = _neighbours(S, k)
    votes = _votes_by_k(idx, w, y_ref, [k], n_classes)[k]
    prior = np.asarray(prior, float)
    den = np.where(prior > 0, prior ** a, np.inf)
    pred = np.argmax(votes / den, axis=1).astype(int)
    return np.where(w.sum(1) > 0, pred, majority)


# --------------------------------------------------------------------------- arms
def _first_max(scores: dict, order: list):
    """Key of the best mean score; ties go to the earlier entry of `order`."""
    return max(order, key=lambda t: (np.mean(scores[t]), -order.index(t)))


def sector_fold_v9(mats: dict, y: np.ndarray, train: np.ndarray, test: np.ndarray, n_classes: int,
                   selections: list | None = None) -> dict[str, np.ndarray]:
    """Predict `test` for the four arms. Reads y[train] only (inner CV, priors, reference labels)."""
    train, test = np.asarray(train), np.asarray(test)
    y_tr = y[train]
    majority = int(np.bincount(y_tr, minlength=n_classes).argmax())
    score = {(mn, k, a): [] for mn in MATRIX_V9 for k in KNN_KS for a in A_GRID}
    for itr, iva in _inner_splits(y_tr):
        a_idx, b_idx = train[itr], train[iva]
        maj = int(np.bincount(y[a_idx], minlength=n_classes).argmax())
        for mn in MATRIX_V9:
            preds = scan_predictions(_masked_sims(mats[mn], a_idx, b_idx), y[a_idx], n_classes, maj)
            for (k, a), p in preds.items():
                score[(mn, k, a)].append(_macro(y[b_idx], p, n_classes))
    final = {}
    out = {}
    for arm in ARMS:
        names, use_a = ARM_SPEC[arm]
        order = [(mn, k, a) for mn in names for k in KNN_KS for a in (A_GRID if use_a else (0,))]
        mn, k, a = _first_max(score, order)
        if mn not in final:
            final[mn] = scan_predictions(_masked_sims(mats[mn], train, test), y_tr, n_classes, majority)
        out[arm] = final[mn][(k, a)]
        if selections is not None:
            selections.append((arm, mn, k, a))
    return out


def run_sector_repeat_v9(mats: dict, y: np.ndarray, n_classes: int, repeat: int,
                         selections: list | None = None) -> dict[str, np.ndarray]:
    oof = {m: np.full(len(y), -1, dtype=int) for m in ARMS}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        folds = list(StratifiedKFold(5, shuffle=True, random_state=repeat).split(np.zeros(len(y)), y))
    for tr, te in folds:
        pred = sector_fold_v9(mats, y, tr, te, n_classes, selections)
        for m in ARMS:
            oof[m][te] = pred[m]
    return oof


# --------------------------------------------------------------------------- paired deltas
def bootstrap_ci_delta(truth, preds_arm, preds_ref, metric_fn, n_boot: int = N_BOOT,
                       seed: int = BOOT_SEED) -> dict[str, tuple]:
    """Stock bootstrap of the repeat-averaged difference metric(arm) - metric(ref); the same resampled
    stocks are used for both arms. preds_*: list (one array per repeat). Returns {metric: (lo, hi)}."""
    rng = np.random.default_rng(seed)
    truth = np.asarray(truth)
    n = len(truth)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        per = []
        for pa, pr in zip(preds_arm, preds_ref):
            ma, mr = metric_fn(truth[i], np.asarray(pa)[i]), metric_fn(truth[i], np.asarray(pr)[i])
            per.append({k: ma[k] - mr[k] for k in ma})
        vals.append({k: np.mean([d[k] for d in per]) for k in per[0]})
    df = pd.DataFrame(vals)
    return {k: (float(np.percentile(df[k].dropna(), 2.5)), float(np.percentile(df[k].dropna(), 97.5))) for k in df}


def delta_rows(target, arms, ref, truth, preds_by_rep, n_stocks, n_classes, n_boot: int = N_BOOT,
               extra: dict | None = None) -> list[dict]:
    """Rows metric = delta_<m>_vs_<ref> for every arm in `arms`: mean/sd over repeats of the per-repeat
    difference, stock-bootstrap CI of the repeat-averaged difference."""
    def mfn(t, q):
        return class_metrics(t, q, n_classes)

    def ffn(t, q):
        return fast_class_metrics(t, q, n_classes)

    rows = []
    for arm in arms:
        diffs = []
        for pr in preds_by_rep:
            ma, mr = mfn(truth, pr[arm]), mfn(truth, pr[ref])
            diffs.append({k: ma[k] - mr[k] for k in ma})
        diff = pd.DataFrame(diffs)
        ci = bootstrap_ci_delta(truth, [pr[arm] for pr in preds_by_rep], [pr[ref] for pr in preds_by_rep],
                                ffn, n_boot=n_boot)
        for met in diff.columns:
            lo, hi = ci.get(met, (np.nan, np.nan))
            row = dict(target=target, model=arm, metric=f"delta_{met}_vs_{ref}", mean=diff[met].mean(),
                       sd=diff[met].std(ddof=1) if len(diff) > 1 else np.nan, ci_lo=lo, ci_hi=hi,
                       n_stocks=n_stocks, n_repeats=len(diff), n_classes=n_classes)
            row.update(extra or {})
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- experiment
def run_experiment(mats: dict, y: np.ndarray, n_classes: int, repeats: int = 10, n_boot: int = N_BOOT,
                   verbose: bool = False):
    """Returns (results frame, selection-count frame). `mats`: name -> matrix for MATRIX_V9 (NaN diagonal)."""
    t0 = time.time()
    reps, sels = [], []
    for r in range(repeats):
        reps.append(run_sector_repeat_v9(mats, y, n_classes, r, sels))
        if verbose:
            print(f"  sector v9 repeat {r} done ({time.time() - t0:.0f}s)", flush=True)
    n = len(y)
    rows = summarize("sector", ARMS, lambda t, q: class_metrics(t, q, n_classes),
                     lambda t, ps: bootstrap_ci_mean(t, ps, lambda a, b: fast_class_metrics(a, b, n_classes), n_boot=n_boot),
                     y, reps, n, n_classes)
    rows += delta_rows("sector", [a for a in ARMS if a != REF_ARM], REF_ARM, y, reps, n, n_classes, n_boot=n_boot)
    sel = pd.DataFrame(sels, columns=["arm", "matrix", "k", "a"]).value_counts().rename("count").reset_index()
    sel = sel.sort_values(["arm", "count"], ascending=[True, False], kind="stable").reset_index(drop=True)
    return pd.DataFrame(rows), sel


def reproduction_check(res: pd.DataFrame, ref_csv, tol: float = 1e-9) -> float | None:
    """Max |difference| of the knn_auto sector rows (mean, sd) against the committed static_cv.csv; None if
    the file is missing or has a different number of repeats."""
    ref_csv = Path(ref_csv)
    if not ref_csv.exists():
        return None
    ref = pd.read_csv(ref_csv)
    ref = ref[(ref.target == "sector") & (ref.model == REF_ARM) & ref.metric.isin(CLASS_METRICS)].set_index("metric")
    new = res[(res.model == REF_ARM) & res.metric.isin(CLASS_METRICS)].set_index("metric")
    if ref.empty or int(ref["n_repeats"].iloc[0]) != int(new["n_repeats"].iloc[0]):
        return None
    return float(max((new[c] - ref.loc[new.index, c]).abs().max() for c in ("mean", "sd", "ci_lo", "ci_hi")))


def run(panel, repeats: int = 10, out_dir="results", ref_csv=None) -> pd.DataFrame:
    tickers, classes, y, C, fe, dropped = sector_inputs(panel)
    nc = len(classes)
    print(f"sector v9: {len(tickers)} stocks, {nc} classes {classes}", flush=True)
    mats = build_matrices_v9(panel.ret[tickers], panel.mkt)
    res, sel = run_experiment(mats, y, nc, repeats, verbose=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "static_v9_sector.csv", index=False)
    sel.to_csv(out / "static_v9_sector_selection.csv", index=False)
    diff = reproduction_check(res, ref_csv if ref_csv is not None else out / "static_cv.csv")
    if diff is None:
        print("reproduction check: skipped (static_cv.csv missing or different repeat count)")
    else:
        print(f"reproduction check knn_auto vs static_cv.csv: max abs diff {diff:.3e} ({'OK' if diff <= 1e-9 else 'MISMATCH'})")
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args(argv)
    res = run(build_panel(), a.repeats, a.out_dir)
    show = res[res["metric"].isin(CLASS_METRICS) | res["metric"].str.startswith("delta_macro_f1")]
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(show.drop(columns=["n_repeats"]).to_string(index=False))


if __name__ == "__main__":
    main()
