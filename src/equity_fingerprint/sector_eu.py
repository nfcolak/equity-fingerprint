"""v9 European references for the Swiss sector kNN (approved add-on; the headline rows stay Swiss-label-only).

Non-Swiss (non-US) stocks from data/pretrain (EF_PRETRAIN_DIR overrides the directory) whose yfinance sector
is one of the Swiss classes serve ONLY as labelled kNN references for Swiss query stocks; they are never
evaluated, never carry a .SW ticker (ValueError otherwise) and enter with data up to 2021-12-31 only.

cross space (data cut 2021-12-31 on both sides): EU log returns from adj_close (|r| >= 0.5 -> NaN) on the
Swiss calendar; factor = equal-weight mean EU return per day; Swiss and EU returns are residualised on it
(static_cv.resid_on); pairwise_corr(min_common=504) of the concatenation gives C_chch and C_cheu.
arms (k, and a for the last arm, by inner CV macro-F1 on the Swiss outer train fold; first max over k -> a):
  ch_le2021         Swiss train refs, C_chch
  eu_only           EU refs, C_cheu
  pooled            Swiss train + EU refs, <=2021 space
  pooled_full       Swiss-Swiss similarities from the full-history `resid` matrix, Swiss-EU from C_cheu
  pooled_full_prior pooled_full with the prior exponent a searched on A_GRID
Fallback when no neighbour has positive weight: majority class of the Swiss outer train fold.
The knn_auto reference row is copied from results/static_cv.csv, never recomputed. Paired rows
delta_<m>_vs_ch_le2021. Output: results/sector_eu.csv (static_cv columns + n_eu_refs).
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .features import build_panel
from .static_cv import (KNN_KS, N_BOOT, _inner_splits, _macro, _masked_sims, bootstrap_ci_mean, build_matrices,
                        class_metrics, fast_class_metrics, pairwise_corr, resid_on, sector_inputs, summarize)
from .static_v9_sector import A_GRID, CLASS_METRICS, delta_rows, scan_predictions

CUT = pd.Timestamp("2021-12-31")
MIN_EU_DAYS = 504
EU_ARMS = ["ch_le2021", "eu_only", "pooled", "pooled_full", "pooled_full_prior"]
REF_ARM = "ch_le2021"
STATIC_COLUMNS = ["target", "model", "metric", "mean", "sd", "ci_lo", "ci_hi", "n_stocks", "n_repeats", "n_classes"]


# --------------------------------------------------------------------------- European side (loading)
def pretrain_dir(path=None) -> Path:
    """Same resolution as pretrain_data._pretrain_dir (argument, EF_PRETRAIN_DIR, data/pretrain)."""
    if path is None:
        path = os.environ.get("EF_PRETRAIN_DIR")
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data" / "pretrain"
    return Path(path)


def load_eu_frames(path=None):
    """(prices, info, universe) frames from the pretrain directory (parquets in raw/, universe.csv in the root)."""
    root = pretrain_dir(path)
    raw = root / "raw" if (root / "raw" / "prices.parquet").exists() else root
    return (pd.read_parquet(raw / "prices.parquet"), pd.read_parquet(raw / "info.parquet"),
            pd.read_csv(root / "universe.csv"))


def check_no_swiss(tickers) -> None:
    bad = sorted(t for t in tickers if str(t).upper().endswith(".SW"))
    if bad:
        raise ValueError(f"European reference universe must not contain .SW tickers: {bad[:5]}")


def select_eu_universe(info: pd.DataFrame, uni: pd.DataFrame, price_tickers, classes) -> pd.Series:
    """ticker -> Swiss class name for non-US stocks with a yfinance sector in `classes` and prices.
    Raises ValueError if any selected ticker is a .SW ticker."""
    sec = info.set_index("ticker")["sector"].map(lambda s: s if s in classes else np.nan).dropna()
    ctry = uni.drop_duplicates("ticker").set_index("ticker")["country"]
    have = set(price_tickers)
    eu = [t for t in sec.index if ctry.get(t) not in (None, "US") and not pd.isna(ctry.get(t)) and t in have]
    check_no_swiss(eu)
    return sec.loc[eu]


def eu_returns(prices: pd.DataFrame, tickers, calendar: pd.DatetimeIndex, cut=CUT,
               min_valid: int = MIN_EU_DAYS) -> pd.DataFrame:
    """EU daily log returns (|r| >= 0.5 -> NaN) from adj_close up to `cut`, on the Swiss `calendar`; stocks
    with fewer than min_valid valid returns are dropped."""
    check_no_swiss(tickers)
    col = "adj_close" if "adj_close" in prices.columns else "close"
    w = prices[prices["ticker"].isin(list(tickers)) & (pd.to_datetime(prices["date"]) <= cut)]
    w = w.assign(date=pd.to_datetime(w["date"])).pivot(index="date", columns="ticker", values=col).sort_index()
    w = w.reindex(calendar.union(w.index)).sort_index()
    r = np.log(w).diff().reindex(calendar)
    r = r.where(r.abs() < 0.5)
    return r.loc[:, r.notna().sum() >= min_valid]


# --------------------------------------------------------------------------- cross space
def cross_space(ret_ch: pd.DataFrame, r_eu: pd.DataFrame, min_common: int = MIN_EU_DAYS):
    """C_chch (n x n, NaN diagonal) and C_cheu (n x m) of Swiss and EU returns residualised on the
    equal-weight EU mean return. Both frames share the (<= cut) Swiss calendar."""
    fac = r_eu.mean(axis=1).to_numpy()
    both = pd.concat([resid_on(ret_ch, fac), resid_on(r_eu, fac)], axis=1)
    C = pairwise_corr(both, min_common=min_common).to_numpy()
    n = ret_ch.shape[1]
    return C[:n, :n], C[:n, n:]


# --------------------------------------------------------------------------- experiment
def _blocks(arm: str, R: np.ndarray, Q: np.ndarray, y: np.ndarray, y_eu: np.ndarray, M: dict):
    """(similarity block, reference labels) of `arm` for Swiss refs R and Swiss queries Q (EU = all refs)."""
    eu = M["C_cheu"][Q]
    if arm == "eu_only":
        return eu, y_eu
    if arm == "ch_le2021":
        return _masked_sims(M["C_chch"], R, Q), y[R]
    if arm == "pooled":
        return np.hstack([_masked_sims(M["C_chch"], R, Q), eu]), np.concatenate([y[R], y_eu])
    return np.hstack([_masked_sims(M["C_full"], R, Q), eu]), np.concatenate([y[R], y_eu])


def _grid(arm: str):
    return A_GRID if arm == "pooled_full_prior" else (0,)


def eu_fold(M: dict, y: np.ndarray, y_eu: np.ndarray, train: np.ndarray, test: np.ndarray,
            n_classes: int) -> dict[str, np.ndarray]:
    """Predict the Swiss `test` stocks for every arm. Reads y[train] and y_eu only; EU stocks are references."""
    train, test = np.asarray(train), np.asarray(test)
    y_tr = y[train]
    majority = int(np.bincount(y_tr, minlength=n_classes).argmax())
    score = {arm: {(k, a): [] for k in KNN_KS for a in _grid(arm)} for arm in EU_ARMS}
    for itr, iva in _inner_splits(y_tr):
        a_idx, b_idx = train[itr], train[iva]
        maj = int(np.bincount(y[a_idx], minlength=n_classes).argmax())
        for arm in EU_ARMS:
            S, yr = _blocks(arm, a_idx, b_idx, y, y_eu, M)
            for ka, p in scan_predictions(S, yr, n_classes, maj, KNN_KS, _grid(arm)).items():
                score[arm][ka].append(_macro(y[b_idx], p, n_classes))
    out = {}
    for arm in EU_ARMS:
        order = list(score[arm])
        best = max(order, key=lambda t: (np.mean(score[arm][t]), -order.index(t)))
        S, yr = _blocks(arm, train, test, y, y_eu, M)
        out[arm] = scan_predictions(S, yr, n_classes, majority, KNN_KS, _grid(arm))[best]
    return out


def run_sector_eu(C_chch, C_cheu, C_full, y, y_eu, n_classes, repeats: int = 10, n_boot: int = N_BOOT,
                  eu_tickers=None, ref_rows: pd.DataFrame | None = None, verbose: bool = False,
                  swiss_refs: dict | None = None) -> pd.DataFrame:
    """All arms over `repeats` stratified 5-fold repeats of the Swiss stocks; EU stocks never are queries.
    swiss_refs (optional): name -> list (one per repeat) of committed Swiss-only out-of-fold class indices
    (same folds, e.g. static_cv knn_resid / knn_auto); adds paired delta rows of pooled_full(_prior) vs them."""
    if eu_tickers is not None:
        check_no_swiss(eu_tickers)
    y, y_eu = np.asarray(y), np.asarray(y_eu)
    assert C_chch.shape == (len(y), len(y)) and C_cheu.shape == (len(y), len(y_eu)) and C_full.shape == C_chch.shape
    M = {"C_chch": C_chch, "C_cheu": C_cheu, "C_full": C_full}
    t0 = time.time()
    reps = []
    for r in range(repeats):
        oof = {m: np.full(len(y), -1, dtype=int) for m in EU_ARMS}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            folds = list(StratifiedKFold(5, shuffle=True, random_state=r).split(np.zeros(len(y)), y))
        for tr, te in folds:
            pred = eu_fold(M, y, y_eu, tr, te, n_classes)
            for m in EU_ARMS:
                oof[m][te] = pred[m]
        reps.append(oof)
        if verbose:
            print(f"  sector_eu repeat {r} done ({time.time() - t0:.0f}s)", flush=True)
    n = len(y)
    rows = summarize("sector", EU_ARMS, lambda t, q: class_metrics(t, q, n_classes),
                     lambda t, ps: bootstrap_ci_mean(t, ps, lambda a, b: fast_class_metrics(a, b, n_classes), n_boot=n_boot),
                     y, reps, n, n_classes)
    rows += delta_rows("sector", [a for a in EU_ARMS if a != REF_ARM], REF_ARM, y, reps, n, n_classes, n_boot=n_boot)
    for ref_name, ref_preds in (swiss_refs or {}).items():  # paired deltas vs committed Swiss-only arms, same folds
        if len(ref_preds) != len(reps):
            raise ValueError(f"{ref_name}: {len(ref_preds)} repeats, expected {len(reps)}")
        both = [{**rep, ref_name: np.asarray(rp)} for rep, rp in zip(reps, ref_preds)]
        rows += delta_rows("sector", ["pooled_full", "pooled_full_prior"], ref_name, y, both, n, n_classes, n_boot=n_boot)
    res = pd.DataFrame(rows)
    res["n_eu_refs"] = len(y_eu)
    if ref_rows is not None and len(ref_rows):
        res = pd.concat([ref_rows.assign(n_eu_refs=np.nan), res], ignore_index=True)
    return res[STATIC_COLUMNS + ["n_eu_refs"]]


def reference_rows(ref_csv) -> pd.DataFrame | None:
    """The sector knn_auto rows of the committed static_cv.csv (copied, not recomputed)."""
    ref_csv = Path(ref_csv)
    if not ref_csv.exists():
        return None
    ref = pd.read_csv(ref_csv)
    return ref[(ref.target == "sector") & (ref.model == "knn_auto")][STATIC_COLUMNS].reset_index(drop=True)


SWISS_REF_MODELS = ("knn_resid", "knn_auto")


def swiss_oof(path, tickers, classes, repeats: int) -> dict | None:
    """Committed static_cv out-of-fold sector predictions (class indices, ticker order of `tickers`) of
    SWISS_REF_MODELS per repeat; None if the file is missing or lacks a repeat."""
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df = df[(df["target"] == "sector") & df["model"].isin(SWISS_REF_MODELS)]
    out = {}
    for m in SWISS_REF_MODELS:
        reps = []
        for r in range(repeats):
            s = df[(df["model"] == m) & (df["repeat"] == r)].set_index("ticker")["y_pred"]
            if set(s.index) != set(tickers):
                return None
            reps.append(np.array([classes.index(c) for c in s.reindex(tickers)]))
        out[m] = reps
    return out


def run(panel, repeats: int = 10, out_dir="results", pretrain_path=None) -> pd.DataFrame:
    tickers, classes, y, C, fe, dropped = sector_inputs(panel)
    nc = len(classes)
    prices, info, uni = load_eu_frames(pretrain_path)
    sec = select_eu_universe(info, uni, prices["ticker"].unique(), classes)
    ret = panel.ret[tickers]
    ret = ret[ret.index <= CUT]
    r_eu = eu_returns(prices, list(sec.index), ret.index)
    eu_t = list(r_eu.columns)
    y_eu = np.array([classes.index(sec[t]) for t in eu_t])
    print(f"sector_eu: {len(tickers)} Swiss stocks, {len(eu_t)} EU references; EU class counts:", flush=True)
    print(pd.Series(np.bincount(y_eu, minlength=nc), index=classes).to_string(), flush=True)
    C_chch, C_cheu = cross_space(ret, r_eu)
    C_full = build_matrices(panel.ret[tickers], panel.mkt)["resid"]
    out = Path(out_dir)
    ref_csv = out / "static_cv.csv"
    refs = reference_rows(ref_csv)
    swiss = swiss_oof(out / "static_cv_oof.parquet", tickers, classes, repeats)
    res = run_sector_eu(C_chch, C_cheu, C_full, y, y_eu, nc, repeats, eu_tickers=eu_t, ref_rows=refs, verbose=True,
                        swiss_refs=swiss)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "sector_eu.csv", index=False)
    if refs is not None:
        f = lambda m: float(res[(res.model == m) & (res.metric == "macro_f1")]["mean"].iloc[0])  # noqa: E731
        print(f"reference check: static_cv knn_auto macro_f1 {f('knn_auto'):.6f} (copied) vs {REF_ARM} {f(REF_ARM):.6f}")
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--pretrain-dir", default=None)
    a = ap.parse_args(argv)
    res = run(build_panel(), a.repeats, a.out_dir, a.pretrain_dir)
    show = res[res["metric"].isin(CLASS_METRICS) | res["metric"].str.startswith("delta_macro_f1")]
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(show.drop(columns=["n_repeats"]).to_string(index=False))


if __name__ == "__main__":
    main()
