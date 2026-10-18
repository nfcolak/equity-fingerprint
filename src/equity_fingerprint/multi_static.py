"""v10 multi-market static targets: sector and log_mcap with European stocks (add-on, issuer-grouped).

European stocks only (``FOREIGN_REGIONS``; user decision 2026-10-18: the US references lowered the Swiss
sector score in the first v10 run and were removed, so the EU arm is a post-hoc choice, not a fresh test).

Inputs (multi_market.load_markets): the Swiss panel, one panel per foreign local index, the kept foreign
table. Folds are ISSUER-GROUPED everywhere (Swiss groups = `company` of universe.csv, foreign = normalised
name): outer StratifiedGroupKFold(5, shuffle, random_state=repeat) for sector, GroupKFold for log_mcap,
inner folds the same with INNER_SEED, on the outer train fold only.

Similarity menu (label-free, built once, NaN diagonal; all stocks = Swiss sector stocks + foreign stocks):
  d_local  daily residuals on the stock's local index (Swiss: ^SSHI), CH + EU stocks, union calendar,
           pairwise_corr(min_common=504).
  w_local  multi_market.weekly_resid (local index + equal-weight all-stock factor), all stocks,
           pairwise_corr(min_common=104).
kNN: argpartition versions of static_v9_sector.scan_predictions / knn_predict_prior (identical predictions),
inner search over (menu entry, k in KNN_KS, prior exponent a in A_GRID); ties: earlier entry, smaller k, a = 0.

Evaluation A `ch_query` (Swiss stocks are the only queries and scored stocks; foreign stocks are references):
  ch_knn_auto  control, static_cv.run_sector_repeat knn_auto on the grouped folds (Swiss refs only)
  ch_menu      menu + prior, Swiss refs only          ch_eu   + EU refs
  Foreign refs are restricted to the Swiss classes. PRIMARY comparison: ch_eu vs ch_knn_auto
  (rows `delta_<m>_vs_ch_knn_auto`, paired stock bootstrap of the repeat-averaged difference).
  log_mcap: only the grouped controls ch_ridge / ch_lgbm (static_cv.run_mcap_repeat with groups).
Evaluation B `pooled` (descriptive benchmark, every eligible stock is a query):
  sector: classes with >= 5 stocks overall, folds stratified on region|class, models all_knn
  (w_local, search over k and a) and baseline; metrics per region (CH, EU) and ALL, macro over the labels
  present in that region's truth.
  log_mcap: foreign stocks whose snapshot close x shares differs from marketCap by more than 0.25 in log are
  dropped (counts per region printed); target log(close CHF x shares), features FE_BASE_FEATURES + region
  one-hot; all_ridge, all_lgbm, baseline; r2 / mae per region and ALL.

Outputs (out dir): multi_static.csv (static_cv columns + eval, region), multi_static_oof.parquet,
multi_static_selection.csv. The CLI prints a reproduction check of the Swiss-only controls on the OLD
per-ticker folds against static_cv.csv (1e-9). Real data only inside main().

CLI: python -m equity_fingerprint.multi_static [--repeats 10] [--max-foreign N]
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import data
from . import multi_market as mm
from .static_cv import (INNER_SEED, KNN_KS, N_BOOT, _inner_splits, _macro, _masked_sims,
                        bootstrap_ci_mean, class_metrics, fast_class_metrics, fast_reg_metrics, issuer_ids,
                        mcap_inputs, pairwise_corr, reg_metrics, resid_on, run_mcap_repeat, run_sector_repeat,
                        sector_inputs, sector_matrices, summarize)
from .static_v9_sector import A_GRID, _first_max, _votes_by_k, delta_rows

MENU = ["d_local", "w_local"]
FOREIGN_REGIONS = ("EU",)
ARM_REFS = {"ch_menu": (), "ch_eu": ("EU",)}
CH_ARMS = ["ch_knn_auto", "ch_menu", "ch_eu"]
PRIMARY = ("ch_eu", "ch_knn_auto")
POOLED_MIN_CLASS = 5
SHARE_GAP = 0.25
REGIONS = ["CH", "EU"]
COLUMNS = ["target", "model", "metric", "mean", "sd", "ci_lo", "ci_hi", "n_stocks", "n_repeats", "n_classes",
           "eval", "region"]
SELECTION_COLUMNS = ["eval", "arm", "repeat", "fold", "entry", "k", "a"]
CLASS_METRICS = ["macro_f1", "weighted_f1", "accuracy", "balanced_accuracy"]


# --------------------------------------------------------------------------- argpartition kNN
def neighbours_ap(S: np.ndarray, kmax: int):
    """Top-kmax neighbours per row with the same indices and order as static_v9_sector._neighbours (stable
    argsort of -S, non-finite last), found with argpartition; rows whose k-th value is tied with values
    outside the selection use the stable argsort."""
    Sf = np.where(np.isfinite(S), S, -np.inf)
    n_r = Sf.shape[1]
    neg = -Sf
    if n_r <= kmax:
        idx = np.argsort(neg, axis=1, kind="stable")[:, :kmax]
    else:
        part = np.argpartition(neg, kmax - 1, axis=1)[:, :kmax]
        pv = np.take_along_axis(neg, part, axis=1)
        order = np.lexsort((part, pv), axis=1)
        idx = np.take_along_axis(part, order, axis=1)
        tied = np.flatnonzero((neg <= pv.max(axis=1)[:, None]).sum(axis=1) > kmax)
        if len(tied):
            idx[tied] = np.argsort(neg[tied], axis=1, kind="stable")[:, :kmax]
    w = np.clip(np.take_along_axis(Sf, idx, axis=1), 0.0, None)
    w[~np.isfinite(w)] = 0.0
    return idx, w


def scan_predictions_ap(S, y_ref, n_classes, majority, ks=KNN_KS, a_grid=A_GRID) -> dict:
    """{(k, a): predictions}; identical to static_v9_sector.scan_predictions, argpartition neighbour search."""
    y_ref = np.asarray(y_ref)
    prior = np.bincount(y_ref, minlength=n_classes) / len(y_ref)
    idx, w = neighbours_ap(S, max(ks))
    votes = _votes_by_k(idx, w, y_ref, ks, n_classes)
    out = {}
    for k in ks:
        any_w = w[:, :k].sum(1) > 0
        for a in a_grid:
            den = np.where(prior > 0, prior ** a, np.inf)
            pred = np.argmax(votes[k] / den, axis=1).astype(int)
            out[(k, a)] = np.where(any_w, pred, majority)
    return out


def knn_predict_prior_ap(S: np.ndarray, y_ref: np.ndarray, k: int, n_classes: int, majority: int,
                         prior: np.ndarray, a: float) -> np.ndarray:
    """Identical to static_v9_sector.knn_predict_prior, argpartition neighbour search."""
    y_ref = np.asarray(y_ref)
    idx, w = neighbours_ap(S, k)
    votes = _votes_by_k(idx, w, y_ref, [k], n_classes)[k]
    prior = np.asarray(prior, float)
    den = np.where(prior > 0, prior ** a, np.inf)
    pred = np.argmax(votes / den, axis=1).astype(int)
    return np.where(w.sum(1) > 0, pred, majority)


# --------------------------------------------------------------------------- search fold
def sector_search_fold(mats: dict, y: np.ndarray, train: np.ndarray, test: np.ndarray, ref_extra: np.ndarray,
                       entries: list, n_classes: int, groups: np.ndarray, strata: np.ndarray | None = None,
                       a_grid=A_GRID, ks=KNN_KS):
    """Predict `test` with the prior-corrected kNN whose (menu entry, k, a) is chosen by inner grouped CV
    over `train` (macro-F1, first max over entry -> k -> a). References = labelled stocks of the inner /
    outer train part plus `ref_extra` (foreign stocks, in every inner fold; never queries). Reads
    y[train] and y[ref_extra] only. groups / strata: full-length arrays (issuer id, stratification label;
    default y). Returns (predictions, (entry, k, a))."""
    train, test = np.asarray(train), np.asarray(test)
    ref_extra = np.asarray(ref_extra, dtype=int)
    strata = y if strata is None else strata
    score = {(e, k, a): [] for e in entries for k in ks for a in a_grid}
    for itr, iva in _inner_splits(np.asarray(strata)[train], INNER_SEED, np.asarray(groups)[train]):
        a_idx, b_idx = train[itr], train[iva]
        refs = np.concatenate([a_idx, ref_extra])
        maj = int(np.bincount(y[a_idx], minlength=n_classes).argmax())
        for e in entries:
            preds = scan_predictions_ap(_masked_sims(mats[e], refs, b_idx), y[refs], n_classes, maj, ks, a_grid)
            for key, p in preds.items():
                score[(e, key[0], key[1])].append(_macro(y[b_idx], p, n_classes))
    order = [(e, k, a) for e in entries for k in ks for a in a_grid]
    e, k, a = _first_max(score, order)
    majority = int(np.bincount(y[train], minlength=n_classes).argmax())
    refs = np.concatenate([train, ref_extra])
    pred = scan_predictions_ap(_masked_sims(mats[e], refs, test), y[refs], n_classes, majority, (k,), (a,))[(k, a)]
    return pred, (e, k, a)


def arm_entries(arm: str) -> list:
    """Menu entries searched by an arm (every arm: the full menu)."""
    if arm not in ARM_REFS:
        raise KeyError(arm)
    return list(MENU)


def ch_query_fold(mats: dict, y: np.ndarray, region: np.ndarray, groups: np.ndarray, train: np.ndarray,
                  test: np.ndarray, n_classes: int, arm: str, n_ch: int, a_grid=A_GRID):
    """One arm, one outer fold of evaluation A. Rows [0, n_ch) of mats / y are the Swiss stocks (the only
    queries); foreign references are those of the arm's regions with a Swiss class (y >= 0)."""
    train, test = np.asarray(train), np.asarray(test)
    if len(train) and train.max() >= n_ch or len(test) and test.max() >= n_ch:
        raise ValueError("ch_query queries and training stocks must be Swiss rows")
    extra = np.flatnonzero(np.isin(region, ARM_REFS[arm]) & (y >= 0) & (np.arange(len(y)) >= n_ch))
    return sector_search_fold(mats, y, train, test, extra, arm_entries(arm), n_classes, groups, None, a_grid)


def swiss_folds(y_ch: np.ndarray, groups_ch: np.ndarray, repeat: int):
    """The outer issuer-grouped folds of one repeat (the same split as static_cv.run_sector_repeat)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return list(StratifiedGroupKFold(5, shuffle=True, random_state=repeat).split(
            np.zeros(len(y_ch)), y_ch, groups_ch))


def ch_query_repeat(mats, y, region, groups, n_ch, n_classes, repeat, arms=None, selections=None, a_grid=A_GRID):
    """Out-of-fold class predictions of the menu arms for the Swiss stocks (every Swiss stock once)."""
    arms = [a for a in (arms or CH_ARMS) if a in ARM_REFS]
    oof = {a: np.full(n_ch, -1, dtype=int) for a in arms}
    for f, (tr, te) in enumerate(swiss_folds(y[:n_ch], groups[:n_ch], repeat)):
        for arm in arms:
            pred, sel = ch_query_fold(mats, y, region, groups, tr, te, n_classes, arm, n_ch, a_grid)
            oof[arm][te] = pred
            if selections is not None:
                selections.append(("ch_query", arm, repeat, f, *sel))
    return oof


def pooled_sector_repeat(mats, y, strata, groups, n_classes, repeat, selections=None, a_grid=A_GRID):
    """Evaluation B sector: every stock is a query; models all_knn (entry w_local) and baseline."""
    n = len(y)
    oof = {"all_knn": np.full(n, -1, dtype=int), "baseline": np.full(n, -1, dtype=int)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=repeat).split(np.zeros(n), strata, groups))
    for f, (tr, te) in enumerate(folds):
        pred, sel = sector_search_fold(mats, y, tr, te, np.zeros(0, int), ["w_local"], n_classes, groups, strata, a_grid)
        oof["all_knn"][te] = pred
        oof["baseline"][te] = int(np.bincount(y[tr], minlength=n_classes).argmax())
        if selections is not None:
            selections.append(("pooled", "all_knn", repeat, f, *sel))
    return oof


# --------------------------------------------------------------------------- metrics
def fast_present_metrics(y_true, y_pred, n_classes) -> dict[str, float]:
    """macro_f1 over the classes present in y_true, weighted_f1, accuracy (confusion-matrix based)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    cm = cm.astype(float)
    tp, sup, pred = np.diag(cm), cm.sum(1), cm.sum(0)
    den = sup + pred
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    present = sup > 0
    return {"macro_f1": float(f1[present].mean()), "weighted_f1": float((f1 * sup).sum() / sup.sum()),
            "accuracy": float(tp.sum() / cm.sum())}


def _tag(rows: list[dict], eval_: str, region: str) -> list[dict]:
    for r in rows:
        r["eval"], r["region"] = eval_, region
    return rows


def region_rows(target, models, truth, preds_by_rep, region, eval_, metric_fn, boot_fn, n_classes) -> list[dict]:
    """summarize() per region (CH, EU as present) and ALL; stocks are bootstrapped within the region."""
    rows = []
    for reg in [r for r in REGIONS if (region == r).any()] + ["ALL"]:
        m = np.ones(len(region), bool) if reg == "ALL" else region == reg
        if m.sum() < 2:
            continue
        sub = [{mod: np.asarray(p[mod])[m] for mod in models} for p in preds_by_rep]
        rows += _tag(summarize(target, models, metric_fn, boot_fn, np.asarray(truth)[m], sub, int(m.sum()), n_classes),
                     eval_, reg)
    return rows


def class_boot(n_classes, n_boot):
    return lambda t, ps: bootstrap_ci_mean(t, ps, lambda a, b: fast_present_metrics(a, b, n_classes), n_boot=n_boot)


def reg_boot(n_boot):
    return lambda t, ps: bootstrap_ci_mean(t, ps, fast_reg_metrics, n_boot=n_boot)


# --------------------------------------------------------------------------- data
class SectorData:
    """Aligned stock list (Swiss sector stocks first, then foreign), labels, groups and the menu matrices."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _weekly_view(swiss, panels, swiss_tickers, foreign_by_market):
    views = {"CH": SimpleNamespace(ret=swiss.ret[swiss_tickers], mkt=swiss.mkt)}
    for mkt, cols in foreign_by_market.items():
        views[mkt] = SimpleNamespace(ret=panels[mkt].ret[cols], mkt=panels[mkt].mkt)
    return views


def _in_panels(table: pd.DataFrame, panels: dict) -> pd.DataFrame:
    have = set().union(*[p.tickers for p in panels.values()]) if panels else set()
    return table[table["ticker"].isin(have)]


def build_sector_data(swiss, panels, table: pd.DataFrame, universe: pd.DataFrame, max_foreign: int | None = None,
                      seed: int = 0, verbose: bool = False) -> SectorData:
    """Menu matrices, labels and issuer groups for the Swiss sector stocks and the kept foreign stocks.

    table: kept foreign stocks (multi_market.load_markets); max_foreign: seeded subsample (timing runs)."""
    t0 = time.time()
    s_tickers, classes, y_s, C_s, fe_s, dropped = sector_inputs(swiss)
    tab = _in_panels(table[table["sector"].notna() & table["region"].isin(FOREIGN_REGIONS)], panels)
    if max_foreign is not None and max_foreign < len(tab):
        tab = tab.sample(n=max_foreign, random_state=seed).sort_index()
    tab = tab.reset_index(drop=True)
    f_tickers = tab["ticker"].tolist()
    tickers = s_tickers + f_tickers
    pos = {t: i for i, t in enumerate(tickers)}
    n_ch, n = len(s_tickers), len(tickers)
    region = np.array(["CH"] * n_ch + tab["region"].tolist())
    by_market = {m: [t for t in g["ticker"] if t in panels[m].tickers] for m, g in tab.groupby("market_index")}

    # ---- d_local: CH + EU only, daily residuals on the local index
    parts = [resid_on(swiss.ret[s_tickers].astype(float), swiss.mkt.to_numpy(float))]
    for m, cols in by_market.items():
        if panels[m].region == "EU" and cols:
            parts.append(resid_on(panels[m].ret[cols].astype(float), panels[m].mkt.to_numpy(float)))
    cal = parts[0].index
    for p in parts[1:]:
        cal = cal.union(p.index)
    res = pd.concat([p.reindex(cal) for p in parts], axis=1)
    c = pairwise_corr(res, min_common=504)
    d_local = np.full((n, n), np.nan)
    ix = [pos[t] for t in c.columns]
    d_local[np.ix_(ix, ix)] = c.to_numpy()
    del res, c
    # ---- w_local: weekly, two-factor residuals, all stocks
    wres = mm.weekly_resid(_weekly_view(swiss, panels, s_tickers, by_market))
    c = pairwise_corr(wres.astype(float), min_common=mm.DEDUPE_MIN_WEEKS)
    w_local = np.full((n, n), np.nan)
    ix = [pos[t] for t in c.columns]
    w_local[np.ix_(ix, ix)] = c.to_numpy()
    del wres, c
    if verbose:
        print(f"  menu matrices built: {n_ch} Swiss + {len(f_tickers)} foreign stocks ({time.time() - t0:.0f}s)",
              flush=True)

    sector = np.array([None] * n_ch + tab["sector"].tolist(), dtype=object)
    sector[:n_ch] = [classes[v] for v in y_s]
    y = np.array([classes.index(s) if s in classes else -1 for s in sector], dtype=int)
    g_ch = issuer_ids(s_tickers, universe)
    g_f = mm.issuer_groups(f_tickers, pd.DataFrame({"ticker": [], "company": []}), dict(zip(tab["ticker"], tab["name"])))
    groups = np.concatenate([g_ch, g_f + (g_ch.max() + 1 if len(g_ch) else 0)])
    return SectorData(tickers=tickers, region=region, sector=sector, y=y, classes=classes, groups=groups, n_ch=n_ch,
                      mats={"d_local": d_local, "w_local": w_local}, ctrl_C=C_s, ctrl_fe=fe_s, dropped=dropped,
                      ctrl_mats=sector_matrices(swiss, s_tickers), table=tab)


# --------------------------------------------------------------------------- evaluation A (sector)
def available_arms(region: np.ndarray, y: np.ndarray, n_ch: int) -> list[str]:
    """Menu arms whose reference regions have at least one Swiss-class foreign stock."""
    out = ["ch_menu"]
    have = {r for r in FOREIGN_REGIONS if ((region == r) & (y >= 0) & (np.arange(len(y)) >= n_ch)).any()}
    out += [a for a in ARM_REFS if a != "ch_menu" and set(ARM_REFS[a]) <= have]
    return out


def run_ch_control(sd: SectorData, repeat: int, grouped: bool, selections: list | None = None):
    """static_cv.run_sector_repeat on the Swiss stocks; returns the knn_auto out-of-fold predictions."""
    sels = []
    g = sd.groups[:sd.n_ch] if grouped else None
    oof = run_sector_repeat(sd.ctrl_C, sd.y[:sd.n_ch], sd.ctrl_fe, len(sd.classes), repeat, sd.ctrl_mats, sels, g)
    if selections is not None:
        selections += [("ch_query", "ch_knn_auto", repeat, f, mn, k, 0.0) for f, (mn, k) in enumerate(sels)]
    return oof["knn_auto"]


def evaluate_ch_query_sector(sd: SectorData, repeats: int, n_boot: int = N_BOOT, verbose: bool = False):
    """Rows, out-of-fold frame parts and selections of evaluation A (sector)."""
    t0 = time.time()
    n_ch, nc = sd.n_ch, len(sd.classes)
    arms = ["ch_knn_auto"] + available_arms(sd.region, sd.y, n_ch)
    reps, sels = [], []
    for r in range(repeats):
        pred = ch_query_repeat(sd.mats, sd.y, sd.region, sd.groups, n_ch, nc, r, arms[1:], sels)
        pred["ch_knn_auto"] = run_ch_control(sd, r, True, sels)
        reps.append(pred)
        if verbose:
            print(f"  ch_query sector repeat {r} done ({time.time() - t0:.0f}s)", flush=True)
    truth = sd.y[:n_ch]
    rows = _tag(summarize("sector", arms, lambda t, q: class_metrics(t, q, nc),
                          lambda t, ps: bootstrap_ci_mean(t, ps, lambda a, b: fast_class_metrics(a, b, nc),
                                                          n_boot=n_boot),
                          truth, reps, n_ch, nc), "ch_query", "CH")
    pairs = [(a, "ch_knn_auto") for a in arms[1:]] + [(a, "ch_menu") for a in arms[2:]]
    for ref in dict.fromkeys(r for _, r in pairs):
        rows += _tag(delta_rows("sector", [a for a, r in pairs if r == ref], ref, truth, reps, n_ch, nc, n_boot=n_boot),
                     "ch_query", "CH")
    oof = _oof_parts("ch_query", "sector", arms, sd.tickers[:n_ch], sd.region[:n_ch], truth, reps, sd.classes)
    return rows, oof, sels, reps


# --------------------------------------------------------------------------- evaluation B (sector)
def evaluate_pooled_sector(sd: SectorData, repeats: int, n_boot: int = N_BOOT, verbose: bool = False):
    t0 = time.time()
    sector = pd.Series(sd.sector)
    keep_classes = sorted(c for c, v in sector.value_counts().items() if v >= POOLED_MIN_CLASS)
    sel = np.flatnonzero(sector.isin(keep_classes).to_numpy())
    classes = keep_classes
    y = np.array([classes.index(sd.sector[i]) for i in sel])
    region = sd.region[sel]
    strata = pd.factorize(pd.Series(region) + "|" + pd.Series(y).astype(str))[0]
    groups = sd.groups[sel]
    M = {"w_local": sd.mats["w_local"][np.ix_(sel, sel)]}
    nc = len(classes)
    reps, sels = [], []
    for r in range(repeats):
        reps.append(pooled_sector_repeat(M, y, strata, groups, nc, r, sels))
        if verbose:
            print(f"  pooled sector repeat {r} done ({time.time() - t0:.0f}s)", flush=True)
    models = ["all_knn", "baseline"]
    rows = region_rows("sector", models, y, reps, region, "pooled", lambda t, q: fast_present_metrics(t, q, nc),
                       class_boot(nc, n_boot), nc)
    tickers = [sd.tickers[i] for i in sel]
    oof = _oof_parts("pooled", "sector", models, tickers, region, y, reps, classes)
    return rows, oof, sels


# --------------------------------------------------------------------------- log_mcap
def share_gap_filter(panels: dict, table: pd.DataFrame, threshold: float = SHARE_GAP):
    """Foreign stocks whose snapshot close x shares / marketCap is off by more than `threshold` in log are
    dropped (the share count is not the one behind the market cap). Local close = last adjusted close of the
    panel (pence lines /100). Returns (kept tickers, log frame ticker / region / gap / kept)."""
    rows = []
    for r in table.itertuples():
        p = next((p for p in panels.values() if r.ticker in p.tickers), None)
        last = p.adj[r.ticker].dropna() if p is not None else pd.Series(dtype=float)
        scale = 0.01 if r.currency == "GBp" else 1.0
        with np.errstate(all="ignore"):
            gap = (abs(np.log(last.iloc[-1] * scale * r.shares_outstanding / r.market_cap))
                   if len(last) else np.nan)
        rows.append(dict(ticker=r.ticker, region=r.region, gap=float(gap), kept=bool(np.isfinite(gap) and gap <= threshold)))
    log = pd.DataFrame(rows, columns=["ticker", "region", "gap", "kept"])
    return log.loc[log["kept"], "ticker"].tolist(), log


def build_mcap_data(swiss, panels, table: pd.DataFrame, universe: pd.DataFrame, max_foreign: int | None = None,
                    seed: int = 0):
    """(tickers, region, y, X with region one-hot, groups, n_ch, gap log) for the pooled log_mcap benchmark."""
    st, sy, sX, _ = mcap_inputs(swiss)
    tab = _in_panels(table[table["region"].isin(FOREIGN_REGIONS)], panels)
    if max_foreign is not None and max_foreign < len(tab):
        tab = tab.sample(n=max_foreign, random_state=seed).sort_index()
    kept, log = share_gap_filter(panels, tab)
    for reg in FOREIGN_REGIONS:
        sub = log[log["region"] == reg]
        if len(sub):
            print(f"  share-count gap filter {reg}: dropped {int((~sub['kept']).sum())} of {len(sub)} "
                  f"(|log gap| > {SHARE_GAP})", flush=True)
    tab = tab[tab["ticker"].isin(kept)].set_index("ticker")
    ft, fy, fX = [], [], []
    for p in panels.values():
        t_, y_, X_, _ = mcap_inputs(p)
        ok = np.array([t in tab.index for t in t_], dtype=bool)
        ft += [t for t, o in zip(t_, ok) if o]
        fy.append(y_[ok])
        fX.append(X_[ok])
    freg = tab.loc[ft, "region"].tolist()
    tickers = st + ft
    y = np.concatenate([sy] + fy) if fy else sy
    X = pd.concat([sX] + fX, ignore_index=True) if fX else sX
    region = np.array(["CH"] * len(st) + freg)
    X = X.assign(**{f"reg_{r}": (region == r).astype(float) for r in FOREIGN_REGIONS})
    g_ch = issuer_ids(st, universe)
    names = dict(zip(tab.index, tab["name"]))
    g_f = mm.issuer_groups(ft, pd.DataFrame({"ticker": [], "company": []}), names)
    groups = np.concatenate([g_ch, g_f + (g_ch.max() + 1 if len(g_ch) else 0)])
    return tickers, region, y, X.reset_index(drop=True), groups, len(st), log


def evaluate_mcap(swiss, panels, table, universe, repeats: int, n_boot: int = N_BOOT, max_foreign=None,
                  verbose: bool = False):
    """Rows / out-of-fold parts of log_mcap: grouped Swiss controls (ch_query) and the pooled benchmark."""
    t0 = time.time()
    rows, oofs = [], []
    st, sy, sX, _ = mcap_inputs(swiss)
    sg = issuer_ids(st, universe)
    reps = []
    for r in range(repeats):
        o = run_mcap_repeat(sX, sy, r, sg)
        reps.append({"ch_ridge": o["ridge"], "ch_lgbm": o["lgbm"]})
    rows += _tag(summarize("log_mcap", ["ch_ridge", "ch_lgbm"], reg_metrics, reg_boot(n_boot), sy, reps, len(st), np.nan),
                 "ch_query", "CH")
    oofs += _oof_parts("ch_query", "log_mcap", ["ch_ridge", "ch_lgbm"], st, np.array(["CH"] * len(st)), sy, reps, None)
    if verbose:
        print(f"  ch_query log_mcap done ({time.time() - t0:.0f}s)", flush=True)

    tickers, region, y, X, groups, n_ch, _ = build_mcap_data(swiss, panels, table, universe, max_foreign)
    reps = []
    for r in range(repeats):
        o = run_mcap_repeat(X, y, r, groups)
        reps.append({"all_ridge": o["ridge"], "all_lgbm": o["lgbm"], "baseline": o["baseline"]})
        if verbose:
            print(f"  pooled log_mcap repeat {r} done ({time.time() - t0:.0f}s)", flush=True)
    models = ["all_ridge", "all_lgbm", "baseline"]
    rows += region_rows("log_mcap", models, y, reps, region, "pooled", reg_metrics, reg_boot(n_boot), np.nan)
    oofs += _oof_parts("pooled", "log_mcap", models, tickers, region, y, reps, None)
    return rows, oofs


# --------------------------------------------------------------------------- outputs
def _oof_parts(eval_, target, models, tickers, region, truth, preds_by_rep, classes):
    """Out-of-fold frames: sector -> class names in y_true / y_pred, log_mcap -> floats in y_true_num / y_pred_num."""
    parts = []
    for r, preds in enumerate(preds_by_rep):
        for m in models:
            d = {"eval": eval_, "target": target, "model": m, "repeat": r, "ticker": list(tickers),
                 "region": list(region)}
            if classes is None:
                d.update(y_true=None, y_pred=None, y_true_num=np.asarray(truth, float),
                         y_pred_num=np.asarray(preds[m], float))
            else:
                d.update(y_true=[classes[v] for v in truth], y_pred=[classes[v] for v in preds[m]],
                         y_true_num=np.nan, y_pred_num=np.nan)
            parts.append(pd.DataFrame(d))
    return parts


def write_outputs(rows, oofs, sels, out_dir) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)[COLUMNS]
    res.to_csv(out / "multi_static.csv", index=False)
    pd.concat(oofs, ignore_index=True).to_parquet(out / "multi_static_oof.parquet", index=False)
    pd.DataFrame(sels, columns=SELECTION_COLUMNS).to_csv(out / "multi_static_selection.csv", index=False)
    return res


# --------------------------------------------------------------------------- reproduction check
def control_rows(sd: SectorData, mcap, repeats: int, n_boot: int = N_BOOT) -> pd.DataFrame:
    """static_cv rows (knn_auto sector, ridge / lgbm log_mcap) recomputed on the OLD per-ticker folds.
    mcap = (tickers, y, X) of the Swiss log_mcap inputs."""
    nc = len(sd.classes)
    truth = sd.y[:sd.n_ch]
    reps = [{"knn_auto": run_ch_control(sd, r, False)} for r in range(repeats)]
    rows = summarize("sector", ["knn_auto"], lambda t, q: class_metrics(t, q, nc),
                     lambda t, ps: bootstrap_ci_mean(t, ps, lambda a, b: fast_class_metrics(a, b, nc), n_boot=n_boot),
                     truth, reps, sd.n_ch, nc)
    mt, my, mX = mcap
    reps = []
    for r in range(repeats):
        o = run_mcap_repeat(mX, my, r)
        reps.append({"ridge": o["ridge"], "lgbm": o["lgbm"]})
    rows += summarize("log_mcap", ["ridge", "lgbm"], reg_metrics, reg_boot(n_boot), my, reps, len(mt), np.nan)
    return pd.DataFrame(rows)


def reproduction_check(new: pd.DataFrame, ref_csv) -> float | None:
    """Max |difference| of mean / sd / ci over the control rows against the committed static_cv.csv; None when
    the file is missing or has a different number of repeats."""
    ref_csv = Path(ref_csv)
    if not ref_csv.exists():
        return None
    ref = pd.read_csv(ref_csv)
    key = ["target", "model", "metric"]
    ref = ref.merge(new[key], on=key).set_index(key).sort_index()
    cur = new.set_index(key).sort_index()
    if ref.empty or int(ref["n_repeats"].iloc[0]) != int(cur["n_repeats"].iloc[0]):
        return None
    return float(max((cur.loc[ref.index, c] - ref[c]).abs().max() for c in ("mean", "sd", "ci_lo", "ci_hi")))


def print_reproduction(diff) -> None:
    if diff is None:
        print("reproduction check: skipped (static_cv.csv missing or different repeat count)")
    else:
        print(f"reproduction check (old per-ticker folds: knn_auto, ridge, lgbm vs static_cv.csv): "
              f"max abs diff {diff:.3e} ({'OK' if diff <= 1e-9 else 'MISMATCH'})", flush=True)


# --------------------------------------------------------------------------- driver
def run(swiss, panels, table, universe, repeats: int = 10, out_dir="results", n_boot: int = N_BOOT,
        max_foreign: int | None = None, ref_csv=None, repro: bool = True, verbose: bool = True) -> pd.DataFrame:
    t0 = time.time()
    sd = build_sector_data(swiss, panels, table, universe, max_foreign, verbose=verbose)
    print(f"sector: {sd.n_ch} Swiss stocks, {len(sd.classes)} classes {sd.classes}; dropped {sd.dropped}; "
          f"{len(sd.tickers) - sd.n_ch} foreign candidates "
          f"(EU {int((sd.region == 'EU').sum())})", flush=True)
    rows, oofs, sels, _ = evaluate_ch_query_sector(sd, repeats, n_boot, verbose)
    r2, o2, s2 = evaluate_pooled_sector(sd, repeats, n_boot, verbose)
    rows, oofs, sels = rows + r2, oofs + o2, sels + s2
    r3, o3 = evaluate_mcap(swiss, panels, table, universe, repeats, n_boot, max_foreign, verbose)
    rows, oofs = rows + r3, oofs + o3
    res = write_outputs(rows, oofs, sels, out_dir)
    if repro:
        ref = Path(ref_csv) if ref_csv is not None else Path(out_dir) / "static_cv.csv"
        st, sy, sX, _ = mcap_inputs(swiss)
        print_reproduction(reproduction_check(control_rows(sd, (st, sy, sX), repeats, n_boot), ref))
    print(f"done in {time.time() - t0:.0f}s")
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--max-foreign", type=int, default=None, help="seeded subsample of foreign stocks (timing only)")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--pretrain-dir", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--skip-repro", action="store_true", help="skip the old-fold reproduction check")
    a = ap.parse_args(argv)
    swiss, panels, table, drop_log = mm.load_markets(a.data_dir, a.pretrain_dir, FOREIGN_REGIONS)
    universe = data.load_universe(a.data_dir)
    res = run(swiss, panels, table, universe, a.repeats, a.out_dir, a.n_boot, a.max_foreign,
              ref_csv=Path(a.out_dir) / "static_cv.csv", repro=not a.skip_repro)
    show = res[res["metric"].isin(["macro_f1", "r2"]) | res["metric"].str.startswith("delta_macro_f1")]
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        print(show.drop(columns=["n_repeats"]).to_string(index=False))


if __name__ == "__main__":
    main()
