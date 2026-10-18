"""v10 longer Swiss history: vol_1m / beta_1y trained on rows before 2014-03, and a longer-history sector kNN.

User decision 2026-10-18 ("3- okay"): rows before 2014-03 enter the TRAIN split only; validation and test rows
are the control's. ^SSMI stands in for the market before ^SSHI starts (2013-03; daily corr 0.992), for the
features and for the beta_1y target of train rows (the series is spliced at the level of the index, so the
junction return is the ^SSMI return of that day; after the first ^SSHI date the series IS ^SSHI).

Arms (forward targets vol_1m, beta_1y; FE_FEATURES; linear / lgbm / avg as in ``forward_v7``)
  ctrl      the normal panel, rows from 2014-03; reproduces results/metrics.csv linear_fe / lgbm_fe.
  long2011  the cache from 2010-01-04 with the spliced market, rows from the first month-end with 253 rows
            (about 2011-01). No download. PRIMARY: avg vs ctrl avg.
A 2000-2009 download arm was run once and removed (user decision 2026-10-18: it added nothing over long2011).
Every choice (ridge alpha, lgbm grid and early stopping, the avg weight) is made on the validation rows of the
arm; test is scored once per arm. Validation and test rows of an arm must have the control's keys and target
values (checked, ``check_same_eval``): the extra rows are train rows only.
Deltas (R2 of the arm minus R2 of ctrl, same model kind): ticker-block and calendar-month-block bootstrap
(1000 draws, seed 42, ``forward_v7.bootstrap_delta_r2``); ``ci_lo``/``ci_hi`` hold the WIDER of the two, both are
kept in ``ci_lo_ticker`` ... ``ci_hi_month``. The primary row is model ``avg_long2011``, metric
``delta_r2_vs_ctrl``, split test; the validation delta stands beside it.

Quality table (descriptive, cached years): per calendar year, the year passes if >= 80 stocks have data, the
zero-volume share of listed stock-days is <= 15% and the NaN-volume share (before the panel's zero fill) is <= 15%.

Sector arm ``knn_auto_long`` (descriptive; 2026 labels on older returns): the six knn_auto matrices are built from
the arm's returns (from the arm start, spliced market) for the same stocks and classes as the control; folds are
the per-ticker ``static_cv`` folds (paired against the committed knn_auto out-of-fold predictions) and issuer-grouped
folds (paired against a grouped control rerun). Paired deltas use the stock bootstrap of ``static_v9_sector``.

CLI (real data only inside ``main``; the main agent runs it):
  --quality    per-year quality table of the cache -> history_quality.csv
  --arms [list] run the forward arms (+ sector arm) -> history_ext.csv, history_quality.csv
Outputs: results/history_ext.csv (columns ``COLUMNS``), results/history_quality.csv.

Run: env -u PYTHONPATH PYTHONPATH=src python -m equity_fingerprint.history_ext --quality
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from . import features as F
from . import forward_v7 as V
from . import static_cv as SC
from . import targets as T
from .data import PRICE_COLS, load_index, load_info, load_prices
from .models import regression_metrics
from .static_v9_sector import delta_rows

TARGETS = ("vol_1m", "beta_1y")
ARMS = ("ctrl", "long2011")
KINDS = ("linear", "lgbm", "avg")
REF_ARM = "ctrl"
CACHE_START = pd.Timestamp("2010-01-04")
SPLICE = "SPLICED_MKT"
MIN_STOCKS, MAX_ZERO_VOL, MAX_NAN_VOL = 80, 0.15, 0.15
SPLIT_RATIOS = (1.5, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0)
SPLIT_TOL = 0.04
EVAL_RTOL, EVAL_ATOL = 1e-9, 1e-12
ROOT = V.ROOT
COLUMNS = ["target", "model", "features", "split", "metric", "value", "n_obs", "ci_lo", "ci_hi", "selected",
           "region", "selection", "ci_lo_ticker", "ci_hi_ticker", "ci_lo_month", "ci_hi_month", "folds", "sd",
           "n_repeats"]
QUALITY_COLUMNS = ["year", "n_stocks", "n_cells", "zero_vol_share", "nan_vol_share", "artifact_returns",
                   "split_like_jumps", "pass_stocks", "pass_zero_vol", "pass_nan_vol", "passes", "admitted"]


# --------------------------------------------------------------------------- long panel
def _close(index: pd.DataFrame, ticker: str) -> pd.Series:
    s = (index.loc[index["ticker"] == ticker, ["date", "close"]].dropna().drop_duplicates("date")
         .set_index("date")["close"].sort_index())
    return s[s > 0]


def spliced_market(index: pd.DataFrame) -> pd.Series:
    """^SSHI closes, preceded by ^SSMI closes rescaled to the ^SSHI level on its first date.

    The junction log return is the ^SSMI return of the first ^SSHI day; from the first ^SSHI date on the series
    equals ^SSHI exactly (so validation / test rows keep their ^SSHI market)."""
    ssh, ssm = _close(index, F.MARKET), _close(index, "^SSMI")
    if ssh.empty or ssm.empty:
        raise ValueError(f"splicing needs both {F.MARKET} and ^SSMI in the index frame")
    first = ssh.index[0]
    before = ssm[ssm.index < first]
    ref = ssm[ssm.index <= first]
    if before.empty or ref.empty:
        return ssh
    return pd.concat([before * (ssh.iloc[0] / ref.iloc[-1]), ssh])


def splice_index(index: pd.DataFrame) -> pd.DataFrame:
    """Index frame plus a pseudo ticker ``SPLICE`` holding ``spliced_market`` (market name for panel_from_frames)."""
    s = spliced_market(index)
    extra = pd.DataFrame({"date": s.index, "ticker": SPLICE, "open": s.to_numpy(), "high": s.to_numpy(),
                          "low": s.to_numpy(), "close": s.to_numpy(), "adj_close": s.to_numpy(), "volume": 0.0})
    return pd.concat([index, extra[PRICE_COLS]], ignore_index=True)


def build_long_panel(start=CACHE_START, data_dir=None) -> F.Panel:
    """Panel of the cache from ``start`` on (``start`` must not precede the cache), built by
    ``features.panel_from_frames`` with the spliced ^SSMI / ^SSHI series as market."""
    start = pd.Timestamp(start)
    prices, index, info = load_prices(data_dir), load_index(data_dir), load_info(data_dir)
    if start < prices["date"].min():
        raise ValueError(f"start {start.date()} precedes the cached prices ({prices['date'].min().date()})")
    prices, index = prices[prices["date"] >= start], index[index["date"] >= start]
    return F.panel_from_frames(prices, splice_index(index), info, market=SPLICE)


def month_end_positions_from(panel: F.Panel, obs_start) -> np.ndarray:
    """``features.month_end_positions`` with a free first date (the original is tied to OBS_START). Positions
    before MIN_PRIOR_ROWS calendar rows are dropped: no stock has 253 rows there, and ``_max_drawdown`` slices
    252 rows back."""
    cal = panel.calendar
    s = pd.Series(np.arange(len(cal)), index=cal)
    last = s.groupby(cal.to_period("M")).last().iloc[:-1]  # the last month may be incomplete
    pos = last.to_numpy()
    return pos[(cal[pos] >= pd.Timestamp(obs_start)) & (pos >= F.MIN_PRIOR_ROWS)]


# --------------------------------------------------------------------------- quality table and year rule
def quality_table(prices: pd.DataFrame, info: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per calendar year: stocks with data, zero-volume and NaN-volume share of listed stock-days (the volume is
    read BEFORE the panel's zero fill; "listed" = inside a stock's first..last valid row), counts of artifact
    returns (|log return| > ARTIFACT_LOGRET) and split-like jumps (|log return| near log of 1.5, 2, 3, 4, 5, 10, 20).

    info: optional, restricts to CHF tickers (the panel's universe). Years without any data inside the range are
    kept as rows with zero stocks (they fail the rule)."""
    if info is not None:
        chf = set(info.loc[info["currency"] == "CHF", "ticker"])
        prices = prices[prices["ticker"].isin(chf)]
    adj = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    vol = prices.pivot(index="date", columns="ticker", values="volume").reindex(index=adj.index, columns=adj.columns)
    valid = adj.notna()
    span = (valid.cumsum() > 0) & (valid[::-1].cumsum()[::-1] > 0)
    r = np.log(adj.where(adj > 0).ffill().where(span)).diff()
    absr = r.abs()
    jump = pd.DataFrame(False, index=r.index, columns=r.columns)
    for ratio in SPLIT_RATIOS:
        jump |= (absr - np.log(ratio)).abs() < SPLIT_TOL
    yr = adj.index.year
    n_cells = span.groupby(yr).sum().sum(axis=1)
    zero = ((vol == 0) & span).groupby(yr).sum().sum(axis=1)
    nan = (vol.isna() & span).groupby(yr).sum().sum(axis=1)
    out = pd.DataFrame({
        "n_stocks": valid.groupby(yr).any().sum(axis=1),
        "n_cells": n_cells,
        "zero_vol_share": zero / n_cells.where(n_cells > 0),
        "nan_vol_share": nan / n_cells.where(n_cells > 0),
        "artifact_returns": (absr > F.ARTIFACT_LOGRET).groupby(yr).sum().sum(axis=1),
        "split_like_jumps": (jump & absr.notna()).groupby(yr).sum().sum(axis=1)})
    out = out.reindex(range(int(out.index.min()), int(out.index.max()) + 1))
    out[["n_stocks", "n_cells", "artifact_returns", "split_like_jumps"]] = (
        out[["n_stocks", "n_cells", "artifact_returns", "split_like_jumps"]].fillna(0).astype(int))
    out.index.name = "year"
    return apply_year_rule(out.reset_index())


def apply_year_rule(table: pd.DataFrame, min_stocks: int = MIN_STOCKS, max_zero: float = MAX_ZERO_VOL,
                    max_nan: float = MAX_NAN_VOL) -> pd.DataFrame:
    """Add pass_* / passes / admitted columns. A year passes with >= min_stocks stocks, zero-volume share <=
    max_zero and NaN-volume share <= max_nan (NaN shares fail). admitted = year >= the start year of
    ``admitted_start_year``."""
    t = table.copy()
    t["pass_stocks"] = t["n_stocks"] >= min_stocks
    t["pass_zero_vol"] = t["zero_vol_share"] <= max_zero
    t["pass_nan_vol"] = t["nan_vol_share"] <= max_nan
    t["passes"] = t["pass_stocks"] & t["pass_zero_vol"] & t["pass_nan_vol"]
    start = admitted_start_year(t)
    t["admitted"] = False if start is None else t["year"] >= start
    return t[[c for c in QUALITY_COLUMNS if c in t.columns]]


def admitted_start_year(table: pd.DataFrame) -> int | None:
    """First year from which every later year of the table passes (None when the last year fails). A missing year
    inside the range counts as failing."""
    t = table.sort_values("year")
    years = t["year"].astype(int).tolist()
    ok = dict(zip(years, t["passes"].astype(bool)))
    start = None
    for y in range(years[-1], years[0] - 1, -1):
        if not ok.get(y, False):
            break
        start = y
    return start


# --------------------------------------------------------------------------- forward arms
def arm_datasets(panel: F.Panel, obs_start=None, targets=TARGETS) -> dict[str, pd.DataFrame]:
    """{target: rows} built as ``run.main`` builds the time-target datasets (no fingerprint embeddings): month-end
    base features, PC loadings fitted on train dates, forward targets, time split, vol_1m windows with zero
    realised vol dropped. obs_start None = ``features.month_end_positions`` (the control)."""
    pos = F.month_end_positions(panel) if obs_start is None else month_end_positions_from(panel, obs_start)
    base = F.base_features(panel, pos)
    tickers = sorted(base["ticker"].unique())
    time_df = base.merge(F.pc_loadings(panel, F.pc_factors(panel, tickers), pos), on=V.KEYS, how="left")
    time_df = pd.concat([time_df, T.forward_targets(panel, time_df)], axis=1)
    out = {}
    for t in targets:
        d = time_df.copy()
        d["split"] = T.time_split(d["date"], d[f"end_{t}"])
        d = d[d[t].notna() & d["split"].notna()]
        if t == "vol_1m":
            d = d[~d["vol_1m_floored"]]
        out[t] = d
    return out


def fit_arm(d: pd.DataFrame, target: str) -> dict:
    """linear / lgbm / avg predictions (validation, test) of one arm; the avg weight is chosen on validation."""
    tr, va, te = (d[d["split"] == s] for s in ("train", "val", "test"))
    va, te = (x.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True) for x in (va, te))
    fp = V.fit_predict(tr, va, te, F.FE_FEATURES, y_col=target)
    w = V.choose_weight(va[target], fp["linear"][0], fp["lgbm"][0])
    preds = {"linear": fp["linear"], "lgbm": fp["lgbm"],
             "avg": tuple(w * fp["linear"][i] + (1 - w) * fp["lgbm"][i] for i in (0, 1))}
    keep = ["date", "ticker", target, f"end_{target}"]
    return {"preds": preds, "weight": w, "val": va[keep], "test": te[keep],
            "n_rows": {"train": len(tr), "val": len(va), "test": len(te)},
            "n_tickers": {"train": int(tr["ticker"].nunique()), "val": int(va["ticker"].nunique()),
                          "test": int(te["ticker"].nunique())}}


def check_same_eval(ctrl: dict, arm: dict, target: str) -> None:
    """Raise unless the arm's validation and test rows have the control's keys and target values (floats to 1e-9
    relative: pandas rolling windows carry prefix-dependent rounding)."""
    for split in ("val", "test"):
        a, b = ctrl[split], arm[split]
        if len(a) != len(b) or not (a["ticker"].to_numpy() == b["ticker"].to_numpy()).all() \
                or not (a["date"].to_numpy() == b["date"].to_numpy()).all():
            raise ValueError(f"{target} {split}: row keys differ from the control")
        if not (a[f"end_{target}"].to_numpy() == b[f"end_{target}"].to_numpy()).all():
            raise ValueError(f"{target} {split}: forward-window end dates differ from the control")
        if not np.allclose(a[target].to_numpy(float), b[target].to_numpy(float), rtol=EVAL_RTOL, atol=EVAL_ATOL):
            raise ValueError(f"{target} {split}: target values differ from the control")


def _row(target, model, features, split, metric, value, n_obs, selected=0, **extra) -> dict:
    row = dict(target=target, model=model, features=features, split=split, metric=metric, value=float(value),
               n_obs=int(n_obs), ci_lo=np.nan, ci_hi=np.nan, selected=int(selected), region="CH",
               selection="ch_val", ci_lo_ticker=np.nan, ci_hi_ticker=np.nan, ci_lo_month=np.nan,
               ci_hi_month=np.nan, folds="", sd=np.nan, n_repeats=np.nan)
    row.update(extra)
    return row


def _delta_ci(y, pa, pb, tickers, dates, n_boot) -> dict:
    lo_t, hi_t = V.bootstrap_delta_r2(y, pa, pb, tickers, n=n_boot)
    lo_m, hi_m = V.bootstrap_delta_r2(y, pa, pb, dates, n=n_boot)  # one block per month-end date
    return dict(ci_lo=min(lo_t, lo_m), ci_hi=max(hi_t, hi_m), ci_lo_ticker=lo_t, ci_hi_ticker=hi_t,
                ci_lo_month=lo_m, ci_hi_month=hi_m)


def forward_rows(target: str, fits: dict[str, dict], n_boot: int = V.BOOT_N) -> list[dict]:
    """Metric rows of every arm for one target. fits: arm -> ``fit_arm``; must contain ``ctrl``."""
    ctrl = fits[REF_ARM]
    for arm, fit in fits.items():
        if arm != REF_ARM:
            check_same_eval(ctrl, fit, target)
    val_r2 = {(arm, k): V.r2(f["val"][target], f["preds"][k][0]) for arm, f in fits.items() for k in KINDS}
    best = max(val_r2, key=val_r2.get)  # first max on ties: arms in dict order, kinds in KINDS order
    rows = []
    for arm, f in fits.items():
        for split in ("train", "val", "test"):
            rows.append(_row(target, arm, "data", split, "n_rows", f["n_rows"][split], f["n_rows"][split]))
            rows.append(_row(target, arm, "data", split, "n_tickers", f["n_tickers"][split], f["n_rows"][split]))
        for k in KINDS:
            model, sel = f"{k}_{arm}", int((arm, k) == best)
            for si, (split, part) in enumerate((("val", f["val"]), ("test", f["test"]))):
                y, p = part[target].to_numpy(float), f["preds"][k][si]
                for m, v in regression_metrics(y, p, False).items():
                    rows.append(_row(target, model, "fe", split, m, v, len(part), sel))
                if arm != REF_ARM:
                    pr = ctrl["preds"][k][si]
                    rows.append(_row(target, model, "fe", split, "delta_r2_vs_ctrl", V.r2(y, p) - V.r2(y, pr),
                                     len(part), sel, **_delta_ci(y, p, pr, part["ticker"].to_numpy(),
                                                                part["date"].to_numpy(), n_boot)))
            if k == "avg":
                rows.append(_row(target, model, "fe", "val", "weight_linear", f["weight"], len(f["val"]), sel))
    return rows


def forward_experiment(datasets: dict[str, dict[str, pd.DataFrame]], targets=TARGETS, n_boot: int = V.BOOT_N,
                       verbose: bool = False) -> pd.DataFrame:
    """datasets: arm -> {target: rows} (``arm_datasets``); ``ctrl`` required. Returns the forward rows."""
    if REF_ARM not in datasets:
        raise ValueError("the control arm 'ctrl' is required")
    rows = []
    for t in targets:
        fits = {}
        for arm in datasets:
            fits[arm] = fit_arm(datasets[arm][t], t)
            if verbose:
                print(f"[{t}] {arm}: rows {fits[arm]['n_rows']}, avg weight {fits[arm]['weight']}", flush=True)
        rows += forward_rows(t, fits, n_boot)
    return pd.DataFrame(rows, columns=COLUMNS)


def reproduction_check(out: pd.DataFrame, metrics_csv) -> pd.DataFrame:
    """Compare the ctrl linear / lgbm r2 and mae rows with results/metrics.csv (linear_fe / lgbm_fe)."""
    ref = pd.read_csv(metrics_csv)
    ref = ref[ref["target"].isin(TARGETS) & ref["model"].isin(["linear_fe", "lgbm_fe"])
              & ref["metric"].isin(["r2", "mae"])]
    new = out[out["model"].isin(["linear_ctrl", "lgbm_ctrl"]) & out["metric"].isin(["r2", "mae"])].copy()
    new["model"] = new["model"].str.replace("_ctrl", "_fe")
    k = ["target", "model", "split", "metric"]
    mg = ref.merge(new, on=k, suffixes=("_metrics", "_v10"))
    mg["abs_diff"] = (mg["value_metrics"] - mg["value_v10"]).abs()
    mg["n_equal"] = mg["n_obs_metrics"] == mg["n_obs_v10"]
    return mg[k + ["value_metrics", "value_v10", "abs_diff", "n_obs_metrics", "n_obs_v10", "n_equal"]]


# --------------------------------------------------------------------------- sector arm
def knn_auto_fold(mats: dict, y: np.ndarray, train: np.ndarray, test: np.ndarray, n_classes: int,
                  groups: np.ndarray | None = None, selections: list | None = None) -> np.ndarray:
    """``static_cv.sector_fold``'s knn_auto alone: inner CV over (matrix, k) on the train fold (tie -> earlier
    matrix, smaller k), then the chosen kNN on the test stocks. Reads y[train] only."""
    train, test = np.asarray(train), np.asarray(test)
    y_tr = y[train]
    majority = int(np.bincount(y_tr, minlength=n_classes).argmax())
    score = {(mn, k): [] for mn in SC.MATRIX_NAMES for k in SC.KNN_KS}
    for itr, iva in SC._inner_splits(y_tr, groups=None if groups is None else np.asarray(groups)[train]):
        a, b = train[itr], train[iva]
        maj = int(np.bincount(y[a], minlength=n_classes).argmax())
        for mn in SC.MATRIX_NAMES:
            Sm = SC._masked_sims(mats[mn], a, b)
            for k in SC.KNN_KS:
                score[(mn, k)].append(SC._macro(y[b], SC.knn_predict(Sm, y[a], k, n_classes, maj), n_classes))
    order = [(mn, k) for mn in SC.MATRIX_NAMES for k in SC.KNN_KS]
    best_mn, best_k = max(order, key=lambda t: (np.mean(score[t]), -order.index(t)))
    if selections is not None:
        selections.append((best_mn, best_k))
    return SC.knn_predict(SC._masked_sims(mats[best_mn], train, test), y_tr, best_k, n_classes, majority)


def knn_auto_repeat(mats: dict, y: np.ndarray, n_classes: int, repeat: int, groups: np.ndarray | None = None,
                    selections: list | None = None) -> np.ndarray:
    """One outer repeat (every stock predicted once). groups (issuer ids): StratifiedGroupKFold(5, shuffle,
    random_state=repeat); None: the per-stock StratifiedKFold of static_cv."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if groups is None:
            folds = list(StratifiedKFold(5, shuffle=True, random_state=repeat).split(np.zeros(len(y)), y))
        else:
            folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=repeat).split(np.zeros(len(y)), y, groups))
    oof = np.full(len(y), -1, dtype=int)
    for tr, te in folds:
        oof[te] = knn_auto_fold(mats, y, tr, te, n_classes, groups, selections)
    return oof


def committed_knn_auto(path, tickers, classes, repeats: int) -> list[np.ndarray]:
    """The committed knn_auto out-of-fold class indices (results/static_cv_oof.parquet) per repeat."""
    df = pd.read_parquet(path, columns=["target", "model", "repeat", "ticker", "y_pred"])
    df = df[(df["target"] == "sector") & (df["model"] == "knn_auto")]
    idx = {c: i for i, c in enumerate(classes)}
    out = []
    for r in range(repeats):
        s = df[df["repeat"] == r].set_index("ticker")["y_pred"].reindex(list(tickers))
        if s.isna().any():
            raise ValueError(f"{path}: knn_auto repeat {r} lacks {int(s.isna().sum())} of the stocks")
        out.append(s.map(idx).to_numpy(int))
    return out


def long_sector_matrices(panel: F.Panel, tickers) -> dict[str, np.ndarray]:
    """The six knn_auto matrices from the long panel's returns and spliced market (same stocks as the control)."""
    missing = [t for t in tickers if t not in panel.ret.columns]
    if missing:
        raise ValueError(f"long panel lacks {len(missing)} control stocks, e.g. {missing[:3]}")
    return SC.build_matrices(panel.ret[list(tickers)], panel.mkt)


def sector_experiment(classes, y, mats: dict[str, dict], groups, repeats: int = 10, committed=None,
                      n_boot: int = SC.N_BOOT, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """knn_auto on the control matrices (mats['ctrl']) and on every long arm's matrices.

    folds 'ticker': reference = the committed knn_auto OOF when given (else the control rerun); folds 'issuer':
    reference = the grouped control rerun. Rows: model knn_auto (features committed / ctrl) and knn_auto_long
    (features = arm), the four class metrics, plus delta_<m>_vs_knn_auto with a paired stock bootstrap.
    Returns (rows in ``COLUMNS``, selection counts, {'rerun_mismatches': n differing control predictions vs
    committed or None})."""
    nc, n = len(classes), len(y)
    long_arms = [a for a in mats if a != REF_ARM]
    rows, sels, info = [], [], {"rerun_mismatches": None}
    for scheme, g in (("ticker", None), ("issuer", groups)):
        reps = []
        for r in range(repeats):
            pr = {}
            for arm, m in mats.items():
                sel = []
                pr[arm] = knn_auto_repeat(m, y, nc, r, g, sel)
                sels += [(scheme, arm, mn, k) for mn, k in sel]
            reps.append(pr)
            if verbose:
                print(f"  sector {scheme} repeat {r} done", flush=True)
        ref_name = "committed" if (scheme == "ticker" and committed is not None) else "ctrl"
        if ref_name == "committed":
            info["rerun_mismatches"] = int(sum((reps[r][REF_ARM] != committed[r]).sum() for r in range(repeats)))
        by_rep = [{"knn_auto": committed[r] if ref_name == "committed" else reps[r][REF_ARM],
                   **{f"knn_auto_long|{a}": reps[r][a] for a in long_arms}} for r in range(repeats)]
        models = list(by_rep[0])
        summ = SC.summarize("sector", models, lambda t, q: SC.class_metrics(t, q, nc),
                            lambda t, ps: SC.bootstrap_ci_mean(t, ps, lambda a, b: SC.fast_class_metrics(a, b, nc),
                                                               n_boot=n_boot), y, by_rep, n, nc)
        delt = delta_rows("sector", [m for m in models if m != "knn_auto"], "knn_auto", y, by_rep, n, nc, n_boot=n_boot)
        for r in summ + delt:
            name, _, arm = r["model"].partition("|")
            rows.append(_row("sector", name, arm or ref_name, "cv", r["metric"], r["mean"], r["n_stocks"],
                             ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], folds=scheme, sd=r["sd"], n_repeats=r["n_repeats"],
                             selection="inner_cv"))
    sel = pd.DataFrame(sels, columns=["folds", "arm", "matrix", "k"]).value_counts().rename("count").reset_index()
    return pd.DataFrame(rows, columns=COLUMNS), sel, info


# --------------------------------------------------------------------------- CLI
def _quality_for(data_dir) -> pd.DataFrame:
    return quality_table(load_prices(data_dir), load_info(data_dir))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quality", action="store_true", help="per-year quality table -> history_quality.csv")
    ap.add_argument("--arms", nargs="?", const="auto", default=None,
                    help="comma list of ctrl,long2011 (default: both)")
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--repeats", type=int, default=10, help="sector repeats")
    ap.add_argument("--no-sector", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--ref-dir", type=Path, default=ROOT / "results",
                    help="committed results read for the reproduction checks (metrics.csv, static_cv_oof.parquet)")
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args(argv)
    if not (a.quality or a.arms):
        ap.error("give --quality or --arms")
    dd = a.data_dir
    a.out_dir.mkdir(parents=True, exist_ok=True)
    qt = _quality_for(dd)
    qt.to_csv(a.out_dir / "history_quality.csv", index=False)
    start_year = admitted_start_year(qt)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(qt.to_string(index=False))
    print(f"[quality rule] admitted start year: {start_year} (rule: >= {MIN_STOCKS} stocks, zero-volume <= "
          f"{MAX_ZERO_VOL:.0%}, NaN-volume <= {MAX_NAN_VOL:.0%}, every later year passing)")
    if not a.arms:
        return

    arms = list(ARMS) if a.arms == "auto" else a.arms.split(",")
    bad = [x for x in arms if x not in ARMS]
    if bad:
        ap.error(f"unknown arms {bad}; choose from {list(ARMS)}")
    if "ctrl" not in arms:
        arms = ["ctrl"] + arms
    targets = tuple(a.targets.split(","))
    datasets, mats, tickers = {}, {}, None
    ctrl_panel = F.build_panel(dd)
    datasets["ctrl"] = arm_datasets(ctrl_panel, None, targets)
    sector = not a.no_sector
    if sector:
        tickers, classes, y, _, _, _ = SC.sector_inputs(ctrl_panel)
        mats["ctrl"] = SC.sector_matrices(ctrl_panel, tickers)
        groups = SC.issuer_ids(tickers, None if dd is None else Path(dd) / "universe.csv")
    del ctrl_panel
    for arm in [x for x in arms if x != "ctrl"]:
        panel = build_long_panel(CACHE_START, dd)
        datasets[arm] = arm_datasets(panel, panel.calendar[0], targets)
        if sector:
            mats[arm] = long_sector_matrices(panel, tickers)
        print(f"[{arm}] panel from {panel.calendar[0].date()}: " +
              ", ".join(f"{t} train rows {int((d['split'] == 'train').sum())}" for t, d in datasets[arm].items()),
              flush=True)
        del panel
    out = forward_experiment({k: datasets[k] for k in arms}, targets, verbose=True)
    metrics_csv = a.ref_dir / "metrics.csv"
    if metrics_csv.exists():
        chk = reproduction_check(out, metrics_csv)
        print("[reproduction of ctrl vs metrics.csv linear_fe / lgbm_fe]\n", chk.to_string(index=False))
        print(f"[reproduction] max abs diff {chk['abs_diff'].max():.3e}, n_obs equal: {bool(chk['n_equal'].all())}")
    else:
        print(f"[reproduction] skipped: {metrics_csv} not found")
    if sector:
        committed = None
        oof = a.ref_dir / "static_cv_oof.parquet"
        if oof.exists():
            committed = committed_knn_auto(oof, tickers, classes, a.repeats)
        else:
            print(f"[sector reproduction] {oof} not found: the ticker-fold reference is the control rerun")
        sec, sel, info = sector_experiment(classes, y, mats, groups, a.repeats, committed, verbose=True)
        print(f"[sector reproduction] control rerun vs committed knn_auto OOF: {info['rerun_mismatches']} differing "
              "predictions (0 = identical)")
        print("[sector knn_auto selections (folds, arm, matrix, k, count over outer folds)]\n", sel.to_string(index=False))
        out = pd.concat([out, sec], ignore_index=True)
    out.to_csv(a.out_dir / "history_ext.csv", index=False)
    show = out[(out["metric"] == "delta_r2_vs_ctrl") | out["metric"].str.startswith("delta_macro_f1")]
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        print(show[["target", "model", "features", "split", "metric", "value", "ci_lo", "ci_hi", "folds"]].to_string(index=False))


if __name__ == "__main__":
    main()
