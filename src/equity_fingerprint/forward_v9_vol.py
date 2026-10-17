"""v9 forward experiment for vol_1m: HExp / seasonal / month / VIX features, fe3 and a validation-fitted stack.

Protocol (all selection on validation only, test scored once per arm):

1. Rows exactly as ``forward_minirocket`` for vol_1m (``build_time_frame`` + ``target_frame(..., "vol_1m")``;
   windows with zero realised vol are dropped), so val/test ``n_obs`` equal those of results/metrics.csv.
2. New causal features (annualisation 252, vol floor 0.005 before every log; copied from the v9 probe):
   * ``HEXP``  hexp1, hexp5, hexp21, hexp63: log sqrt(252 * EWMA(r^2, halflife h, min_periods h)).
   * ``SEAS``  seas1, seas2: log realised vol of the 21-day window that ends 252y - 21 rows before the feature
     date (the forward window shifted back y years); seas1_rel, seas2_rel: minus the 252-day log RV at the same
     shift.
   * ``MONTH`` mon2..mon12: feature-date month dummies (January is the base).
   * ``VIX``   lvix (log VIX close, forward-filled on the Swiss calendar), dvix21 (its 21-row change),
     lstoxx21 (log annualised 21-day RV of STOXX50E log returns). A US close on date t is day t's data; the
     target window starts at t + 1.
3. Arms in this order: ``linear_fe``, ``lgbm_fe`` (reproduce results/metrics.csv), ``linear_fehexp``,
   ``linear_feseas``, ``linear_femonth``, ``linear_fevix`` (one group each), ``linear_feall``, ``lgbm_feall``,
   ``avg_feall`` (w * linear + (1 - w) * lgbm, w on validation), ``linear_fe3`` (FE + HAR/GARCH/market columns
   of ``volmodels.vol_features`` as in run.py), ``ridge_mrfe`` (``forward_minirocket.run_target(preds_out=...)``,
   aligned by row index) and ``stack``: non-negative weights summing to 1 over (linear_feall, lgbm_feall,
   ridge_mrfe, linear_fe3), least squares on the validation predictions (NNLS on a system augmented with a heavily
   weighted sum-to-one row, then renormalised; deterministic). Linear arms: Prep + ridge (alpha on validation
   MSE); lgbm: forward_v7 grid with early stopping on validation, n_jobs=1.
4. ``selected = 1`` on all rows of the arm with the highest validation R2 (first on ties, arms in the order
   above). Test deltas ``delta_r2_vs_linear_fe`` and ``delta_r2_vs_ridge_mrfe`` with ticker-block bootstrap
   intervals (1000 draws, seed 42).

Output ``results/forward_v9_vol.csv``: target, model, features, split, metric, value, n_obs, ci_lo, ci_hi,
selected. Val rows ``weight_linear`` (avg_feall) and ``weight_<arm>`` (stack) hold the chosen weights.

Run: env -u PYTHONPATH PYTHONPATH=src python -m equity_fingerprint.forward_v9_vol
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from . import features as F
from . import forward_minirocket as M
from .extra_market import load_extra
from .forward_v7 import KEYS, ROOT, _lgbm, bootstrap_delta_r2, choose_weight, r2
from .models import Prep, fit_ridge, regression_metrics
from .volmodels import GARCH_COLS, HAR_COLS, MKT_COLS, vol_features

TARGET = "vol_1m"
ANNF = 252
FLOOR = 0.005
HALFLIVES = (1, 5, 21, 63)
HEXP = [f"hexp{h}" for h in HALFLIVES]
SEAS = ["seas1", "seas2", "seas1_rel", "seas2_rel"]
MONTH = [f"mon{k}" for k in range(2, 13)]
VIX = ["lvix", "dvix21", "lstoxx21"]
NEW_COLS = HEXP + SEAS + MONTH + VIX
FE3 = F.FE_FEATURES + HAR_COLS + GARCH_COLS + MKT_COLS
GROUPS = {"hexp": HEXP, "seas": SEAS, "month": MONTH, "vix": VIX}
STACK_ARMS = ("linear_feall", "lgbm_feall", "ridge_mrfe", "linear_fe3")
REFS = ("linear_fe", "ridge_mrfe")
COLS = M.COLS
STACK_LAMBDA = 1e4


# --------------------------------------------------------------------------- features
def _log_floor(x):
    return np.log(x.clip(lower=FLOOR))


def _lrv(r2_: pd.DataFrame, w: int, shift: int = 0) -> pd.DataFrame:
    """Log annualised realised vol over w rows, then shifted ``shift`` rows into the past (shift >= 0)."""
    return _log_floor(np.sqrt(r2_.rolling(w, min_periods=w).mean() * ANNF)).shift(shift)


def extra_series(extra: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """VIX / STOXX50E closes forward-filled onto the Swiss trading calendar (row t uses closes up to day t)."""
    missing = [t for t in ("^VIX", "^STOXX50E") if t not in extra.columns]
    if missing:
        raise KeyError(f"extra frame lacks {missing}")
    ex = extra.sort_index()
    ex = ex[~ex.index.duplicated()]
    return ex.reindex(calendar.union(ex.index)).ffill().reindex(calendar)


def new_features(panel, obs: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """HEXP, SEAS, MONTH and VIX columns for the rows of ``obs`` (columns pos, ticker, date); index of ``obs``."""
    R2 = panel.ret ** 2
    p = obs["pos"].to_numpy()
    c = panel.ret.columns.get_indexer(obs["ticker"])
    take = lambda df: df.to_numpy()[p, c]  # noqa: E731
    out = {}
    for h in HALFLIVES:
        e = R2.ewm(halflife=h, min_periods=h).mean()
        out[f"hexp{h}"] = take(_log_floor(np.sqrt(e * ANNF)))
    for y in (1, 2):
        sh = ANNF * y - 21  # the 21-day window ending sh rows before the feature date
        out[f"seas{y}"] = take(_lrv(R2, 21, sh))
        out[f"seas{y}_rel"] = take(_lrv(R2, 21, sh) - _lrv(R2, 252, sh))
    month = pd.DatetimeIndex(obs["date"]).month.to_numpy()
    for k in range(2, 13):
        out[f"mon{k}"] = (month == k).astype(float)
    ex = extra_series(extra, panel.calendar)
    lvix = np.log(ex["^VIX"])
    out["lvix"] = lvix.to_numpy()[p]
    out["dvix21"] = (lvix - lvix.shift(21)).to_numpy()[p]
    sr = np.log(ex["^STOXX50E"]).diff()
    out["lstoxx21"] = _log_floor(np.sqrt((sr ** 2).rolling(21).mean() * ANNF)).to_numpy()[p]
    return pd.DataFrame(out, index=obs.index)[NEW_COLS]


# --------------------------------------------------------------------------- dataset
def build_dataset(panel, time_df: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """vol_1m rows of forward_minirocket (index reset) + new feature columns + the fe3 vol columns."""
    d = M.target_frame(time_df, TARGET).reset_index(drop=True)
    d = pd.concat([d, new_features(panel, d, extra)], axis=1)
    vf = vol_features(panel, d[KEYS])
    for c in HAR_COLS + GARCH_COLS + MKT_COLS:
        d[c] = vf[c]
    return d


# --------------------------------------------------------------------------- models
def fit_linear(tr, va, te, cols) -> tuple[np.ndarray, np.ndarray]:
    pl = Prep(cols, linear=True).fit(tr)
    Xtr, Xva, Xte = (pl.transform(x) for x in (tr, va, te))
    m, _ = fit_ridge(Xtr, tr[TARGET], Xva, va[TARGET])
    return m.predict(Xva), m.predict(Xte)


def fit_lgbm(tr, va, te, cols) -> tuple[np.ndarray, np.ndarray]:
    pt = Prep(cols, linear=False).fit(tr)
    Xtr, Xva, Xte = (pt.transform(x) for x in (tr, va, te))
    m = _lgbm(Xtr, tr[TARGET], Xva, va[TARGET])
    return m.predict(Xva), m.predict(Xte)


def stack_weights(y_val, P_val: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to 1 minimising ||P_val w - y_val||^2 (validation only, deterministic)."""
    P = np.asarray(P_val, float)
    y = np.asarray(y_val, float)
    k = P.shape[1]
    lam = STACK_LAMBDA * max(1.0, float(np.abs(P).max()))
    A = np.vstack([P, lam * np.ones((1, k))])
    b = np.concatenate([y, [lam]])
    w, _ = nnls(A, b, maxiter=50 * k)
    s = w.sum()
    return w / s if s > 0 else np.full(k, 1.0 / k)


def _align(mr, ref_index_va, ref_index_te) -> tuple[np.ndarray, np.ndarray]:
    pv, pt, iv, it = mr
    a = pd.Series(np.asarray(pv), index=iv).reindex(ref_index_va)
    b = pd.Series(np.asarray(pt), index=it).reindex(ref_index_te)
    if a.isna().any() or b.isna().any():
        raise ValueError("ridge_mrfe predictions do not cover the rows of the v9 dataset")
    return a.to_numpy(), b.to_numpy()


# --------------------------------------------------------------------------- experiment
def run_arms(d: pd.DataFrame, mr_preds: dict) -> tuple[pd.DataFrame, dict]:
    """All arms on the prepared dataset ``d``; ``mr_preds`` = ``preds_out`` of ``forward_minirocket.run_target``."""
    tr, va, te = (d[d["split"] == s] for s in ("train", "val", "test"))
    yva, yte = va[TARGET].to_numpy(), te[TARGET].to_numpy()
    preds, feats, extra_rows = {}, {}, {}

    def add(name, f, pv_pt):
        preds[name], feats[name] = pv_pt, f

    add("linear_fe", "fe", fit_linear(tr, va, te, F.FE_FEATURES))
    add("lgbm_fe", "fe", fit_lgbm(tr, va, te, F.FE_FEATURES))
    for g, cols in GROUPS.items():
        add(f"linear_fe{g}", f"fe{g}", fit_linear(tr, va, te, F.FE_FEATURES + cols))
    allc = F.FE_FEATURES + NEW_COLS
    add("linear_feall", "feall", fit_linear(tr, va, te, allc))
    add("lgbm_feall", "feall", fit_lgbm(tr, va, te, allc))
    w = choose_weight(yva, preds["linear_feall"][0], preds["lgbm_feall"][0])  # validation only
    add("avg_feall", "feall", tuple(w * preds["linear_feall"][i] + (1 - w) * preds["lgbm_feall"][i] for i in (0, 1)))
    extra_rows["avg_feall"] = {"weight_linear": w}
    add("linear_fe3", "fe3", fit_linear(tr, va, te, FE3))
    add("ridge_mrfe", "mrfe", _align(mr_preds["ridge_mrfe"], va.index, te.index))
    Pva = np.column_stack([preds[a][0] for a in STACK_ARMS])
    Pte = np.column_stack([preds[a][1] for a in STACK_ARMS])
    sw = stack_weights(yva, Pva)  # validation labels only
    add("stack", "stack", (Pva @ sw, Pte @ sw))
    extra_rows["stack"] = {f"weight_{a}": float(v) for a, v in zip(STACK_ARMS, sw)}

    val_r2 = {m: r2(yva, p[0]) for m, p in preds.items()}
    selected = max(val_r2, key=val_r2.get)  # first on ties, arms in documented order
    rows = []
    for m, (pv, pt) in preds.items():
        base = dict(target=TARGET, model=m, features=feats[m], ci_lo=np.nan, ci_hi=np.nan,
                    selected=int(m == selected))
        rv = float(np.var(yva - pv))
        for split, y, p in (("val", yva, pv), ("test", yte, pt)):
            for k, v in regression_metrics(y, p, True, rv).items():
                rows.append({**base, "split": split, "metric": k, "value": float(v), "n_obs": len(y)})
        for k, v in extra_rows.get(m, {}).items():
            rows.append({**base, "split": "val", "metric": k, "value": float(v), "n_obs": len(yva)})
        for ref in REFS:
            if m == ref:
                continue
            lo, hi = bootstrap_delta_r2(yte, pt, preds[ref][1], te["ticker"])
            rows.append({**base, "split": "test", "metric": f"delta_r2_vs_{ref}",
                         "value": r2(yte, pt) - r2(yte, preds[ref][1]), "n_obs": len(yte), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)[COLS], {"selected": selected, "avg_weight": w,
                                     "stack_weights": dict(zip(STACK_ARMS, sw.tolist()))}


def run_experiment(panel, time_df: pd.DataFrame, extra: pd.DataFrame, log=print) -> tuple[pd.DataFrame, dict]:
    d = build_dataset(panel, time_df, extra)
    mr = {}
    M.run_target(panel, time_df, TARGET, log=log, preds_out=mr)
    return run_arms(d, mr)


def reproduction_check(out: pd.DataFrame, metrics_csv: Path) -> pd.DataFrame:
    """linear_fe / lgbm_fe rows vs the vol_1m rows of results/metrics.csv."""
    ref = pd.read_csv(metrics_csv)
    ms = ["r2", "mae", "rmse", "qlike"]
    ref = ref[(ref["target"] == TARGET) & ref["model"].isin(["linear_fe", "lgbm_fe"]) & ref["metric"].isin(ms)]
    new = out[out["model"].isin(["linear_fe", "lgbm_fe"]) & out["metric"].isin(ms)]
    k = ["model", "split", "metric"]
    mg = ref.merge(new, on=k, suffixes=("_metrics", "_v9"))
    mg["abs_diff"] = (mg["value_metrics"] - mg["value_v9"]).abs()
    mg["n_equal"] = mg["n_obs_metrics"] == mg["n_obs_v9"]
    return mg[k + ["value_metrics", "value_v9", "abs_diff", "n_obs_metrics", "n_obs_v9", "n_equal"]]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    a = ap.parse_args(argv)
    extra = load_extra()
    panel = F.build_panel()
    time_df = M.build_time_frame(panel)
    out, info = run_experiment(panel, time_df, extra)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out_dir / "forward_v9_vol.csv", index=False)
    chk = reproduction_check(out, ROOT / "results" / "metrics.csv")
    print("[reproduction vs results/metrics.csv]\n", chk.to_string(index=False))
    print(f"[reproduction] max abs diff {chk['abs_diff'].max():.3g} (tolerance 1e-9), "
          f"n_obs equal: {bool(chk['n_equal'].all())}")
    print("[avg weight]", info["avg_weight"], "[stack weights]", info["stack_weights"],
          "[selected on val]", info["selected"])
    piv = out[out["metric"].isin(["r2", "delta_r2_vs_linear_fe", "delta_r2_vs_ridge_mrfe"])].pivot_table(
        index="model", columns=["split", "metric"], values="value")
    print(piv.round(4).to_string())


if __name__ == "__main__":
    main()
