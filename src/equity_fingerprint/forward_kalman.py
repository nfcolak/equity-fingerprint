"""v8 forward experiment for beta_1y: causal Kalman-filter beta features (random-walk state space).

Protocol (selection on validation only, test scored once; same as ``forward_v7``):

1. Dataset: ``forward_v7.build_dataset`` (identical to run.main for the time targets, incl. Welch columns).
2. Kalman beta (``kalman_beta.py``): per stock r_t = alpha + beta_t m_t + e, beta a random walk, filtered (causal),
   initialised from the first 252-day OLS. One noise ratio q = var(w)/var(e) for all stocks, chosen from a grid by
   the summed one-step-ahead predictive log-likelihood on dates <= 2021-12-31 only. Features ``beta_kalman`` and
   its posterior sd ``beta_kalman_sd``, looked up at each observation date.
3. Feature sets: ``fekalman`` = FE_FEATURES + Kalman; ``fewelchkalman`` = FE + Welch + Kalman. References from
   forward_v7: ``linear_fe`` (and ``lgbm_fe``) and ``avg_fewelch``.
4. Models: ``linear_*`` (ridge), ``lgbm_*`` (LightGBM n_jobs=1), ``avg_*`` = w*linear + (1-w)*lgbm with w from
   {0,.25,.5,.75,1} chosen on validation R2.
5. Arm with the highest validation R2 gets ``selected = 1`` (single-feature diagnostics are excluded from
   selection). Test is computed once for all arms.
6. Test delta vs ``linear_fe`` with ticker-block bootstrap (1000 draws, seed 42), 2.5/97.5 percentiles.
7. Diagnostic: single-feature OLS (fit on train) R2 on validation of ``beta_kalman`` alone vs ``beta_252`` alone,
   on the validation rows where both exist (models ``ols_beta_kalman`` / ``ols_beta_252``, features ``single``).

Output ``results/forward_kalman.csv``: target, model, features, split, metric, value, n_obs, ci_lo, ci_hi,
selected. ``kalman_q`` rows (split train, metric ``q_chosen``/``loglik_q=<q>``) document the chosen q and the grid.

Run: env -u PYTHONPATH PYTHONPATH=src python -m equity_fingerprint.forward_kalman
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from . import kalman_beta as K
from .forward_v7 import (REF, ROOT, TARGET, WELCH_COLS, bootstrap_delta_r2, build_dataset, choose_weight,
                         fit_predict, r2, reproduction_check)
from .models import regression_metrics

KAL_COLS = ["beta_kalman", "beta_kalman_sd"]
OUT_COLS = ["target", "model", "features", "split", "metric", "value", "n_obs", "ci_lo", "ci_hi", "selected"]


def add_kalman(d: pd.DataFrame, panel, q: float | None = None):
    """Add beta_kalman / beta_kalman_sd to observations ``d`` (pos, ticker). Returns (d, q, grid scores)."""
    beta, sd, q, scores = K.kalman_beta(panel.ret, panel.mkt, q)
    d = d.copy()
    p = d["pos"].to_numpy()
    c = panel.ret.columns.get_indexer(d["ticker"])
    d["beta_kalman"] = beta.to_numpy()[p, c]
    d["beta_kalman_sd"] = sd.to_numpy()[p, c]
    return d, q, scores


def single_feature_r2(tr, va, col: str, rows_va) -> float:
    """OLS of beta_1y on one feature (fit on train), R2 on the validation rows ``rows_va``."""
    t = tr[[col, TARGET]].dropna()
    slope, icpt = np.polyfit(t[col], t[TARGET], 1)
    v = va.loc[rows_va]
    return r2(v[TARGET], icpt + slope * v[col])


def run_experiment(d: pd.DataFrame, q: float, scores: dict) -> tuple[pd.DataFrame, dict]:
    tr, va, te = (d[d["split"] == s] for s in ("train", "val", "test"))
    sets = {"fe": F.FE_FEATURES, "fewelch": F.FE_FEATURES + WELCH_COLS,
            "fekalman": F.FE_FEATURES + KAL_COLS, "fewelchkalman": F.FE_FEATURES + WELCH_COLS + KAL_COLS}
    preds, weights = {}, {}
    for feats, cols in sets.items():
        fp = fit_predict(tr, va, te, cols)
        w = choose_weight(va[TARGET], fp["linear"][0], fp["lgbm"][0])  # validation only
        weights[feats] = w
        avg = tuple(w * fp["linear"][i] + (1 - w) * fp["lgbm"][i] for i in (0, 1))
        if feats in ("fe", "fewelch"):  # references only: linear_fe, lgbm_fe, avg_fewelch
            keep = {"fe": ("linear", "lgbm"), "fewelch": ()}[feats]
        else:
            keep = ("linear", "lgbm")
        for kind in keep:
            preds[f"{kind}_{feats}"] = fp[kind]
        if feats != "fe":
            preds[f"avg_{feats}"] = avg
    feat_of = lambda m: m.split("_", 1)[1]  # noqa: E731
    val_r2 = {m: r2(va[TARGET], p[0]) for m, p in preds.items()}
    selected = max(val_r2, key=val_r2.get)
    ref_test = preds[REF][1]
    rows = []
    for m, (pv, pt) in preds.items():
        base = dict(target=TARGET, model=m, features=feat_of(m), ci_lo=np.nan, ci_hi=np.nan,
                    selected=int(m == selected))
        for split, part, p in (("val", va, pv), ("test", te, pt)):
            for k, v in regression_metrics(part[TARGET], p, False).items():
                rows.append({**base, "split": split, "metric": k, "value": float(v), "n_obs": len(part)})
        if m.startswith("avg_"):
            rows.append({**base, "split": "val", "metric": "weight_linear", "value": weights[feat_of(m)],
                         "n_obs": len(va)})
        if m != REF:
            lo, hi = bootstrap_delta_r2(te[TARGET], pt, ref_test, te["ticker"])
            rows.append({**base, "split": "test", "metric": "delta_r2_vs_linear_fe",
                         "value": r2(te[TARGET], pt) - r2(te[TARGET], ref_test), "n_obs": len(te),
                         "ci_lo": lo, "ci_hi": hi})
    # diagnostics: single-feature OLS on common validation rows
    common = va.index[va["beta_kalman"].notna() & va["beta_252"].notna()]
    for col in ("beta_kalman", "beta_252"):
        rows.append(dict(target=TARGET, model=f"ols_{col}", features="single", split="val", metric="r2",
                         value=single_feature_r2(tr, va, col, common), n_obs=len(common), ci_lo=np.nan,
                         ci_hi=np.nan, selected=0))
    # q documentation
    base = dict(target=TARGET, model="kalman_q", features="kalman", split="train", n_obs=0, ci_lo=np.nan,
                ci_hi=np.nan, selected=0)
    rows.append({**base, "metric": "q_chosen", "value": q})
    for g, ll in scores.items():
        rows.append({**base, "metric": f"loglik_q={g:g}", "value": ll})
    return pd.DataFrame(rows)[OUT_COLS], {"selected": selected, "weights": weights}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    a = ap.parse_args(argv)
    t0 = time.time()
    panel = F.build_panel()
    d = build_dataset(panel)
    d, q, scores = add_kalman(d, panel)
    print("[q grid loglik]", {f"{k:g}": round(v, 1) for k, v in scores.items()}, "[q chosen]", q)
    out, info = run_experiment(d, q, scores)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out_dir / "forward_kalman.csv", index=False)
    print("[reproduction vs results/metrics.csv]\n",
          reproduction_check(out, ROOT / "results" / "metrics.csv").to_string(index=False))
    print("[weights]", info["weights"], "[selected on val]", info["selected"])
    piv = out[out["metric"].isin(["r2", "delta_r2_vs_linear_fe"])].pivot_table(
        index="model", columns=["split", "metric"], values="value")
    print(piv.round(4).to_string())
    print(out[out["metric"] == "delta_r2_vs_linear_fe"][["model", "value", "ci_lo", "ci_hi"]].round(4).to_string(index=False))
    print(f"[runtime] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
