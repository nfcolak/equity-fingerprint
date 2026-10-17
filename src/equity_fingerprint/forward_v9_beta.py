"""v9 forward experiment for beta_1y: residual target (predict the change from a shrunk past beta).

Protocol (selection on validation only, test scored once; same machinery as ``forward_v7``):

1. Dataset: ``forward_v7.build_dataset`` plus causal daily-panel columns (all use data up to the feature date):
   ``se_beta_252`` (OLS standard error of the 252-day beta), ``beta_vasicek`` (Vasicek shrinkage of
   ``beta_welch252`` toward the same-date cross-sectional mean over all stocks with a value that day; weight on
   the own estimate = cs_var / (cs_var + se^2)), ``beta_vasicek_size`` (same, prior = same-date mean within the
   stock's tercile of log1p(252-day mean traded value, min_periods 126)) and the HAR bundle
   ``HAR_B = [beta_126, beta_504, beta_756]`` (rolling OLS betas).
2. Bases b (NaN falls back along the chain, then to the train mean of beta_1y):
   ols = beta_252; w252 = beta_welch252 -> beta_252; wew = beta_welch_ew -> beta_welch252 -> beta_252;
   vck = beta_vasicek -> wew; mix = 0.5 wew + 0.5 (beta_vasicek_size -> wew).
3. Arms in this order: controls ``linear/lgbm/avg_fe`` and ``..._fewelch`` (must equal results/forward_v7.csv);
   residual arms ``linear/lgbm/avg_res<b>`` for b in (ols, w252, wew, vck, mix): target beta_1y - base, features
   FE_FEATURES + WELCH_COLS (+ beta_vasicek, beta_vasicek_size for vck / mix), prediction = base + model output;
   ``linear/lgbm/avg_resmixhar``: base mix with the resmix features + HAR_B.
4. ``avg_*`` = w * linear + (1 - w) * lgbm, w in {0, .25, .5, .75, 1} chosen on validation R2 (first on ties).
5. The arm with the highest validation R2 gets ``selected = 1`` on all its rows (first on ties, order above).
   Test is computed once per arm and never used for a choice.
6. Test deltas of every arm vs ``linear_fe`` and vs ``avg_fewelch`` (self-comparison skipped) with the ticker-block
   bootstrap (1000 draws, seed 42), 2.5/97.5 percentiles.

Output ``results/forward_v9_beta.csv``: target, model, features, split, metric, value, n_obs, ci_lo, ci_hi,
selected (as forward_v7.csv). ``weight_linear`` (split val) holds w of the avg arms.

Run: env -u PYTHONPATH PYTHONPATH=src python -m equity_fingerprint.forward_v9_beta
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from . import forward_v7 as V7
from .forward_v7 import ROOT, TARGET, WELCH_COLS, bootstrap_delta_r2, choose_weight, fit_predict, r2, welch_beta
from .models import regression_metrics

HAR_B = ["beta_126", "beta_504", "beta_756"]
VCK_COLS = ["beta_vasicek", "beta_vasicek_size"]
NEW_COLS = ["se_beta_252", *VCK_COLS, *HAR_B]
BASES = ("ols", "w252", "wew", "vck", "mix")
REFS = ("linear_fe", "avg_fewelch")
OUT_COLS = ["target", "model", "features", "split", "metric", "value", "n_obs", "ci_lo", "ci_hi", "selected"]


# --------------------------------------------------------------------------- causal beta columns
def beta_panel_cols(panel) -> dict[str, pd.DataFrame]:
    """Daily (date x ticker) frames of the new columns. Row t uses data up to and including t only."""
    r, m = panel.ret, panel.mkt
    mv = lambda w: m.rolling(w, min_periods=w).var()  # noqa: E731
    ols = lambda w: r.rolling(w, min_periods=w).cov(m).div(mv(w), axis=0)  # noqa: E731
    out = {f"beta_{w}": ols(w) for w in (126, 504, 756)}
    b252 = ols(252)
    vm = mv(252).to_numpy()[:, None]
    res_var = (r.rolling(252, min_periods=252).var() - b252 ** 2 * vm).clip(lower=1e-10)
    se = np.sqrt(res_var / (252 * vm))
    out["se_beta_252"] = se
    bw = welch_beta(r, m, 252)
    cs_mean, cs_var = bw.mean(axis=1).to_numpy()[:, None], bw.var(axis=1).to_numpy()[:, None]
    wt = cs_var / (cs_var + se ** 2)  # weight on the own estimate
    out["beta_vasicek"] = wt * bw + (1 - wt) * cs_mean
    # size prior: terciles of the 252-day mean traded value at each date (market cap history is not free)
    rk = np.log1p(panel.tv.rolling(252, min_periods=126).mean()).rank(axis=1, pct=True)
    grp = pd.DataFrame(np.select([rk < 1 / 3, rk < 2 / 3], [0, 1], 2).astype(float), index=rk.index,
                       columns=rk.columns).where(rk.notna())
    gm = pd.DataFrame(index=bw.index, columns=bw.columns, dtype=float)
    for g in (0, 1, 2):
        msk = grp == g
        gm = gm.where(~msk, bw.where(msk).mean(axis=1).to_numpy()[:, None] + 0 * bw)
    out["beta_vasicek_size"] = wt * bw + (1 - wt) * gm
    return out


def add_beta_cols(d: pd.DataFrame, panel) -> pd.DataFrame:
    """Add the new columns to the observations ``d`` (columns pos, ticker) by lookup in the daily panel."""
    d = d.copy()
    p = d["pos"].to_numpy()
    c = panel.ret.columns.get_indexer(d["ticker"])
    for k, v in beta_panel_cols(panel).items():
        d[k] = v.to_numpy()[p, c]
    return d


def build_dataset(panel=None) -> pd.DataFrame:
    """forward_v7 dataset plus the v9 beta columns."""
    panel = panel or F.build_panel()
    return add_beta_cols(V7.build_dataset(panel), panel)


# --------------------------------------------------------------------------- bases
def make_bases(d: pd.DataFrame, train_mean: float) -> dict[str, np.ndarray]:
    """Base betas for every row of ``d`` (fallback chains, then the train mean of beta_1y)."""
    wew = d["beta_welch_ew"].fillna(d["beta_welch252"]).fillna(d["beta_252"])
    mix = 0.5 * wew + 0.5 * d["beta_vasicek_size"].fillna(wew)
    b = {"ols": d["beta_252"], "w252": d["beta_welch252"].fillna(d["beta_252"]), "wew": wew,
         "vck": d["beta_vasicek"].fillna(wew), "mix": mix}
    return {k: v.fillna(train_mean).to_numpy(float) for k, v in b.items()}


def res_features(base: str) -> list[str]:
    cols = F.FE_FEATURES + WELCH_COLS
    return cols + VCK_COLS if base in ("vck", "mix") else cols


def fit_residual(tr, va, te, cols, base_tr, base_va, base_te) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """fit_predict on y - base (train/val); predictions = base + model output."""
    tr, va = tr.copy(), va.copy()
    tr["y_res"], va["y_res"] = tr[TARGET] - base_tr, va[TARGET] - base_va
    out = fit_predict(tr, va, te, cols, y_col="y_res")
    return {k: (pv + base_va, pt + base_te) for k, (pv, pt) in out.items()}


# --------------------------------------------------------------------------- experiment
def run_experiment(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    tr, va, te = (d[d["split"] == s] for s in ("train", "val", "test"))
    mean_tr = float(tr[TARGET].mean())
    bs = {s: make_bases(part, mean_tr) for s, part in (("tr", tr), ("va", va), ("te", te))}
    specs = [("fe", F.FE_FEATURES, None), ("fewelch", F.FE_FEATURES + WELCH_COLS, None)]
    specs += [(f"res{b}", res_features(b), b) for b in BASES]
    specs.append(("resmixhar", res_features("mix") + HAR_B, "mix"))
    preds, weights, feat_of = {}, {}, {}
    for name, cols, base in specs:
        if base is None:
            fp = fit_predict(tr, va, te, cols)
        else:
            fp = fit_residual(tr, va, te, cols, bs["tr"][base], bs["va"][base], bs["te"][base])
        w = choose_weight(va[TARGET], fp["linear"][0], fp["lgbm"][0])  # validation only
        weights[name] = w
        for kind in ("linear", "lgbm"):
            preds[f"{kind}_{name}"] = fp[kind]
        preds[f"avg_{name}"] = tuple(w * fp["linear"][i] + (1 - w) * fp["lgbm"][i] for i in (0, 1))
        for kind in ("linear", "lgbm", "avg"):
            feat_of[f"{kind}_{name}"] = name
    val_r2 = {m: r2(va[TARGET], p[0]) for m, p in preds.items()}
    selected = max(val_r2, key=val_r2.get)  # first on ties, arms in the documented order
    rows = []
    for m, (pv, pt) in preds.items():
        base = dict(target=TARGET, model=m, features=feat_of[m], ci_lo=np.nan, ci_hi=np.nan,
                    selected=int(m == selected))
        for split, part, p in (("val", va, pv), ("test", te, pt)):
            for k, v in regression_metrics(part[TARGET], p, False).items():
                rows.append({**base, "split": split, "metric": k, "value": float(v), "n_obs": len(part)})
        if m.startswith("avg_"):
            rows.append({**base, "split": "val", "metric": "weight_linear", "value": weights[feat_of[m]],
                         "n_obs": len(va)})
        for ref in REFS:
            if m == ref:
                continue
            lo, hi = bootstrap_delta_r2(te[TARGET], pt, preds[ref][1], te["ticker"])
            rows.append({**base, "split": "test", "metric": f"delta_r2_vs_{ref}",
                         "value": r2(te[TARGET], pt) - r2(te[TARGET], preds[ref][1]), "n_obs": len(te),
                         "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)[OUT_COLS], {"selected": selected, "weights": weights}


def reproduction_check(out: pd.DataFrame, v7_csv: Path, models=REFS) -> pd.DataFrame:
    """Compare r2/mae rows of ``models`` with results/forward_v7.csv (agreement expected to 1e-9)."""
    ref = pd.read_csv(v7_csv)
    keep = lambda x: x[x["model"].isin(models) & x["metric"].isin(["r2", "mae"])]  # noqa: E731
    k = ["model", "split", "metric"]
    mg = keep(ref).merge(keep(out), on=k, suffixes=("_v7", "_v9"))
    mg["abs_diff"] = (mg["value_v7"] - mg["value_v9"]).abs()
    mg["ok"] = (mg["abs_diff"] < 1e-9) & (mg["n_obs_v7"] == mg["n_obs_v9"])
    return mg[k + ["value_v7", "value_v9", "abs_diff", "n_obs_v7", "n_obs_v9", "ok"]]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    a = ap.parse_args(argv)
    d = build_dataset()
    out, info = run_experiment(d)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out_dir / "forward_v9_beta.csv", index=False)
    chk = reproduction_check(out, ROOT / "results" / "forward_v7.csv")
    print("[reproduction vs results/forward_v7.csv]\n", chk.to_string(index=False))
    print("[all agree to 1e-9]", bool(chk["ok"].all()), "[weights]", info["weights"],
          "[selected on val]", info["selected"])
    piv = out[out["metric"] == "r2"].pivot_table(index="model", columns="split", values="value", sort=False)
    print(piv.round(4).to_string())
    dl = out[out["metric"].str.startswith("delta_r2_vs_")]
    print(dl[["model", "metric", "value", "ci_lo", "ci_hi"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
