"""v7 forward experiment for beta_1y: Welch slope-winsorised beta features and linear/lgbm averaging.

Protocol (all selection on validation only, test scored once at the end):

1. Dataset: built exactly as ``run.main`` does for the time targets (build_panel, month_end_positions,
   base_features, pc_loadings(pc_factors(panel, all tickers)), forward_targets, time_split; no fingerprint
   embeddings). Same rows, so val/test ``n_obs`` equal those of results/metrics.csv.
2. Welch (2022) slope-winsorised beta: each stock's daily return is clipped to
   [min(-2m, 4m), max(-2m, 4m)] with m the same day's market return (``panel.mkt``), then
   ``beta_welch252`` = rolling 252-day cov(clipped r, m) / var(m) (min_periods=252) and
   ``beta_welch_ew`` = the same clipped returns with exponentially weighted moments (halflife 126, 756-day
   minimum history). Both use data up to and including the feature date only (causal).
3. Feature sets: ``fe`` = FE_FEATURES (reference), ``fewelch`` = FE_FEATURES + both Welch columns.
4. Models, same preprocessing and selection as ``models.run_target``: ``linear_*`` (Prep + ridge, alpha on
   validation MSE) and ``lgbm_*`` (Prep + LightGBM, grid and early stopping on validation; n_jobs=1).
   ``linear_fe`` / ``lgbm_fe`` must reproduce the beta_1y rows of results/metrics.csv (checked on the CLI).
5. ``avg_<set>`` = w * linear + (1 - w) * lgbm with w in {0, .25, .5, .75, 1} chosen on validation R2
   (first maximum on ties). Done for ``fe`` (control) and ``fewelch``.
6. The arm with the highest validation R2 is marked ``selected = 1``. Test is computed once for every arm.
7. Test delta of each arm vs ``linear_fe`` (the best v1-v3 beta model on validation): ticker-block bootstrap
   (ticker clusters resampled, 1000 draws, seed 42) of the R2 difference, 2.5/97.5 percentiles.

Output ``results/forward_v7.csv``: target, model, features, split, metric, value, n_obs, ci_lo, ci_hi, selected.
``ci_lo/ci_hi`` are filled only on the test rows with metric ``delta_r2_vs_linear_fe`` (empty elsewhere);
metric ``weight_linear`` (split val) holds the chosen w of the avg arms; ``selected`` is 1 on all rows of the
arm chosen on validation, else 0.

Run: env -u PYTHONPATH PYTHONPATH=src python -m equity_fingerprint.forward_v7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import features as F
from . import targets as T
from .models import LGBM_GRID, Prep, _lgbm_params, fit_ridge, regression_metrics

TARGET = "beta_1y"
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
WELCH_COLS = ["beta_welch252", "beta_welch_ew"]
BOOT_N, BOOT_SEED = 1000, 42
REF = "linear_fe"
ROOT = Path(__file__).resolve().parents[2]
KEYS = ["pos", "ticker"]


# --------------------------------------------------------------------------- Welch beta
def welch_clip(ret: pd.DataFrame, mkt: pd.Series, lo: float = -2.0, hi: float = 4.0) -> pd.DataFrame:
    """Clip each stock's return to [min(lo*m, hi*m), max(lo*m, hi*m)], m = same-day market return."""
    a, b = lo * mkt, hi * mkt
    return ret.clip(lower=np.minimum(a, b), upper=np.maximum(a, b), axis=0)


def welch_beta(ret: pd.DataFrame, mkt: pd.Series, window: int = 252, halflife: float | None = None,
               lo: float = -2.0, hi: float = 4.0) -> pd.DataFrame:
    """Slope-winsorised beta. Rolling ``window``-day OLS slope on clipped returns, or (``halflife`` given)
    exponentially weighted moments with ``window`` days of minimum history. Row t uses data up to t only."""
    rw = welch_clip(ret, mkt, lo, hi)
    if halflife is None:
        cov = rw.rolling(window, min_periods=window).cov(mkt)
        return cov.div(mkt.rolling(window, min_periods=window).var(), axis=0)

    def ew(x):
        return x.ewm(halflife=halflife, min_periods=window).mean()

    m1 = ew(mkt.to_frame()).iloc[:, 0]
    m2 = ew((mkt ** 2).to_frame()).iloc[:, 0]
    cov = ew(rw.mul(mkt, axis=0)) - ew(rw).mul(m1, axis=0)
    return cov.div(m2 - m1 ** 2, axis=0)


def add_welch(d: pd.DataFrame, panel) -> pd.DataFrame:
    """Add the Welch columns to the observations ``d`` (columns pos, ticker) by lookup in the daily panel."""
    d = d.copy()
    p = d["pos"].to_numpy()
    c = panel.ret.columns.get_indexer(d["ticker"])
    d["beta_welch252"] = welch_beta(panel.ret, panel.mkt, 252).to_numpy()[p, c]
    d["beta_welch_ew"] = welch_beta(panel.ret, panel.mkt, 756, halflife=126).to_numpy()[p, c]
    return d


# --------------------------------------------------------------------------- dataset
def build_dataset(panel=None) -> pd.DataFrame:
    """beta_1y benchmark dataset as in run.main (no fingerprint embeddings), plus the Welch columns."""
    panel = panel or F.build_panel()
    pos = F.month_end_positions(panel)
    base = F.base_features(panel, pos)
    tickers = sorted(base["ticker"].unique())
    time_df = base.merge(F.pc_loadings(panel, F.pc_factors(panel, tickers), pos), on=KEYS, how="left")
    time_df = pd.concat([time_df, T.forward_targets(panel, time_df)], axis=1)
    d = time_df.copy()
    d["split"] = T.time_split(d["date"], d[f"end_{TARGET}"])
    d = d[d[TARGET].notna() & d["split"].notna()]
    return add_welch(d, panel)


# --------------------------------------------------------------------------- models
def _lgbm(Xtr, ytr, Xva, yva):
    """Same grid / early stopping / selection as models.fit_lgbm, but n_jobs=1 (libomp deadlock with torch)."""
    best = None
    for g in LGBM_GRID:
        m = lgb.LGBMRegressor(**{**_lgbm_params(g), "n_jobs": 1})
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
        score = next(iter(m.best_score_["valid_0"].values()))
        if best is None or score < best[0]:
            best = (score, m)
    return best[1]


def fit_predict(tr, va, te, cols, y_col=TARGET) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{'linear': (val_pred, test_pred), 'lgbm': (...)} fitted on train, hyper-parameters on validation."""
    out = {}
    pl = Prep(cols, linear=True).fit(tr)
    Xtr, Xva, Xte = (pl.transform(d) for d in (tr, va, te))
    m, _ = fit_ridge(Xtr, tr[y_col], Xva, va[y_col])
    out["linear"] = (m.predict(Xva), m.predict(Xte))
    pt = Prep(cols, linear=False).fit(tr)
    Xtr, Xva, Xte = (pt.transform(d) for d in (tr, va, te))
    m = _lgbm(Xtr, tr[y_col], Xva, va[y_col])
    out["lgbm"] = (m.predict(Xva), m.predict(Xte))
    return out


def r2(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def choose_weight(y_val, lin_val, gbm_val, weights=WEIGHTS) -> float:
    """Linear weight with the highest R2 on validation (first on ties). Uses validation data only."""
    scores = [r2(y_val, w * np.asarray(lin_val) + (1 - w) * np.asarray(gbm_val)) for w in weights]
    return float(weights[int(np.argmax(scores))])


def bootstrap_delta_r2(y, pa, pb, groups, n: int = BOOT_N, seed: int = BOOT_SEED) -> tuple[float, float]:
    """95% interval of R2(pa) - R2(pb) under a block bootstrap that resamples whole groups (tickers)."""
    y, pa, pb = (np.asarray(v, float) for v in (y, pa, pb))
    u, inv = np.unique(np.asarray(groups), return_inverse=True)
    k = len(u)
    cnt = np.bincount(inv, minlength=k).astype(float)
    s1 = np.bincount(inv, weights=y, minlength=k)
    s2 = np.bincount(inv, weights=y ** 2, minlength=k)
    ea = np.bincount(inv, weights=(y - pa) ** 2, minlength=k)
    eb = np.bincount(inv, weights=(y - pb) ** 2, minlength=k)
    rng = np.random.default_rng(seed)
    w = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(n)]).astype(float)
    sst = w @ s2 - (w @ s1) ** 2 / (w @ cnt)
    delta = (w @ eb - w @ ea) / sst  # (1 - ea/sst) - (1 - eb/sst)
    return float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))


# --------------------------------------------------------------------------- experiment
def run_experiment(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    tr, va, te = (d[d["split"] == s] for s in ("train", "val", "test"))
    sets = {"fe": F.FE_FEATURES, "fewelch": F.FE_FEATURES + WELCH_COLS}
    preds, weights = {}, {}  # model -> (val, test); set -> chosen w
    for feats, cols in sets.items():
        fp = fit_predict(tr, va, te, cols)
        for kind in ("linear", "lgbm"):
            preds[f"{kind}_{feats}"] = fp[kind]
        w = choose_weight(va[TARGET], fp["linear"][0], fp["lgbm"][0])  # validation only
        weights[feats] = w
        preds[f"avg_{feats}"] = tuple(w * fp["linear"][i] + (1 - w) * fp["lgbm"][i] for i in (0, 1))
    feat_of = lambda m: m.split("_", 1)[1]  # noqa: E731
    val_r2 = {m: r2(va[TARGET], p[0]) for m, p in preds.items()}
    selected = max(val_r2, key=val_r2.get)  # best validation R2 over all arms
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
    cols = ["target", "model", "features", "split", "metric", "value", "n_obs", "ci_lo", "ci_hi", "selected"]
    return pd.DataFrame(rows)[cols], {"selected": selected, "weights": weights}


def reproduction_check(out: pd.DataFrame, metrics_csv: Path) -> pd.DataFrame:
    """Compare linear_fe / lgbm_fe rows with results/metrics.csv beta_1y rows."""
    ref = pd.read_csv(metrics_csv)
    ref = ref[(ref["target"] == TARGET) & ref["model"].isin(["linear_fe", "lgbm_fe"])
              & ref["metric"].isin(["r2", "mae"])]
    new = out[out["model"].isin(["linear_fe", "lgbm_fe"]) & out["metric"].isin(["r2", "mae"])]
    k = ["model", "split", "metric"]
    mg = ref.merge(new, on=k, suffixes=("_metrics", "_v7"))
    mg["abs_diff"] = (mg["value_metrics"] - mg["value_v7"]).abs()
    mg["n_equal"] = mg["n_obs_metrics"] == mg["n_obs_v7"]
    return mg[k + ["value_metrics", "value_v7", "abs_diff", "n_obs_metrics", "n_obs_v7", "n_equal"]]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    a = ap.parse_args(argv)
    d = build_dataset()
    out, info = run_experiment(d)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out_dir / "forward_v7.csv", index=False)
    chk = reproduction_check(out, ROOT / "results" / "metrics.csv")
    print("[reproduction vs results/metrics.csv]\n", chk.to_string(index=False))
    print("[weights]", info["weights"], "[selected on val]", info["selected"])
    piv = out[out["metric"].isin(["r2", "delta_r2_vs_linear_fe"])].pivot_table(
        index="model", columns=["split", "metric"], values="value")
    print(piv.round(4).to_string())
    print(out[out["metric"] == "delta_r2_vs_linear_fe"][["model", "value", "ci_lo", "ci_hi"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
