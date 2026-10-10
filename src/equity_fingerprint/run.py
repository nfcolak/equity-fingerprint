"""Full pipeline: panel -> features -> targets -> splits -> models -> results/.

Run: env -u PYTHONPATH PYTHONPATH=src python -m equity_fingerprint.run
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from . import features as F  # noqa: E402
from . import targets as T  # noqa: E402
from .data import _data_dir  # noqa: E402
from .embed import sector_profile_features  # noqa: E402
from .models import run_target  # noqa: E402

ROB_SEEDS = [42, 1, 2, 3, 4]
ROB_METRICS = {"log_mcap": "r2", "sector": "macro_f1"}

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SHAP_SAMPLE = 1500


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def shap_plot(target: str, model, Xte: pd.DataFrame, classes, path: Path, model_name: str = "lgbm_fe") -> None:
    X = Xte.sample(min(SHAP_SAMPLE, len(Xte)), random_state=T.SEED)
    names = [F.LABELS.get(c, c) for c in X.columns]
    sv = shap.TreeExplainer(model).shap_values(X)
    plt.figure()
    if classes is not None:
        if isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = [sv[:, :, k] for k in range(sv.shape[2])]
        shap.summary_plot(sv, X, feature_names=names, plot_type="bar", class_names=list(classes),
                          max_display=15, show=False)
        plt.title(f"{target}: mean |SHAP| by class (" + model_name + ", test sample)")
    else:
        shap.summary_plot(sv, X, feature_names=names, max_display=15, show=False)
        plt.title(f"{target}: SHAP summary ({model_name}, test sample)")
    plt.gcf().set_size_inches(10, 7)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")


def grouped_datasets(panel, base, pos, tickers, sec_all, sec_map, seed: int):
    """Ticker split for ``seed`` and the log_mcap / sector datasets (PCA + sector indices fit on its train tickers)."""
    tsplit = T.ticker_split(tickers, sec_all.to_dict(), seed)
    train_tickers = [t for t in tickers if tsplit[t] == "train"]
    keys = ["pos", "ticker"]
    grp_df = base.merge(F.pc_loadings(panel, F.pc_factors(panel, train_tickers), pos), on=keys, how="left")
    sec_df, sec_cols = F.sector_features(panel, pos, train_tickers, sec_map)
    grp_df = grp_df.merge(sec_df, on=keys, how="left")
    size = F.size_features(base)
    for c in F.SIZE_FEATURES:
        grp_df[c] = size[c].to_numpy()
    grp_df["log_mcap"] = T.log_mcap_target(panel, grp_df)
    grp_df["tsplit"] = grp_df["ticker"].map(tsplit)
    ds = {}
    d = grp_df[grp_df["log_mcap"].notna()].copy()
    d["split"] = d["tsplit"]
    ds["log_mcap"] = d
    d = grp_df[grp_df["ticker"].isin(sec_map)].copy()
    # v3: co-movement with this seed's train tickers (kNN profile + return embedding), sector dataset only
    prof_df, prof_cols = sector_profile_features(panel, pos, train_tickers, sec_map)
    d = d.merge(prof_df, on=keys, how="left")
    d["sector"] = d["ticker"].map(sec_map)
    d["split"] = d["tsplit"]
    ds["sector"] = d
    fe2 = F.FE_FEATURES + F.SIZE_FEATURES + sec_cols
    return ds, fe2, fe2 + prof_cols


def robustness_rows(seed: int, rows: list[dict], ds: dict) -> list[dict]:
    out = []
    for r in rows:
        if r["split"] in ("val", "test") and r["metric"] == ROB_METRICS[r["target"]]:
            d = ds[r["target"]]
            out.append(dict(target=r["target"], model=r["model"], features=r["features"], seed=seed,
                            split=r["split"], metric=r["metric"], value=r["value"],
                            n_tickers=int(d.loc[d["split"] == r["split"], "ticker"].nunique())))
    return out


def main() -> None:
    t_start = time.time()
    RESULTS.mkdir(exist_ok=True)
    data_dir = _data_dir()
    panel = F.build_panel()
    pos = F.month_end_positions(panel)
    base = F.base_features(panel, pos)
    tickers = sorted(base["ticker"].unique())
    print(f"[panel] {len(panel.tickers)} CHF tickers, {len(tickers)} with observations, {len(base)} observations")

    # ---- ticker split (grouped targets) and sector classes
    sec_all = panel.info["sector"].reindex(tickers)
    sec_map, dropped = T.sector_map(panel, tickers)
    for d in dropped:
        print(f"[sector] dropped class {d['sector']!r} ({d['n_tickers']} tickers < {T.MIN_CLASS_TICKERS})")
    tsplit = T.ticker_split(tickers, sec_all.to_dict(), T.SEED)
    train_tickers = [t for t in tickers if tsplit[t] == "train"]

    # ---- return-structure (PCA) features: time targets use all tickers, grouped targets train tickers only
    keys = ["pos", "ticker"]
    EXTRA_SETS: dict[str, list[tuple[str, list[str]]]] = {}
    L_time = F.pc_loadings(panel, F.pc_factors(panel, tickers), pos)
    time_df = base.merge(L_time, on=keys, how="left")
    grp_ds, fe2_cols, fe3_cols = grouped_datasets(panel, base, pos, tickers, sec_all, sec_map, T.SEED)
    EXTRA_SETS["sector"] = [("fe3", fe3_cols)]

    # ---- targets
    ft = T.forward_targets(panel, time_df)
    time_df = pd.concat([time_df, ft], axis=1)

    datasets, n_info = {}, {}
    for t in T.TIME_TARGETS:
        d = time_df.copy()
        d["split"] = T.time_split(d["date"], d[f"end_{t}"])
        n_before = int(d[t].notna().sum())
        d = d[d[t].notna() & d["split"].notna()]
        n_info[t] = {"n_obs_with_target": n_before, "dropped_split_overlap": int(
            ((time_df[t].notna()) & (T.time_split(time_df["date"], time_df[f"end_{t}"]).isna())).sum())}
        if t == "vol_1m":  # windows with zero realized vol have no defined log: dropped and counted
            n_info[t]["dropped_zero_vol_windows"] = int(d["vol_1m_floored"].sum())
            d = d[~d["vol_1m_floored"]]
        datasets[t] = d
    datasets.update(grp_ds)

    # ---- v3 vol_1m: HAR / GJR-GARCH / market-vol features (past returns only; GARCH parameters fit on <= 2021-12-31)
    from .volmodels import GARCH_COLS, HAR_COLS, MKT_COLS, vol_features
    datasets["vol_1m"] = datasets["vol_1m"].merge(vol_features(panel, datasets["vol_1m"][keys]), on=keys, how="left")
    EXTRA_SETS["vol_1m"] = [("har", HAR_COLS), ("fe3", F.FE_FEATURES + HAR_COLS + GARCH_COLS + MKT_COLS)]

    # ---- models
    metrics, imps, hps, rob = [], [], {}, []
    for t in T.TARGETS:
        d = datasets[t]
        clf = t == "sector"
        print(f"[{t}] " + ", ".join(f"{s}={int((d['split'] == s).sum())}" for s in ("train", "val", "test")))
        grouped = t in T.GROUP_TARGETS
        rows, fitted, info, classes = run_target(t, d, t, F.RAW_FEATURES, F.FE_FEATURES, clf,
                                                 with_qlike=(t == "vol_1m"),
                                                 fe2_cols=fe2_cols if grouped else None,
                                                 extra_sets=EXTRA_SETS.get(t))
        if grouped:
            rob += robustness_rows(T.SEED, rows, datasets)
        metrics += rows
        hps[t] = info
        m, Xte = fitted["lgbm_fe"]
        gain = m.booster_.feature_importance("gain")
        imps.append(pd.DataFrame({"target": t, "feature": Xte.columns,
                                  "label": [F.LABELS.get(c, c) for c in Xte.columns],
                                  "gain": gain, "gain_share": gain / gain.sum()}))
        shap_plot(t, m, Xte, classes, RESULTS / f"shap_{t}.png")
        if grouped:
            m2, Xte2 = fitted["lgbm_fe2"]
            shap_plot(t, m2, Xte2, classes, RESULTS / f"shap_{t}_fe2.png", "lgbm_fe2")
        res = pd.DataFrame(rows)
        te = res[(res.split == "test") & (res.metric == ("macro_f1" if clf else "r2"))]
        print("   test " + ("macro_f1" if clf else "r2") + ": " +
              ", ".join(f"{r.model}={r.value:.3f}" for r in te.itertuples()))

    # ---- robustness of the grouped targets over ticker-split seeds (seed 42 = the main run above)
    for seed in ROB_SEEDS:
        if seed == T.SEED:
            continue
        t0 = time.time()
        ds, cols2, cols3 = grouped_datasets(panel, base, pos, tickers, sec_all, sec_map, seed)
        for t in T.GROUP_TARGETS:
            rows, _, _, _ = run_target(t, ds[t], t, F.RAW_FEATURES, F.FE_FEATURES, t == "sector", fe2_cols=cols2,
                                    extra_sets=[("fe3", cols3)] if t == "sector" else None)
            rob += robustness_rows(seed, rows, ds)
        print(f"[robustness] seed {seed} done in {time.time() - t0:.0f}s")
    rob_df = pd.DataFrame(rob)
    rob_df["_s"] = rob_df["seed"].map({s_: i for i, s_ in enumerate(ROB_SEEDS)})
    rob_df = rob_df.sort_values(["target", "_s"], kind="stable").drop(columns="_s")
    rob_df.to_csv(RESULTS / "robustness.csv", index=False)

    pd.DataFrame(metrics).to_csv(RESULTS / "metrics.csv", index=False)
    pd.concat(imps).sort_values(["target", "gain"], ascending=[True, False]).to_csv(
        RESULTS / "feature_importance.csv", index=False)

    # ---- run_info
    manifest = json.loads((data_dir / "manifest.json").read_text())
    keep = ["run_date", "n_universe", "n_with_prices", "n_with_>=500_rows", "price_date_range",
            "index_date_ranges", "n_with_sector", "sector_counts", "n_with_shares"]
    n_obs = {t: {s: int((datasets[t]["split"] == s).sum()) for s in ("train", "val", "test")} for t in T.TARGETS}
    n_tick = {t: {s: int(datasets[t].loc[datasets[t]["split"] == s, "ticker"].nunique())
                  for s in ("train", "val", "test")} for t in T.TARGETS}
    info = {
        "run_date": str(date.today()),
        "git_sha": git_sha(),
        "n_tickers": len(tickers),
        "n_chf_tickers_with_prices": len(panel.tickers),
        "n_observations_total": int(len(base)),
        "observation_range": [str(base["date"].min())[:10], str(base["date"].max())[:10]],
        "n_obs": n_obs,
        "n_tickers_by_split": n_tick,
        "splits": {"time": {"train": "<= 2021-12-31", "val": "2022-01-01..2023-12-31", "test": ">= 2024-01-01"},
                   "ticker": {"train/val/test": "70/15/15", "seed": T.SEED, "stratified_by": "sector"}},
        "dropped_sector_classes": dropped,
        "n_sector_classes": len(set(sec_map.values())),
        "excluded_non_chf_tickers": panel.excluded_currency,
        "cleaning": {"filled_missing_days": panel.n_filled_days,
                     "artifact_returns_zeroed": panel.n_artifact_returns,
                     "artifact_threshold_abs_log_return": F.ARTIFACT_LOGRET},
        "time_target_notes": n_info,
        "data_manifest": {k: manifest[k] for k in keep if k in manifest}
                         | {"n_failed_tickers": len(manifest.get("failed_tickers", {}))},
        "chosen_hyperparameters": hps,
        "robustness_seeds": ROB_SEEDS,
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    (RESULTS / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    print("[done] results written to", RESULTS)


if __name__ == "__main__":
    main()
