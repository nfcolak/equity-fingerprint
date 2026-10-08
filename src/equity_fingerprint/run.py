"""Full pipeline: panel -> features -> targets -> splits -> models -> results/.

Run: env -u PYTHONPATH PYTHONPATH=src python -m equity_fingerprint.run
"""
from __future__ import annotations

import json
import subprocess
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
from .models import run_target  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SHAP_SAMPLE = 1500


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def shap_plot(target: str, model, Xte: pd.DataFrame, classes, path: Path) -> None:
    X = Xte.sample(min(SHAP_SAMPLE, len(Xte)), random_state=T.SEED)
    names = [F.LABELS.get(c, c) for c in X.columns]
    sv = shap.TreeExplainer(model).shap_values(X)
    plt.figure()
    if classes is not None:
        if isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = [sv[:, :, k] for k in range(sv.shape[2])]
        shap.summary_plot(sv, X, feature_names=names, plot_type="bar", class_names=list(classes),
                          max_display=15, show=False)
        plt.title(f"{target}: mean |SHAP| by class (lgbm_fe, test sample)")
    else:
        shap.summary_plot(sv, X, feature_names=names, max_display=15, show=False)
        plt.title(f"{target}: SHAP summary (lgbm_fe, test sample)")
    plt.gcf().set_size_inches(10, 7)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")


def main() -> None:
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
    L_time = F.pc_loadings(panel, F.pc_factors(panel, tickers), pos)
    L_grp = F.pc_loadings(panel, F.pc_factors(panel, train_tickers), pos)
    time_df = base.merge(L_time, on=keys, how="left")
    grp_df = base.merge(L_grp, on=keys, how="left")

    # ---- targets
    ft = T.forward_targets(panel, time_df)
    time_df = pd.concat([time_df, ft], axis=1)
    grp_df["log_mcap"] = T.log_mcap_target(panel, grp_df)
    grp_df["tsplit"] = grp_df["ticker"].map(tsplit)

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
    d = grp_df[grp_df["log_mcap"].notna()].copy()
    d["split"] = d["tsplit"]
    datasets["log_mcap"] = d
    d = grp_df[grp_df["ticker"].isin(sec_map)].copy()
    d["sector"] = d["ticker"].map(sec_map)
    d["split"] = d["tsplit"]
    datasets["sector"] = d

    # ---- models
    metrics, imps, hps = [], [], {}
    for t in T.TARGETS:
        d = datasets[t]
        clf = t == "sector"
        print(f"[{t}] " + ", ".join(f"{s}={int((d['split'] == s).sum())}" for s in ("train", "val", "test")))
        rows, fitted, info, classes = run_target(t, d, t, F.RAW_FEATURES, F.FE_FEATURES, clf,
                                                 with_qlike=(t == "vol_1m"))
        metrics += rows
        hps[t] = info
        m, Xte = fitted["lgbm_fe"]
        gain = m.booster_.feature_importance("gain")
        imps.append(pd.DataFrame({"target": t, "feature": Xte.columns,
                                  "label": [F.LABELS.get(c, c) for c in Xte.columns],
                                  "gain": gain, "gain_share": gain / gain.sum()}))
        shap_plot(t, m, Xte, classes, RESULTS / f"shap_{t}.png")
        res = pd.DataFrame(rows)
        te = res[(res.split == "test") & (res.metric == ("macro_f1" if clf else "r2"))]
        print("   test " + ("macro_f1" if clf else "r2") + ": " +
              ", ".join(f"{r.model}={r.value:.3f}" for r in te.itertuples()))

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
    }
    (RESULTS / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    print("[done] results written to", RESULTS)


if __name__ == "__main__":
    main()
