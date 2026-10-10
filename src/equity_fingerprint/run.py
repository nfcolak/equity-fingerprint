"""Full pipeline: panel -> features -> targets -> splits -> models -> results/.

Run: env -u PYTHONPATH PYTHONPATH=src python -m equity_fingerprint.run
"""
from __future__ import annotations

import json
import os
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
from .models import classification_metrics, regression_metrics, run_target, ticker_level_metrics  # noqa: E402

ROB_SEEDS = [42, 1, 2, 3, 4]
ROB_METRICS = {"log_mcap": "r2", "sector": "macro_f1"}

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SHAP_SAMPLE = 1500
KEYS = ["pos", "ticker"]
FT_MODELS = {"fpnet": "fp_e2e", "fpnet_a": "fpa_e2e"}  # v4 fine-tuned models -> features label


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


# --------------------------------------------------------------------------- v4: fingerprint encoder
def fp_dir() -> Path:
    return Path(os.environ.get("EF_FP_DIR") or ROOT / "data" / "fingerprint")


def load_fp(d: Path):
    """Frozen embeddings (fp_* from arm ssl, fpa_* from ssl_aux) keyed by pos/ticker; missing files are skipped."""
    frames, cols = [], {"fp": [], "fpa": []}
    for arm, pre in (("ssl", "fp"), ("ssl_aux", "fpa")):
        path = d / f"fp_embeddings_{arm}.parquet"
        if not path.exists():
            continue
        e = pd.read_parquet(path).drop(columns="date")
        e.columns = [c if c in KEYS else c.replace("fp_", f"{pre}_", 1) for c in e.columns]
        cols[pre] = [c for c in e.columns if c not in KEYS]
        frames.append(e)
    return frames, cols


def merge_fp(ds: dict, frames: list) -> dict:
    """Left-merge the embeddings into every dataset (row order unchanged)."""
    for e in frames:
        ds = {t: d.merge(e, on=KEYS, how="left") for t, d in ds.items()}
    return ds


def fp_sets(cols: dict) -> list[tuple[str, list[str]]]:
    sets = [("fp", cols["fp"])] if cols["fp"] else []
    if cols["fpa"]:
        sets += [("fpa", cols["fpa"]), ("fefpa", F.FE_FEATURES + cols["fpa"])]
    return sets


def ft_target_rows(name: str, feats: str, t: str, d: pd.DataFrame, pt: pd.DataFrame) -> list[dict]:
    """Metric rows (same layout as ``run_target``) for the fine-tuned predictions of one target."""
    clf = t == "sector"
    classes = np.array(sorted(d.loc[d["split"] == "train", t].unique())) if clf else None
    pcols = [f"proba_{c}" for c in classes] if clf else ["pred"]
    if not set(pcols) <= set(pt.columns):
        return []
    parts = {s: d[d["split"] == s].merge(pt.loc[pt["split"] == s, KEYS + pcols], on=KEYS) for s in ("val", "test")}
    rv = 0.0
    if t == "vol_1m":
        rv = float(np.var(parts["val"][t] - parts["val"]["pred"]))
    rows = []
    for split, part in parts.items():
        if clf:
            proba = part[pcols].to_numpy()
            met = classification_metrics(part[t], proba, classes)
            tm = ticker_level_metrics(part, t, proba, classes)
            n_tick = tm.pop("_n")
        else:
            met = regression_metrics(part[t], part["pred"], t == "vol_1m", rv)
        for k, v in met.items():
            rows.append(dict(target=t, model=name, features=feats, split=split, metric=k, value=float(v), n_obs=len(part)))
        if clf:
            for k, v in tm.items():
                rows.append(dict(target=t, model=name, features=feats, split=split, metric=k, value=float(v), n_obs=n_tick))
    return rows


def ft_rows(d: Path, seed: int, ds: dict) -> list[dict]:
    rows = []
    for name, feats in FT_MODELS.items():
        path = d / f"{name}_preds_seed{seed}.parquet"
        if not path.exists():
            continue
        preds = pd.read_parquet(path)
        for t in ds:
            pt = preds[preds["target"] == t]
            if len(pt):
                rows += ft_target_rows(name, feats, t, ds[t], pt)
    return rows


def label_frame(seed: int, ds: dict) -> pd.DataFrame:
    """Long-format labels of the datasets; ``y`` is stored as text (sector names and floats share a column)."""
    return pd.concat([pd.DataFrame({"seed": seed, "target": t, "pos": d["pos"].to_numpy(),
                                    "ticker": d["ticker"].to_numpy(), "date": d["date"].to_numpy(),
                                    "y": d[t].astype(str).to_numpy(), "split": d["split"].to_numpy()})
                      for t, d in ds.items()], ignore_index=True)


def dump_labels(path: Path, frames: list) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(path / "labels.parquet", index=False)
    print("[labels] written to", path / "labels.parquet")


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

    # ---- v4: label export (seed 42; robustness seeds are added below)
    dump = os.environ.get("EF_DUMP_LABELS")
    labels = [label_frame(T.SEED, datasets)] if dump else []
    if dump and os.environ.get("EF_LABELS_ONLY") == "1":
        for seed in ROB_SEEDS:
            if seed != T.SEED:
                labels.append(label_frame(seed, grouped_datasets(panel, base, pos, tickers, sec_all, sec_map, seed)[0]))
        dump_labels(Path(dump), labels)
        return

    # ---- v4: frozen fingerprint embeddings (files missing -> v1-v3 run unchanged)
    fp_d = fp_dir()
    fp_frames, fp_cols = load_fp(fp_d)
    if fp_frames:
        datasets = merge_fp(datasets, fp_frames)
        for t in T.TARGETS:
            EXTRA_SETS.setdefault(t, []).extend(fp_sets(fp_cols))

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
    seed_ds = {T.SEED: datasets}
    for seed in ROB_SEEDS:
        if seed == T.SEED:
            continue
        t0 = time.time()
        ds, cols2, cols3 = grouped_datasets(panel, base, pos, tickers, sec_all, sec_map, seed)
        if dump:
            labels.append(label_frame(seed, ds))
        ds = merge_fp(ds, fp_frames)
        seed_ds[seed] = ds
        for t in T.GROUP_TARGETS:
            extra = ([("fe3", cols3)] if t == "sector" else []) + (fp_sets(fp_cols) if fp_frames else [])
            rows, _, _, _ = run_target(t, ds[t], t, F.RAW_FEATURES, F.FE_FEATURES, t == "sector", fe2_cols=cols2,
                                    extra_sets=extra or None)
            rob += robustness_rows(seed, rows, ds)
        print(f"[robustness] seed {seed} done in {time.time() - t0:.0f}s")
    if dump:
        dump_labels(Path(dump), labels)

    # ---- v4: end-to-end fine-tuned encoders (fpnet, fpnet_a): predictions scored like every other model
    for seed, ds in seed_ds.items():
        rows = ft_rows(fp_d, seed, ds)
        if seed == T.SEED:
            metrics += rows
        rob += robustness_rows(seed, [r for r in rows if r["target"] in ROB_METRICS], ds)
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
    fp_info = {arm: json.loads(p.read_text()) for arm in ("ssl", "ssl_aux")
               if (p := fp_d / f"fp_embeddings_{arm}.json").exists()}
    logs = {name: {str(s): json.loads(p.read_text()) for s in ROB_SEEDS
                   if (p := fp_d / f"{name}_log_seed{s}.json").exists()} for name in FT_MODELS}
    logs = {k: v for k, v in logs.items() if v}
    if fp_info or logs:
        info["fingerprint"] = {**fp_info, "fpnet_logs": logs}
        for key, fname in (("outputs_sha256", "outputs_sha256.json"), ("provenance", "provenance.json")):
            if (fp_d / fname).exists():
                info["fingerprint"][key] = json.loads((fp_d / fname).read_text())
    (RESULTS / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    print("[done] results written to", RESULTS)


if __name__ == "__main__":
    main()
