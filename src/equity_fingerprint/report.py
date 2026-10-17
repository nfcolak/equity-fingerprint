"""Rebuild the README results block from results/metrics.csv (and run_info.json).

Run: env -u PYTHONPATH PYTHONPATH=src python -m equity_fingerprint.report
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
START, END = "<!-- results:start -->", "<!-- results:end -->"
TARGETS = ["vol_1m", "beta_1y", "log_mcap", "liquidity_1m", "sector"]
MODELS = ["baseline", "linear_raw", "linear_fe", "lgbm_raw", "lgbm_fe"]
MAIN = {t: ("macro_f1" if t == "sector" else "r2") for t in TARGETS}
MAIN_LABEL = {"r2": "R²", "macro_f1": "macro-F1"}
MODELS_V2 = ["baseline", "linear_raw", "linear_fe", "linear_fe2", "lgbm_raw", "lgbm_fe", "lgbm_fe2"]
ROB_TARGETS = [("log_mcap", "r2"), ("sector", "macro_f1")]


def _pick(df: pd.DataFrame, t: str, mod: str) -> pd.DataFrame:
    return df[(df.target == t) & (df.model == mod) & (df.metric == MAIN[t])]


def _val(df: pd.DataFrame, t: str, mod: str, metric: str, split: str = "test"):
    r = df[(df.target == t) & (df.model == mod) & (df.metric == metric) & (df.split == split)]
    return float(r["value"].iloc[0]) if len(r) else None


def _f(x, nd=3) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def build_v2(results: Path) -> list[str]:
    m = pd.read_csv(results / "metrics.csv")
    rb = pd.read_csv(results / "robustness.csv")
    rt = rb[rb["split"] == "test"]
    seeds = sorted(rb["seed"].unique())
    L = ["### v2: sector and market cap", "",
         "Added `linear_fe2` / `lgbm_fe2` (features `fe2`) for the two grouped targets: the `fe` set plus rolling "
         "252-day correlation and beta against equal-weight sector return indices (built from training-split "
         "tickers only, leave-one-out for a training ticker's own sector) and cross-sectional size proxies "
         "(log traded value minus the date median, and within-date ranks of log traded value and log close). "
         "v1 models and numbers are unchanged. Test split, ticker split seed 42. Source: `results/metrics.csv`.",
         "",
         "| Target | Metric | " + " | ".join(MODELS_V2) + " |",
         "|---|---|" + "|".join(["---:"] * len(MODELS_V2)) + "|"]
    for t, met, lab in (("sector", "macro_f1", "macro-F1"), ("sector", "accuracy", "accuracy"),
                        ("log_mcap", "r2", "R²"), ("log_mcap", "mae", "MAE")):
        L.append(f"| `{t}` | {lab} | " + " | ".join(_f(_val(m, t, mod, met)) for mod in MODELS_V2) + " |")

    L += ["", "Sector at ticker level (class probabilities averaged over a ticker's observations, then argmax). "
          "Tickers: val / test = "
          + " / ".join(str(int(m[(m.target == "sector") & (m.metric == "ticker_macro_f1") & (m.split == s_)]["n_obs"].iloc[0]))
                       for s_ in ("val", "test")) + ".", "",
          "| Model | Val accuracy | Val macro-F1 | Test accuracy | Test macro-F1 |", "|---|---:|---:|---:|---:|"]
    for mod in MODELS_V2:
        L.append(f"| {mod} | " + " | ".join(
            _f(_val(m, "sector", mod, k, s_)) for s_ in ("val", "test") for k in ("ticker_accuracy", "ticker_macro_f1")) + " |")

    ns = {t: sorted(rt[rt.target == t]["n_tickers"].unique().tolist()) for t, _ in ROB_TARGETS}
    L += ["", f"Robustness over ticker-split seeds {', '.join(map(str, seeds))} (same 70/15/15 stratified split, PCA and sector "
          "indices refitted per seed, tuned on validation, test scored once per seed). Test metric, mean ± sd over "
          f"{len(seeds)} seeds. Test tickers per seed: `log_mcap` {ns['log_mcap']}, `sector` {ns['sector']}. "
          "Source: `results/robustness.csv`.", "",
          "| Model | `log_mcap` R² | `sector` macro-F1 |", "|---|---:|---:|"]
    stats = {}
    for mod in MODELS_V2:
        cells = []
        for t, met in ROB_TARGETS:
            v = rt[(rt.target == t) & (rt.model == mod) & (rt.metric == met)]["value"]
            stats[(t, mod)] = (v.mean(), v.std(ddof=1), len(v))
            cells.append(f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else "n/a")
        L.append(f"| {mod} | " + " | ".join(cells) + " |")

    def paired(t, met, a, b):
        pa = rt[(rt.target == t) & (rt.model == a) & (rt.metric == met)].set_index("seed")["value"]
        pb = rt[(rt.target == t) & (rt.model == b) & (rt.metric == met)].set_index("seed")["value"]
        return (pb - pa).dropna()

    d_fe = paired("log_mcap", "r2", "linear_raw", "linear_fe")
    d_f2 = paired("sector", "macro_f1", "linear_fe", "linear_fe2")
    g_f2 = paired("sector", "macro_f1", "lgbm_fe", "lgbm_fe2")
    c_f2 = paired("log_mcap", "r2", "linear_fe", "linear_fe2")
    c_g2 = paired("log_mcap", "r2", "lgbm_fe", "lgbm_fe2")
    within = abs(d_fe.mean()) <= d_fe.std(ddof=1)
    sd_all = rt[(rt.target == "log_mcap") & (rt.model == "linear_raw") & (rt.metric == "r2")]["value"].std(ddof=1)
    L += ["",
          "Test change from `fe` to `fe2` at seed 42: "
          f"`sector` macro-F1 linear {(_val(m, 'sector', 'linear_fe2', 'macro_f1') - _val(m, 'sector', 'linear_fe', 'macro_f1')):+.3f}, "
          f"lgbm {(_val(m, 'sector', 'lgbm_fe2', 'macro_f1') - _val(m, 'sector', 'lgbm_fe', 'macro_f1')):+.3f}; "
          f"`log_mcap` R² linear {(_val(m, 'log_mcap', 'linear_fe2', 'r2') - _val(m, 'log_mcap', 'linear_fe', 'r2')):+.3f}, "
          f"lgbm {(_val(m, 'log_mcap', 'lgbm_fe2', 'r2') - _val(m, 'log_mcap', 'lgbm_fe', 'r2')):+.3f}. "
          f"Mean paired change over seeds (sd): `sector` macro-F1 linear {d_f2.mean():+.3f} ({d_f2.std(ddof=1):.3f}), "
          f"lgbm {g_f2.mean():+.3f} ({g_f2.std(ddof=1):.3f}); `log_mcap` R² linear {c_f2.mean():+.3f} ({c_f2.std(ddof=1):.3f}), "
          f"lgbm {c_g2.mean():+.3f} ({c_g2.std(ddof=1):.3f}).",
          "",
          "The v1 `log_mcap` gap (`linear_fe` minus `linear_raw`, test R²) is "
          f"{_val(m, 'log_mcap', 'linear_fe', 'r2') - _val(m, 'log_mcap', 'linear_raw', 'r2'):+.3f} at seed 42; "
          f"across the {len(d_fe)} seeds the paired gap has mean {d_fe.mean():+.3f} and sd {d_fe.std(ddof=1):.3f} "
          f"(range {d_fe.min():+.3f} to {d_fe.max():+.3f}), so the mean gap is "
          + ("within" if within else "larger than") + " one seed-to-seed sd. "
          f"The seed-to-seed sd of `linear_raw` test R² itself is {sd_all:.3f}.",
          ""]
    info_path = results / "run_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        if "runtime_seconds" in info:
            L += [f"Full pipeline run time (all five targets, v2 models and the {len(seeds)}-seed robustness reruns, "
                  f"CPU): {info['runtime_seconds'] / 60:.1f} min. Ticker-level `n_obs` in `metrics.csv` counts tickers.", ""]
    L += ["SHAP summaries of `lgbm_fe2` on a test sample:", "",
          "- [`sector`](results/shap_sector_fe2.png)", "- [`log_mcap`](results/shap_log_mcap_fe2.png)", ""]
    return L


def build_v3(results: Path) -> list[str]:
    """v3 subsection: HAR / GJR-GARCH volatility models and the sector co-movement-profile models.

    Rows of models that are not in the CSVs yet are skipped.
    """
    m = pd.read_csv(results / "metrics.csv")
    L = ["### v3: co-movement profile for sector, HAR/GARCH for volatility", "",
         "Added models next to v1/v2 (v1/v2 rows are unchanged). For `vol_1m`: `linear_har` / `lgbm_har` use only the "
         "three HAR inputs (log realised vol over the last 1, 5 and 22 trading days); `linear_fe3` / `lgbm_fe3` add "
         "the HAR inputs, a GJR-GARCH(1,1) 21-day mean volatility forecast and its persistence (Gaussian MLE per "
         "ticker on returns up to 2021-12-31, pooled parameters for tickers with fewer than 500 such returns) and "
         "SPI volatility state (log realised vol over 21 and 252 days and their difference) to the `fe` set. "
         "Test split. Source: `results/metrics.csv`.", ""]

    mods = ["baseline", "linear_fe", "lgbm_fe", "linear_har", "lgbm_har", "linear_fe3", "lgbm_fe3"]
    rows = []
    for mod in mods:
        r2, ql = _val(m, "vol_1m", mod, "r2"), _val(m, "vol_1m", mod, "qlike")
        if r2 is not None or ql is not None:
            rows.append(f"| {mod} | {_f(r2)} | {_f(ql)} |")
    if rows:
        L += ["`vol_1m` (R² on log vol; QLIKE on variance, lower is better):", "",
              "| Model | R² | QLIKE |", "|---|---:|---:|"] + rows + [""]

    rows = []
    for mod in ["linear_fe", "linear_fe2", "linear_fe3", "lgbm_fe3"]:
        f1, tf1 = _val(m, "sector", mod, "macro_f1"), _val(m, "sector", mod, "ticker_macro_f1")
        if f1 is not None or tf1 is not None:
            rows.append(f"| {mod} | {_f(f1)} | {_f(tf1)} |")
    if rows:
        L += ["`sector` (ticker split seed 42; `ticker_macro_f1` averages class probabilities per ticker):", "",
              "| Model | macro-F1 | ticker macro-F1 |", "|---|---:|---:|"] + rows + [""]

    rb_path = results / "robustness.csv"
    if rb_path.exists():
        rb = pd.read_csv(rb_path)
        rt = rb[(rb["split"] == "test") & (rb["target"] == "sector") & (rb["metric"] == "macro_f1")]
        rows = []
        for mod in ["linear_fe", "linear_fe2", "linear_fe3", "lgbm_fe", "lgbm_fe2", "lgbm_fe3"]:
            v = rt[rt.model == mod]["value"]
            if len(v) > 1:
                rows.append(f"| {mod} | {v.mean():.3f} ± {v.std(ddof=1):.3f} | {len(v)} |")
        if rows:
            L += ["`sector` robustness: test macro-F1, mean ± sd over ticker-split seeds. "
                  "Source: `results/robustness.csv`.", "",
                  "| Model | macro-F1 | Seeds |", "|---|---:|---:|"] + rows + [""]
    return L


V4_MODELS = ["linear_fp", "lgbm_fp", "linear_fpa", "lgbm_fpa", "lgbm_fefpa", "fpnet", "fpnet_a"]
V4_FEATURES = ["fp", "fpa", "fefpa", "fp_e2e", "fpa_e2e"]
V5_MODELS = ["linear_fpf", "lgbm_fpf", "linear_fpfa", "lgbm_fpfa", "lgbm_fefpfa",
             "linear_fpprof", "lgbm_fpprof", "linear_fe3fpprof", "lgbm_fe3fpprof"]
V5_FEATURES = ["fpf", "fpfa", "fefpfa", "fpprof", "fe3fpprof"]
V5_ARMS = ("ssl_far", "ssl_aux_far")
V5_ROB_MODELS = ["linear_fe3fpprof", "lgbm_fe3fpprof"]
V4_METRICS = {"vol_1m": [("r2", "R²"), ("mae", "MAE"), ("qlike", "QLIKE")],
              "beta_1y": [("r2", "R²"), ("mae", "MAE")], "log_mcap": [("r2", "R²"), ("mae", "MAE")],
              "liquidity_1m": [("r2", "R²"), ("mae", "MAE")],
              "sector": [("macro_f1", "macro-F1"), ("accuracy", "accuracy"), ("ticker_macro_f1", "ticker macro-F1")]}


def _best_old(m: pd.DataFrame, t: str):
    """Best v1-v3 model of target ``t`` by validation main metric (v4 models and the baseline excluded)."""
    v = m[(m.target == t) & (m.split == "val") & (m.metric == MAIN[t]) & (m.model != "baseline")
          & ~m.features.isin(V4_FEATURES + V5_FEATURES) & ~m.model.isin(V5_MODELS)]
    return v.loc[v["value"].idxmax(), "model"] if len(v) else None


def build_v4(results: Path) -> list[str]:
    """v4 subsection: frozen / fine-tuned fingerprint encoder against the best v1-v3 model. Missing rows are skipped."""
    m = pd.read_csv(results / "metrics.csv")
    if not m.model.isin(V4_MODELS).any():
        return []
    L = ["### v4: fingerprint encoder (self-supervised, pretrained outside Switzerland)", "",
         "A Transformer encoder reads the last 252 trading days of a stock (log return, high-low range, overnight gap, "
         "relative volume, market return, zero-volume flag) and is pretrained without Swiss data: masked patch "
         "modelling, contrastive views and a co-movement objective on non-Swiss stocks (Europe and US), dates up to "
         "2021-12-31 only. Arm `ssl` is purely self-supervised; arm `ssl_aux` adds non-Swiss sector and size labels as "
         "auxiliary heads. `linear_/lgbm_fp` use the frozen `ssl` embedding (128 dims), `*_fpa` the frozen `ssl_aux` "
         "embedding, `lgbm_fefpa` the `fe` features plus the `ssl_aux` embedding; `fpnet` / `fpnet_a` fine-tune the "
         "whole encoder (from `ssl` / `ssl_aux`) on the Swiss train split of each target. Test split, ticker split "
         "seed 42. Source: `results/metrics.csv`.", ""]
    info_path = results / "run_info.json"
    fp = json.loads(info_path.read_text()).get("fingerprint", {}) if info_path.exists() else {}
    for arm in ("ssl", "ssl_aux"):
        if arm in fp:
            a = fp[arm]
            L.append(f"- Arm `{arm}`: pretrained on {a.get('n_tickers', 'n/a')} non-Swiss tickers, data up to "
                     f"{a.get('max_date', 'n/a')}, {a.get('steps', 'n/a')} steps (checkpoint sha256 "
                     f"`{str(a.get('ckpt_sha256', 'n/a'))[:12]}`).")
    if fp:
        L.append("")
    test = m[m.split == "test"]
    for t in TARGETS:
        old = _best_old(m, t)
        new = [x for x in V4_MODELS if len(test[(test.target == t) & (test.model == x)])]
        if not new:
            continue
        mods = ([old] if old else []) + new
        head = [f"best v1-v3 by validation ({old})" if old else "best v1-v3"] + new
        n = test[(test.target == t) & (test.metric == V4_METRICS[t][0][0])]["n_obs"]
        L += [f"`{t}` (test, {int(n.iloc[0]) if len(n) else 'n/a'} observations):", "",
              "| Metric | " + " | ".join(head) + " |", "|---|" + "|".join(["---:"] * len(head)) + "|"]
        for met, lab in V4_METRICS[t]:
            vals = [_val(m, t, x, met) for x in mods]
            if any(v is not None for v in vals):
                L.append(f"| {lab} | " + " | ".join(_f(v) for v in vals) + " |")
        L.append("")

    rb_path = results / "robustness.csv"
    if rb_path.exists():
        rb = pd.read_csv(rb_path)
        rt = rb[(rb.split == "test") & (rb.target == "sector") & (rb.metric == "macro_f1")]
        ref = rt[rt.model == "linear_fe3"].set_index("seed")["value"]
        rows = []
        for mod in ["linear_fe3", "lgbm_fpa", "lgbm_fefpa", "fpnet_a"]:
            sub = rt[rt.model == mod].set_index("seed")["value"]
            if len(sub) > 1:
                d = (sub - ref).dropna()
                rows.append(f"| {mod} | {sub.mean():.3f} ± {sub.std(ddof=1):.3f} | {len(sub)} | "
                            + (f"{d.mean():+.3f}" if mod != "linear_fe3" and len(d) else "n/a") + " |")
        if rows:
            L += ["`sector` robustness: test macro-F1, mean ± sd over ticker-split seeds, and the mean paired change "
                  "against `linear_fe3`. Source: `results/robustness.csv`.", "",
                  "| Model | macro-F1 | Seeds | Δ vs linear_fe3 |", "|---|---:|---:|---:|"] + rows + [""]
    L += _v4_summary(m, rb if rb_path.exists() else None, fp)
    return L


def _best_new(m: pd.DataFrame, t: str):
    """Best v4 model of target ``t`` by validation main metric."""
    v = m[(m.target == t) & (m.split == "val") & (m.metric == MAIN[t]) & m.model.isin(V4_MODELS)]
    return v.loc[v["value"].idxmax(), "model"] if len(v) else None


def _v4_summary(m: pd.DataFrame, rb, fp: dict) -> list[str]:
    """Plain factual sentences (all numbers computed from the CSVs / run_info): v4 against v1-v3, sector seeds, provenance."""
    parts, wins, n_cmp = [], 0, 0
    for t in TARGETS:
        old, new = _best_old(m, t), _best_new(m, t)
        if not (old and new):
            continue
        vo, vn = _val(m, t, old, MAIN[t]), _val(m, t, new, MAIN[t])
        d = vn - vo
        n_cmp += 1
        wins += d > 0
        parts.append(f"`{t}` {new} {_f(vn)} vs {old} {_f(vo)} ({d:+.3f}, "
                     + ("v4 better" if d > 0 else "v4 not better") + ")")
    L = []
    if parts:
        L.append(f"On the test split, with the best model of each family chosen by validation "
                 f"(main metric: R² for regression, macro-F1 for `sector`), the best v4 model against the best v1-v3 model is: "
                 + "; ".join(parts) + f". The best v4 model is better on {wins} of {n_cmp} targets"
                 + (", so v4 does not beat the v1-v3 benchmark on every target." if wins < n_cmp else "."))
    if rb is not None:
        rt = rb[(rb.split == "test") & (rb.target == "sector") & (rb.metric == "macro_f1")]
        piv = rt.pivot_table(index="seed", columns="model", values="value")
        new_c = [c for c in piv.columns if c in V4_MODELS and piv[c].notna().all()]
        old_c = [c for c in piv.columns if c not in V4_MODELS and c != "baseline" and piv[c].notna().all()
                 and c not in V5_MODELS
                 and not rt[(rt.model == c)].features.isin(V4_FEATURES + V5_FEATURES).any()]
        if new_c and old_c:
            bn, bo = piv[new_c].mean().idxmax(), piv[old_c].mean().idxmax()
            d = piv[bn] - piv[bo]
            L.append(f"Over the {len(piv)} ticker-split seeds ({', '.join(map(str, piv.index))}), the best v4 `sector` model by "
                     f"mean test macro-F1 is {bn} ({piv[bn].mean():.3f} ± {piv[bn].std(ddof=1):.3f}) and the best v1-v3 model "
                     f"is {bo} ({piv[bo].mean():.3f} ± {piv[bo].std(ddof=1):.3f}); the paired mean difference is "
                     f"{d.mean():+.3f} (v4 higher in {int((d > 0).sum())} of {len(d)} seeds), so the sector gain of v4 is "
                     + ("positive" if d.mean() > 0 else "not positive") + " on average"
                     + (" but within one seed-to-seed sd." if abs(d.mean()) <= max(piv[bn].std(ddof=1), piv[bo].std(ddof=1)) else "."))
    pv = fp.get("provenance", {})
    arms = [fp[a] for a in ("ssl", "ssl_aux") if a in fp]
    if arms:
        nt = sorted({a.get("n_tickers") for a in arms})
        md = sorted({str(a.get("max_date"))[:10] for a in arms})
        st = sorted({a.get("steps") for a in arms})
        txt = (f"Provenance: pretrained on {'/'.join(map(str, nt))} non-Swiss tickers with data up to {'/'.join(md)}, "
               f"{'/'.join(map(str, st))} steps per arm")
        if pv:
            txt += (f", on Kaggle ({pv.get('gpu', 'GPU n/a')}, kernel `{pv.get('kaggle_kernel', 'n/a')}` version "
                    f"{pv.get('kernel_version', 'n/a')}, code commit `{pv.get('code_commit', 'n/a')}`, run {pv.get('run_date', 'n/a')})")
        L.append(txt + ". Checksums of all Kaggle outputs are in `results/run_info.json`.")
    return [" ".join(L), ""] if L else []


def _best_by_val(m: pd.DataFrame, t: str, models):
    """Best model of ``models`` for target ``t`` by validation main metric."""
    v = m[(m.target == t) & (m.split == "val") & (m.metric == MAIN[t]) & m.model.isin(models)]
    return v.loc[v["value"].idxmax(), "model"] if len(v) else None


def _best_v1_v4(m: pd.DataFrame, t: str):
    """Best v1-v4 model of target ``t`` by validation main metric (v5 models and the baseline excluded)."""
    v = m[(m.target == t) & (m.split == "val") & (m.metric == MAIN[t]) & (m.model != "baseline")
          & ~m.model.isin(V5_MODELS) & ~m.features.isin(V5_FEATURES)]
    return v.loc[v["value"].idxmax(), "model"] if len(v) else None


def build_v5(results: Path) -> list[str]:
    """v5 subsection: far-positive fingerprint arms and sector co-movement-profile models. [] when no v5 rows exist."""
    m = pd.read_csv(results / "metrics.csv")
    if not m.model.isin(V5_MODELS).any():
        return []
    L = ["### v5: far positives for the fingerprint encoder, same-date co-movement profile for sector", "",
         "Same Swiss evaluation as v4; new rows only. `linear_/lgbm_fpf` use the frozen `ssl_far` embedding, "
         "`*_fpfa` the frozen `ssl_aux_far` embedding and `lgbm_fefpfa` the `fe` features plus the `ssl_aux_far` "
         "embedding (arms pretrained with far positives: the two contrastive views are the same stock 252-756 trading days apart, so they never overlap). For "
         "`sector` only, `linear_/lgbm_fpprof` use same-date co-movement-profile features against the training "
         "tickers and `linear_/lgbm_fe3fpprof` add them to the `fe3` set. Test split, ticker split seed 42. "
         "Source: `results/metrics.csv`.", ""]
    info_path = results / "run_info.json"
    fp = json.loads(info_path.read_text()).get("fingerprint", {}) if info_path.exists() else {}
    for arm in V5_ARMS:
        if arm in fp:
            a = fp[arm]
            L.append(f"- Arm `{arm}`: pretrained on {a.get('n_tickers', 'n/a')} non-Swiss tickers, data up to "
                     f"{a.get('max_date', 'n/a')}, {a.get('steps', 'n/a')} steps (checkpoint sha256 "
                     f"`{str(a.get('ckpt_sha256', 'n/a'))[:12]}`).")
    if any(a in fp for a in V5_ARMS):
        L.append("")
    test = m[m.split == "test"]
    parts, rob_parts, wins, n_cmp = [], [], 0, 0
    for t in TARGETS:
        old, new = _best_v1_v4(m, t), _best_by_val(m, t, V5_MODELS)
        present = [x for x in V5_MODELS if len(test[(test.target == t) & (test.model == x)])]
        if not present:
            continue
        mods = ([old] if old else []) + present
        head = [f"best v1-v4 by validation ({old})" if old else "best v1-v4"] + present
        n = test[(test.target == t) & (test.metric == V4_METRICS[t][0][0])]["n_obs"]
        L += [f"`{t}` (test, {int(n.iloc[0]) if len(n) else 'n/a'} observations):", "",
              "| Metric | " + " | ".join(head) + " |", "|---|" + "|".join(["---:"] * len(head)) + "|"]
        for met, lab in V4_METRICS[t]:
            vals = [_val(m, t, x, met) for x in mods]
            if any(v is not None for v in vals):
                L.append(f"| {lab} | " + " | ".join(_f(v) for v in vals) + " |")
        L.append("")
        if old and new:
            vo, vn = _val(m, t, old, MAIN[t]), _val(m, t, new, MAIN[t])
            if vo is not None and vn is not None:
                n_cmp += 1
                wins += vn - vo > 0
                parts.append(f"`{t}` {new} {_f(vn)} vs {old} {_f(vo)} ({vn - vo:+.3f}, "
                             + ("v5 better" if vn - vo > 0 else "v5 not better") + ")")
    rb_path = results / "robustness.csv"
    rb = pd.read_csv(rb_path) if rb_path.exists() else None
    if rb is not None:
        rt = rb[(rb.split == "test") & (rb.target == "sector") & (rb.metric == "macro_f1")]
        piv = rt.pivot_table(index="seed", columns="model", values="value")
        v5_c = [c for c in V5_ROB_MODELS if c in piv.columns and piv[c].notna().sum() > 1]
        v4_c = [c for c in V4_MODELS if c in piv.columns and piv[c].notna().all()]
        if v5_c:
            cols = (["linear_fe3"] if "linear_fe3" in piv.columns else [])
            if v4_c:
                cols.append(piv[v4_c].mean().idxmax())
            cols += v5_c
            seeds = list(piv.index)
            ref = piv["linear_fe3"] if "linear_fe3" in piv.columns else None
            rows = []
            for c in cols:
                s = piv[c]
                d = (s - ref).dropna() if ref is not None and c != "linear_fe3" else []
                rows.append(f"| {c} | " + " | ".join(_f(s.get(sd)) for sd in seeds)
                            + f" | {s.mean():.3f} ± {s.std(ddof=1):.3f} | " + (f"{d.mean():+.3f}" if len(d) else "n/a") + " |")
            L += ["`sector` robustness: test macro-F1 per ticker-split seed, mean ± sd over seeds, and the mean paired "
                  "change against `linear_fe3`" + (f" (`{cols[1]}` is the best v4 model by mean test macro-F1)"
                                                   if v4_c and len(cols) > 1 and cols[1] in v4_c else "")
                  + ". Source: `results/robustness.csv`.", "",
                  "| Model | " + " | ".join(f"seed {sd}" for sd in seeds) + " | mean ± sd | Δ vs linear_fe3 |",
                  "|---|" + "|".join(["---:"] * (len(seeds) + 2)) + "|"] + rows + [""]
            old_ref = (['linear_fe3'] if ref is not None else []) + ([cols[1]] if v4_c and cols[1] in v4_c else [])
            ref_c = piv[old_ref].mean().idxmax() if old_ref else None  # best v1-v4 model by mean test macro-F1
            bn = piv[v5_c].mean().idxmax()
            if ref_c:
                d = (piv[bn] - piv[ref_c]).dropna()
                rob_parts.append(f"over the {len(d)} ticker-split seeds the paired mean test macro-F1 difference of {bn} "
                             f"({piv[bn].mean():.3f}) against {ref_c} ({piv[ref_c].mean():.3f}) is {d.mean():+.3f} "
                             f"(higher in {int((d > 0).sum())} of {len(d)} seeds, seed-to-seed sd of the difference "
                             f"{d.std(ddof=1):.3f})")
    if n_cmp:
        L.append(f"On the test split, with the best model of each family chosen by validation (main metric: R² for "
                 f"regression, macro-F1 for `sector`), the best v5 model against the best v1-v4 model is: "
                 + "; ".join(parts) + f". The best v5 model is better on {wins} of {n_cmp} targets"
                 + (", so v5 does not beat the v1-v4 benchmark on every target." if wins < n_cmp else "."))
    if rob_parts:
        L.append("Robustness: " + "; ".join(rob_parts) + ".")
    if L and L[-1] != "":
        L.append("")
    return L


def build_v6(results: Path) -> list[str]:
    """Headline section (v6 protocol): static targets per stock with repeated CV, forward targets against their
    persistence baseline and split-half noise ceiling. [] when neither v6 file exists."""
    sc_p, ce_p = results / "static_cv.csv", results / "ceilings.csv"
    if not sc_p.exists() and not ce_p.exists():
        return []
    L = ["### Headline (v6 protocol)", ""]
    if sc_p.exists():
        sc = pd.read_csv(sc_p)
        L += ["Static targets, one row per stock: repeated stratified 5-fold cross-validation over all eligible Swiss "
              "stocks (10 repeats), hyper-parameters by inner CV on the training folds only. Mean ± sd over repeats; "
              "95% interval from a stock bootstrap of the repeat-averaged metric. `sector` uses each stock's full daily "
              "price history; Utilities (4 stocks) is dropped because classes need at least 5 stocks. `knn` uses raw "
              "return correlations, `knn_resid` correlations of returns after removing the SPI market return, "
              "`knn_auto` lets inner CV choose among six market-adjusted correlation matrices and k (matrices use "
              "returns only, never labels). `log_mcap` is close × today's share count at the latest month-end (no "
              "historical share-count bias); `ridge_logtv21` uses only the 21-day traded value, which carries most "
              "of the signal. Source: `results/static_cv.csv`.", ""]
        for t, cols in (("sector", ["macro_f1", "weighted_f1", "accuracy", "balanced_accuracy"]), ("log_mcap", ["r2", "mae"])):
            d = sc[sc.target == t]
            if d.empty:
                continue
            n = int(d["n_stocks"].iloc[0])
            extra = f", {int(d['n_classes'].iloc[0])} classes" if t == "sector" and "n_classes" in d else ""
            L += [f"`{t}` ({n} stocks{extra}):", "",
                  "| Model | " + " | ".join(cols) + f" | {cols[0]} 95% interval |",
                  "|---|" + "|".join(["---:"] * (len(cols) + 1)) + "|"]
            for mod in dict.fromkeys(d["model"]):
                r = d[d.model == mod].set_index("metric")
                cells = [f"{r.loc[c, 'mean']:.3f} ± {r.loc[c, 'sd']:.3f}" if c in r.index else "n/a" for c in cols]
                ci = (f"[{r.loc[cols[0], 'ci_lo']:.3f}, {r.loc[cols[0], 'ci_hi']:.3f}]"
                      if cols[0] in r.index and pd.notna(r.loc[cols[0], "ci_lo"]) else "n/a")
                L.append(f"| {mod} | " + " | ".join(cells) + f" | {ci} |")
            L.append("")
    if ce_p.exists():
        ce = pd.read_csv(ce_p)
        m = pd.read_csv(results / "metrics.csv")
        L += ["Forward targets on the test split, read against two references: a persistence model (one feature, OLS "
              "fitted on train: past 21-day log vol, past 252-day beta, past 21-day log traded value) and the split-half "
              "reliability of the target (odd vs even days of the forward window, Spearman-Brown corrected). The "
              "reliability describes how noisy the target measurement is; it is not an attainable forecasting bound, "
              "because both halves share shocks that are unknowable at the forecast date. Best model = "
              "highest validation R² over all model rows. Skill = (R² − persistence R²) / (1 − persistence R²). "
              "Source: `results/ceilings.csv`, `results/metrics.csv`.", "",
              "| Target | Best model (by validation) | Test R² | Persistence R² | Skill over persistence | Measurement reliability |",
              "|---|---|---:|---:|---:|---:|"]
        for t in ("vol_1m", "beta_1y", "liquidity_1m"):
            v = m[(m.target == t) & (m.split == "val") & (m.metric == "r2") & (m.model != "baseline")]
            c = ce[(ce.target == t) & (ce.split == "test")]
            if v.empty or c.empty:
                continue
            best = v.loc[v["value"].idxmax(), "model"]
            r2 = _val(m, t, best, "r2")
            if r2 is None:
                continue
            pr, rel = float(c["persistence_ols_r2"].iloc[0]), float(c["reliability"].iloc[0])
            L.append(f"| `{t}` | {best} | {r2:.3f} | {pr:.3f} | {(r2 - pr) / (1 - pr):.2f} | {rel:.3f} |")
        L += ["", "`liquidity_1m` is almost fully explained by last month's traded value, so it is reported but is not a "
                  "headline result.", ""]
    intro = {
        "forward_v7": "Welch slope-winsorised betas (stock return clipped to [-2m, 4m] around the market return m, "
                      "252-day and age-decayed) and linear/LightGBM averages",
        "forward_kalman": "a filtered random-walk Kalman beta (noise ratio chosen by train-period likelihood), alone "
                          "and with the Welch betas",
        "forward_minirocket": "MiniRocket features (84 fixed convolution kernels over the past 252 days of stock "
                              "return, market return and traded value; biases fitted on training windows) with "
                              "ridge, alone and with the engineered features",
    }
    for name, what in intro.items():
        f = results / f"{name}.csv"
        if not f.exists():
            continue
        fw = pd.read_csv(f)
        for t in dict.fromkeys(fw["target"]):
            L += [f"`{t}` with {what}. The arm marked ★ was chosen on validation before test was read; Δ is test R² "
                  f"minus `linear_fe` with a 95% stock-block bootstrap interval. Source: `results/{name}.csv`.", "",
                  "| Arm | Val R² | Test R² | Test Δ vs linear_fe | 95% interval |", "|---|---:|---:|---:|---:|"]
            ft = fw[fw.target == t]
            for mod in dict.fromkeys(ft["model"]):
                r = ft[ft.model == mod]
                g = lambda sp, me: r[(r.split == sp) & (r.metric == me)]
                vr, tr, dl = g("val", "r2"), g("test", "r2"), g("test", "delta_r2_vs_linear_fe")
                if vr.empty or tr.empty:
                    continue
                star = " ★" if int(r["selected"].max()) == 1 else ""
                d = (f"{dl['value'].iloc[0]:+.3f} | [{dl['ci_lo'].iloc[0]:+.3f}, {dl['ci_hi'].iloc[0]:+.3f}]"
                     if not dl.empty else "– | –")
                L.append(f"| {mod}{star} | {vr['value'].iloc[0]:.3f} | {tr['value'].iloc[0]:.3f} | {d} |")
            L.append("")
    return L


def build_block(results: Path) -> str:
    m = pd.read_csv(results / "metrics.csv")
    test = m[m["split"] == "test"]

    v6 = build_v6(results)
    lines = v6 + (["### Per-observation benchmark (v1-v5)", ""] if v6 else []) + [
        "Test split (time targets: observations from 2024-01-01 on; grouped targets: held-out tickers). "
        "Main metric: R² for the four regression targets (on the log scale for `vol_1m`, `log_mcap`, "
        "`liquidity_1m`), macro-F1 for `sector`. Source: `results/metrics.csv`.",
        "",
        "| Target | Metric | Test obs | " + " | ".join(MODELS) + " |",
        "|---|---|---:|" + "|".join(["---:"] * len(MODELS)) + "|",
    ]
    for t in TARGETS:
        cells, n = [], None
        for mod in MODELS:
            r = _pick(test, t, mod)
            cells.append(f"{r['value'].iloc[0]:.3f}" if len(r) else "n/a")
            if len(r):
                n = int(r["n_obs"].iloc[0])
        lines.append(f"| `{t}` | {MAIN_LABEL[MAIN[t]]} | {n if n is not None else 'n/a'} | "
                     + " | ".join(cells) + " |")

    parts = []
    for fam in ("linear", "lgbm"):
        d = []
        for t in TARGETS:
            a, b = _pick(test, t, f"{fam}_raw"), _pick(test, t, f"{fam}_fe")
            if len(a) and len(b):
                d.append(f"`{t}` {b['value'].iloc[0] - a['value'].iloc[0]:+.3f}")
        parts.append(f"{fam}: " + ", ".join(d))
    lines += ["", "What feature engineering changed (test metric, `fe` minus `raw`; "
              "R² for regression, macro-F1 for sector): " + "; ".join(parts) + ".", ""]

    info_path = results / "run_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        no = info["n_obs"]
        lines.append(
            f"Run {info['run_date']} (code at commit `{info['git_sha'][:7]}`): {info['n_tickers']} tickers, "
            f"{info['n_observations_total']} month-end observations "
            f"({info['observation_range'][0]} to {info['observation_range'][1]}). "
            "Observations per target (train / val / test): "
            + "; ".join(f"`{t}` {no[t]['train']} / {no[t]['val']} / {no[t]['test']}" for t in TARGETS)
            + ".")
        dr = info.get("dropped_sector_classes", [])
        if dr:
            lines.append("")
            lines.append("Sector classes dropped (fewer than 5 tickers): "
                         + ", ".join(f"{d['sector']} ({d['n_tickers']})" for d in dr) + ".")
        lines.append("")
    lines += ["SHAP summaries of `lgbm_fe` on a test sample:", ""]
    lines += [f"- [`{t}`](results/shap_{t}.png)" for t in TARGETS]
    lines += [""] + build_v2(results)
    lines += build_v3(results)
    lines += build_v4(results)
    lines += build_v5(results)
    lines += ["All metrics (validation and test, MAE/RMSE/QLIKE, accuracy, log loss): "
              "[`results/metrics.csv`](results/metrics.csv); feature gain: "
              "[`results/feature_importance.csv`](results/feature_importance.csv)."]
    return "\n".join(lines)


def main() -> None:
    readme = ROOT / "README.md"
    text = readme.read_text()
    if START not in text or END not in text:
        raise SystemExit(f"README.md must contain {START} and {END}")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(head + START + "\n" + build_block(ROOT / "results") + "\n" + END + tail)
    print("README results block updated")


if __name__ == "__main__":
    main()
