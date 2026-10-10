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


def build_block(results: Path) -> str:
    m = pd.read_csv(results / "metrics.csv")
    test = m[m["split"] == "test"]

    lines = [
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
