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


def _pick(df: pd.DataFrame, t: str, mod: str) -> pd.DataFrame:
    return df[(df.target == t) & (df.model == mod) & (df.metric == MAIN[t])]


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
    lines += ["", "All metrics (validation and test, MAE/RMSE/QLIKE, accuracy, log loss): "
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
