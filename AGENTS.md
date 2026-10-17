# Equity Fingerprint

Inferring equity metadata from market data. Predict five characteristics of Swiss listed stocks (SIX primary listings, SPI universe) from daily price and volume alone, and show how feature engineering improves linear regression and gradient boosted trees. Portfolio project for quant / data science roles in Switzerland.

## Tech stack

| Item | Value |
|---|---|
| Language | Python 3.12, project venv `.venv/` (run with `env -u PYTHONPATH .venv/bin/python`) |
| Libraries | pandas, numpy, scikit-learn, LightGBM, SHAP, matplotlib, pyarrow, yfinance, torch, tabpfn 9.2.0 (v9 `tabpfn_mcap.py` only, public v2 checkpoint `data/models/tabpfn-v2-regressor.ckpt` or `EF_TABPFN_CKPT`; never import it together with lightgbm) |
| Code | `src/equity_fingerprint/` (package), entry points run as `python -m equity_fingerprint.<module>` with `PYTHONPATH=src` |

## Data sources (all free)

| Source | Use |
|---|---|
| SIX issuer list `https://www.six-group.com/sheldon/equity_issuers/v1/equity_issuers.csv` (`;`-separated, latin-1) | Universe: rows with `Primary listing == TRUE`; yfinance ticker = `<Symbol>.SW` |
| yfinance daily OHLCV | Stock prices/volume; market index `^SSHI` (SPI, from 2013-03) and `^SSMI` (SMI) |
| yfinance `^VIX`, `^STOXX50E` closes | v9 vol_1m features only: `data/raw/extra_index.parquet` via `python -m equity_fingerprint.extra_market` |
| yfinance `Ticker.info` snapshot | Labels: `sector`, `industry`, `sharesOutstanding`, `marketCap` |
| Free index constituent lists (Wikipedia: S&P 500/400/600, FTSE 100/250, DAX/MDAX/SDAX, CAC 40/SBF 120, AEX/AMX, IBEX 35, FTSE MIB, OMX 30/25, OBX, BEL 20, ATX, PSI) + yfinance | v4 pretraining universe only (non-Swiss stocks): `data/pretrain/` (override `EF_PRETRAIN_DIR`) |

yfinance is unofficial and for personal/research use. The universe is today's listing (survivorship bias); say so in the README.

## v4: Fingerprint encoder (user decision 2026-10-10)

| Rule | Why |
|---|---|
| A self-supervised encoder (masked patch modelling + contrastive) is pretrained on NON-Swiss stocks (Europe + US, ~3-5k) on data up to 2021-12-31 only. No `.SW` ticker and no date after 2021-12-31 ever enters pretraining. | Swiss evaluation stays a clean transfer test; no look-ahead. |
| Non-Swiss sector/size labels may be used as auxiliary pretraining heads; Swiss labels only in fine-tuning on each target's train split. | Swiss test labels never touch the encoder. |
| Swiss evaluation is unchanged: same targets, splits, seeds, metrics. New rows are added (`*_fp*`, `fpnet`); v1-v3 rows stay bit-identical. | Comparable to the existing benchmark. |
| GPU training runs on Kaggle (free T4/P100), approved by the user 2026-10-10. Code must also run a tiny CPU smoke (`--device cpu`). | User approval; testable locally. |
| torch is a project dependency (CPU on the Mac, CUDA on Kaggle). | v4 model code. |

## Targets

| Target | Type | Definition | Split |
|---|---|---|---|
| `vol_1m` | regression | annualized realized vol of daily log returns over the next 21 trading days (model on log) | time |
| `beta_1y` | regression | beta vs `^SSHI` over the next 252 trading days | time |
| `log_mcap` | regression | log(close × sharesOutstanding snapshot) at the feature date | by ticker (grouped) |
| `liquidity_1m` | regression | log mean daily traded value (close × volume, CHF) over the next 21 trading days | time |
| `sector` | classification | yfinance sector (static label) | by ticker (grouped) |

## Non-negotiable rules

| Rule | Why |
|---|---|
| Universe: Swiss stocks only (SIX primary listings, SPI universe) for every evaluation; non-Swiss stocks only as v4 pretraining data. | User decisions 2026-10-09 and 2026-10-10. |
| Features use only data up to the feature date; targets only data after it (except `log_mcap`, `sector`, which are contemporaneous/static and therefore split by ticker). | No look-ahead leakage. |
| Time splits: train ≤ 2021-12-31, validation 2022-01-01..2023-12-31, test ≥ 2024-01-01. Fit scalers/winsorization on train only. Tune on validation; score test once. | Honest out-of-sample numbers. |
| Headline protocol (v6, user decision 2026-10-17): static targets (`sector`, `log_mcap`) are scored one row per stock with repeated stratified 5-fold CV over all eligible stocks (10 repeats, inner CV for hyper-parameters, stock-bootstrap intervals); forward targets are reported against a persistence model and their split-half measurement reliability (`static_cv.py`, `ceilings.py`; the reliability is not an attainable bound, so no "share of ceiling"). Intervals bootstrap stocks for the repeat-averaged metric. Sector correlation matrices may use all stocks' unlabeled returns (market-adjusted: `knn_resid`, `knn_auto`); labels only from training folds. Forward-target add-on experiments live in their own result files (`forward_v7.py`, `forward_kalman.py`, `forward_minirocket.py` → `results/forward_*.csv`), selected on validation, test read once. v4/v5 rows need `EF_FP_DIR` (Kaggle fingerprint outputs). The per-observation v1-v5 benchmark stays unchanged below it. | The 32-stock test split had seed noise (±0.05) larger than every effect tried; scores need a ceiling to be read. |
| Every result number in README comes from a committed file under `results/`. Never invent or round-trip numbers by hand. | Evidence. |
| Downloading the free data above and training on CPU is approved (user decision 2026-10-09). Raw/processed data stays git-ignored under `data/`. | User approval. |
| v9 European sector references (user decision 2026-10-17, option 5): `sector_eu.py` may use non-Swiss, non-US stocks from `data/pretrain/` (data ≤ 2021-12-31, never a `.SW` ticker) as labelled kNN references for Swiss query stocks. They are never evaluated, and the result lives only in `results/sector_eu.csv`; every other row stays Swiss-label-only. v9 add-on files: `forward_v9_beta.csv`, `forward_v9_vol.csv`, `static_v9_mcap.csv`, `static_v9_sector.csv`, `sector_eu.csv`. | The user approved all five improvement options ("hepsini"); the cross-market idea is kept apart so the headline stays a Swiss-only benchmark. |
| No notes in this repo: only `AGENTS.md` and `README.md`. Plans, reports and decisions go to ProjectOS `10-Projects/equity-fingerprint/`, rules to Jev-Mem `Notes/equity-fingerprint/`. | User rule. |

## Commands (from repo root)

```bash
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.data      # download + cache data into data/
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.run       # features, targets, models -> results/
```
