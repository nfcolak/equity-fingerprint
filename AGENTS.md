# Equity Fingerprint

Inferring equity metadata from market data. Predict five characteristics of Swiss listed stocks (SIX primary listings, SPI universe) from daily price and volume alone, and show how feature engineering improves linear regression and gradient boosted trees. Portfolio project for quant / data science roles in Switzerland.

## Tech stack

| Item | Value |
|---|---|
| Language | Python 3.12, project venv `.venv/` (run with `env -u PYTHONPATH .venv/bin/python`) |
| Libraries | pandas, numpy, scikit-learn, LightGBM, SHAP, matplotlib, pyarrow, yfinance |
| Code | `src/equity_fingerprint/` (package), entry points run as `python -m equity_fingerprint.<module>` with `PYTHONPATH=src` |

## Data sources (all free)

| Source | Use |
|---|---|
| SIX issuer list `https://www.six-group.com/sheldon/equity_issuers/v1/equity_issuers.csv` (`;`-separated, latin-1) | Universe: rows with `Primary listing == TRUE`; yfinance ticker = `<Symbol>.SW` |
| yfinance daily OHLCV | Stock prices/volume; market index `^SSHI` (SPI, from 2013-03) and `^SSMI` (SMI) |
| yfinance `Ticker.info` snapshot | Labels: `sector`, `industry`, `sharesOutstanding`, `marketCap` |

yfinance is unofficial and for personal/research use. The universe is today's listing (survivorship bias); say so in the README.

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
| Features use only data up to the feature date; targets only data after it (except `log_mcap`, `sector`, which are contemporaneous/static and therefore split by ticker). | No look-ahead leakage. |
| Time splits: train ≤ 2021-12-31, validation 2022-01-01..2023-12-31, test ≥ 2024-01-01. Fit scalers/winsorization on train only. Tune on validation; score test once. | Honest out-of-sample numbers. |
| Every result number in README comes from a committed file under `results/`. Never invent or round-trip numbers by hand. | Evidence. |
| Downloading the free data above and training on CPU is approved (user decision 2026-10-09). Raw/processed data stays git-ignored under `data/`. | User approval. |
| No notes in this repo: only `AGENTS.md` and `README.md`. Plans, reports and decisions go to ProjectOS `10-Projects/equity-fingerprint/`, rules to Jev-Mem `Notes/equity-fingerprint/`. | User rule. |

## Commands (from repo root)

```bash
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.data      # download + cache data into data/
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.run       # features, targets, models -> results/
```
