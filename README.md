# Equity Fingerprint — Inferring Equity Metadata from Market Data

A stock's daily price and volume history is a fingerprint: how much it moves, how it co-moves with the market and with other stocks, how much it trades. This project asks how much of a Swiss stock's metadata can be read back from that fingerprint alone: its future volatility, beta, liquidity, its market capitalisation and its sector. It compares a deliberately simple feature set against an engineered one, with a linear model and a gradient boosted tree model, under a leakage-controlled time / ticker split.

## Data

- Universe: SIX Swiss Exchange primary listings (`Primary listing == TRUE` in the SIX equity issuer list), 231 tickers as `<Symbol>.SW`. Of these, 221 returned prices from yfinance and 213 are quoted in CHF; the 207 with at least 252 prior trading rows enter the experiments (`results/run_info.json`).
- Prices: yfinance daily OHLCV, 2010-01-04 to 2026-10-07 (downloaded 2026-10-09); market index `^SSHI` (SPI) from 2013-03-05.
- Labels: yfinance `Ticker.info` snapshot (`sector`, `sharesOutstanding`), taken at download time.
- Cleaning: returns are log changes of `adj_close`; a daily |log return| above 0.7 is treated as an unadjusted corporate-action artifact and set to 0; missing days inside a listing span are forward filled; tickers not quoted in CHF are excluded. Counts are in `results/run_info.json`.

## Targets

| Target | Type | Definition | Split |
|---|---|---|---|
| `vol_1m` | regression | annualized realized vol of daily log returns over the next 21 trading days (modelled as log) | time |
| `beta_1y` | regression | beta vs `^SSHI` over the next 252 trading days | time |
| `log_mcap` | regression | log(close × sharesOutstanding snapshot) at the feature date | by ticker |
| `liquidity_1m` | regression | log(1 + mean daily traded value, close × volume in CHF) over the next 21 trading days | time |
| `sector` | classification | yfinance sector (static label); classes with fewer than 5 tickers are dropped, not merged | by ticker |

## Method

Observations: one per (ticker, month-end trading date) from 2014-03 on, requiring at least 252 prior trading rows. Features use data up to and including the feature date only.

Feature sets:

- `raw` (4 features): 21-day return, 21-day close-to-close volatility, log mean volume 21d, log close.
- `fe` (superset of `raw`, 39 features): returns over 5/21/63/126/252 days; volatility 21/63/252d; Parkinson and Garman-Klass volatility 21/63d; downside volatility, skewness and kurtosis 63d; max drawdown 252d; beta, correlation and idiosyncratic volatility vs `^SSHI` over 63/252d; Amihud illiquidity 21/63d; log traded value 21/63/252d; volume trend 21d/252d; share of zero-volume and zero-return days 63d; days of history (capped); and 252-day correlations of a stock's returns with the first 6 principal components of daily returns (return structure, aimed at the sector target).

Models, per target: `baseline` (train mean, or majority class / class priors for sector), `linear_raw`, `linear_fe` (Ridge; multinomial logistic regression for sector; skewed features log-transformed), `lgbm_raw`, `lgbm_fe` (LightGBM). Ridge alpha / logistic C are chosen on validation; LightGBM uses a small grid (num_leaves × min_child_samples) with early stopping, both on validation.

Splits:

- Time targets (`vol_1m`, `beta_1y`, `liquidity_1m`): train ≤ 2021-12-31, validation 2022-01-01..2023-12-31, test ≥ 2024-01-01. Train and validation observations whose forward window ends after their split are dropped; test needs a complete forward window.
- Grouped targets (`log_mcap`, `sector`): 70/15/15 by ticker, seed 42, stratified by sector; all dates of a ticker stay in one split.

Leakage controls: features see only data up to the feature date, targets only later data (except `log_mcap`/`sector`, which are contemporaneous/static and therefore split by ticker). Winsorization (1%/99%), log scale, imputation and standardization are fitted on train only. The PCA weights are fitted on training-period returns (≤ 2021-12-31) only, and for the ticker-grouped targets on training tickers only. Hyper-parameters are tuned on validation; the test split is scored once.

## Results

<!-- results:start -->
<!-- results:end -->

## Limitations

- Survivorship bias: the universe is today's SIX listing; delisted stocks are missing, so volatility, liquidity and drawdown histories are those of survivors.
- `sector` and `sharesOutstanding` are single snapshots from download time; `log_mcap` multiplies a historical split-adjusted close with today's share count, so it ignores past issuance and buybacks, and `sector` is assumed constant over time.
- yfinance is unofficial and for personal/research use; its prices contain errors (unadjusted corporate actions, bad ticks), handled only by the simple cleaning above.
- Each ticker contributes many correlated monthly observations; the grouped-target test sets contain few tickers, so those scores are noisy. Sector metrics are observation-level, and the classes are imbalanced (hence macro-F1).
- Principal components for the grouped targets are fitted on training tickers, whose own returns enter their PC factors; held-out tickers' loadings do not have that self-contribution.
- `vol_1m` windows with zero realized volatility are dropped (count in `results/run_info.json`). QLIKE for `vol_1m` uses the log-normal mean variance forecast `exp(2·pred + 2·s²)` with `s²` the validation residual variance.

## How to run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.data     # download + cache data/ (git-ignored)
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.run      # features, targets, models -> results/
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.report   # rebuild the results block above
```

Set `EF_DATA_DIR` to use a data folder other than `./data`.
