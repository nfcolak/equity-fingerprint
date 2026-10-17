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
Test split (time targets: observations from 2024-01-01 on; grouped targets: held-out tickers). Main metric: R² for the four regression targets (on the log scale for `vol_1m`, `log_mcap`, `liquidity_1m`), macro-F1 for `sector`. Source: `results/metrics.csv`.

| Target | Metric | Test obs | baseline | linear_raw | linear_fe | lgbm_raw | lgbm_fe |
|---|---|---:|---:|---:|---:|---:|---:|
| `vol_1m` | R² | 6473 | -0.047 | 0.490 | 0.609 | 0.520 | 0.624 |
| `beta_1y` | R² | 4240 | -0.002 | 0.429 | 0.632 | 0.494 | 0.603 |
| `log_mcap` | R² | 4365 | -0.122 | 0.808 | 0.762 | 0.690 | 0.689 |
| `liquidity_1m` | R² | 6493 | -0.001 | 0.933 | 0.950 | 0.939 | 0.957 |
| `sector` | macro-F1 | 4214 | 0.044 | 0.104 | 0.313 | 0.088 | 0.259 |

What feature engineering changed (test metric, `fe` minus `raw`; R² for regression, macro-F1 for sector): linear: `vol_1m` +0.119, `beta_1y` +0.202, `log_mcap` -0.046, `liquidity_1m` +0.017, `sector` +0.209; lgbm: `vol_1m` +0.104, `beta_1y` +0.108, `log_mcap` -0.001, `liquidity_1m` +0.018, `sector` +0.171.

Run 2026-10-17 (code at commit `0f5692a`): 207 tickers, 28016 month-end observations (2014-03-31 to 2026-09-30). Observations per target (train / val / test): `vol_1m` 16326 / 4326 / 6473; `beta_1y` 14272 / 2141 / 4240; `log_mcap` 18913 / 4587 / 4365; `liquidity_1m` 16385 / 4336 / 6493; `sector` 18762 / 4436 / 4214.

Sector classes dropped (fewer than 5 tickers): Utilities (4).

SHAP summaries of `lgbm_fe` on a test sample:

- [`vol_1m`](results/shap_vol_1m.png)
- [`beta_1y`](results/shap_beta_1y.png)
- [`log_mcap`](results/shap_log_mcap.png)
- [`liquidity_1m`](results/shap_liquidity_1m.png)
- [`sector`](results/shap_sector.png)

### v2: sector and market cap

Added `linear_fe2` / `lgbm_fe2` (features `fe2`) for the two grouped targets: the `fe` set plus rolling 252-day correlation and beta against equal-weight sector return indices (built from training-split tickers only, leave-one-out for a training ticker's own sector) and cross-sectional size proxies (log traded value minus the date median, and within-date ranks of log traded value and log close). v1 models and numbers are unchanged. Test split, ticker split seed 42. Source: `results/metrics.csv`.

| Target | Metric | baseline | linear_raw | linear_fe | linear_fe2 | lgbm_raw | lgbm_fe | lgbm_fe2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `sector` | macro-F1 | 0.044 | 0.104 | 0.313 | 0.323 | 0.088 | 0.259 | 0.238 |
| `sector` | accuracy | 0.248 | 0.244 | 0.430 | 0.433 | 0.287 | 0.389 | 0.370 |
| `log_mcap` | R² | -0.122 | 0.808 | 0.762 | 0.765 | 0.690 | 0.689 | 0.700 |
| `log_mcap` | MAE | 1.614 | 0.643 | 0.718 | 0.709 | 0.823 | 0.767 | 0.749 |

Sector at ticker level (class probabilities averaged over a ticker's observations, then argmax). Tickers: val / test = 32 / 32.

| Model | Val accuracy | Val macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| baseline | 0.250 | 0.044 | 0.250 | 0.044 |
| linear_raw | 0.312 | 0.125 | 0.219 | 0.066 |
| linear_fe | 0.594 | 0.431 | 0.594 | 0.414 |
| linear_fe2 | 0.625 | 0.453 | 0.594 | 0.414 |
| lgbm_raw | 0.250 | 0.071 | 0.281 | 0.083 |
| lgbm_fe | 0.594 | 0.300 | 0.500 | 0.282 |
| lgbm_fe2 | 0.594 | 0.300 | 0.500 | 0.284 |

Robustness over ticker-split seeds 1, 2, 3, 4, 42 (same 70/15/15 stratified split, PCA and sector indices refitted per seed, tuned on validation, test scored once per seed). Test metric, mean ± sd over 5 seeds. Test tickers per seed: `log_mcap` [33], `sector` [32]. Source: `results/robustness.csv`.

| Model | `log_mcap` R² | `sector` macro-F1 |
|---|---:|---:|
| baseline | -0.072 ± 0.067 | 0.046 ± 0.002 |
| linear_raw | 0.809 ± 0.036 | 0.121 ± 0.010 |
| linear_fe | 0.826 ± 0.045 | 0.290 ± 0.025 |
| linear_fe2 | 0.828 ± 0.043 | 0.318 ± 0.030 |
| lgbm_raw | 0.792 ± 0.071 | 0.091 ± 0.011 |
| lgbm_fe | 0.811 ± 0.071 | 0.224 ± 0.031 |
| lgbm_fe2 | 0.815 ± 0.067 | 0.229 ± 0.022 |

Test change from `fe` to `fe2` at seed 42: `sector` macro-F1 linear +0.010, lgbm -0.021; `log_mcap` R² linear +0.003, lgbm +0.011. Mean paired change over seeds (sd): `sector` macro-F1 linear +0.028 (0.019), lgbm +0.005 (0.015); `log_mcap` R² linear +0.001 (0.006), lgbm +0.003 (0.005).

The v1 `log_mcap` gap (`linear_fe` minus `linear_raw`, test R²) is -0.046 at seed 42; across the 5 seeds the paired gap has mean +0.017 and sd 0.037 (range -0.046 to +0.052), so the mean gap is within one seed-to-seed sd. The seed-to-seed sd of `linear_raw` test R² itself is 0.036.

Full pipeline run time (all five targets, v2 models and the 5-seed robustness reruns, CPU): 321.6 min. Ticker-level `n_obs` in `metrics.csv` counts tickers.

SHAP summaries of `lgbm_fe2` on a test sample:

- [`sector`](results/shap_sector_fe2.png)
- [`log_mcap`](results/shap_log_mcap_fe2.png)

### v3: co-movement profile for sector, HAR/GARCH for volatility

Added models next to v1/v2 (v1/v2 rows are unchanged). For `vol_1m`: `linear_har` / `lgbm_har` use only the three HAR inputs (log realised vol over the last 1, 5 and 22 trading days); `linear_fe3` / `lgbm_fe3` add the HAR inputs, a GJR-GARCH(1,1) 21-day mean volatility forecast and its persistence (Gaussian MLE per ticker on returns up to 2021-12-31, pooled parameters for tickers with fewer than 500 such returns) and SPI volatility state (log realised vol over 21 and 252 days and their difference) to the `fe` set. Test split. Source: `results/metrics.csv`.

`vol_1m` (R² on log vol; QLIKE on variance, lower is better):

| Model | R² | QLIKE |
|---|---:|---:|
| baseline | -0.047 | 1.055 |
| linear_fe | 0.609 | 0.368 |
| lgbm_fe | 0.624 | 0.346 |
| linear_har | 0.495 | 0.733 |
| lgbm_har | 0.494 | 0.549 |
| linear_fe3 | 0.617 | 0.351 |
| lgbm_fe3 | 0.610 | 0.355 |

`sector` (ticker split seed 42; `ticker_macro_f1` averages class probabilities per ticker):

| Model | macro-F1 | ticker macro-F1 |
|---|---:|---:|
| linear_fe | 0.313 | 0.414 |
| linear_fe2 | 0.323 | 0.414 |
| linear_fe3 | 0.335 | 0.399 |
| lgbm_fe3 | 0.273 | 0.288 |

`sector` robustness: test macro-F1, mean ± sd over ticker-split seeds. Source: `results/robustness.csv`.

| Model | macro-F1 | Seeds |
|---|---:|---:|
| linear_fe | 0.290 ± 0.025 | 5 |
| linear_fe2 | 0.318 ± 0.030 | 5 |
| linear_fe3 | 0.353 ± 0.031 | 5 |
| lgbm_fe | 0.224 ± 0.031 | 5 |
| lgbm_fe2 | 0.229 ± 0.022 | 5 |
| lgbm_fe3 | 0.282 ± 0.015 | 5 |

### v4: fingerprint encoder (self-supervised, pretrained outside Switzerland)

A Transformer encoder reads the last 252 trading days of a stock (log return, high-low range, overnight gap, relative volume, market return, zero-volume flag) and is pretrained without Swiss data: masked patch modelling, contrastive views and a co-movement objective on non-Swiss stocks (Europe and US), dates up to 2021-12-31 only. Arm `ssl` is purely self-supervised; arm `ssl_aux` adds non-Swiss sector and size labels as auxiliary heads. `linear_/lgbm_fp` use the frozen `ssl` embedding (128 dims), `*_fpa` the frozen `ssl_aux` embedding, `lgbm_fefpa` the `fe` features plus the `ssl_aux` embedding; `fpnet` / `fpnet_a` fine-tune the whole encoder (from `ssl` / `ssl_aux`) on the Swiss train split of each target. Test split, ticker split seed 42. Source: `results/metrics.csv`.

- Arm `ssl`: pretrained on 4314 non-Swiss tickers, data up to 2021-12-31, 30000 steps (checkpoint sha256 `a2563d957806`).
- Arm `ssl_aux`: pretrained on 4314 non-Swiss tickers, data up to 2021-12-31, 30000 steps (checkpoint sha256 `77c6de6d0b19`).

`vol_1m` (test, 6473 observations):

| Metric | best v1-v3 by validation (linear_fe3) | linear_fp | lgbm_fp | linear_fpa | lgbm_fpa | lgbm_fefpa | fpnet | fpnet_a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R² | 0.617 | 0.593 | 0.593 | 0.605 | 0.600 | 0.624 | 0.611 | 0.605 |
| MAE | 0.288 | 0.299 | 0.298 | 0.293 | 0.295 | 0.283 | 0.292 | 0.293 |
| QLIKE | 0.351 | 0.390 | 0.378 | 0.373 | 0.363 | 0.340 | 0.334 | 0.341 |

`beta_1y` (test, 4240 observations):

| Metric | best v1-v3 by validation (linear_fe) | linear_fp | lgbm_fp | linear_fpa | lgbm_fpa | lgbm_fefpa | fpnet | fpnet_a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R² | 0.632 | 0.635 | 0.608 | 0.627 | 0.627 | 0.636 | 0.606 | 0.630 |
| MAE | 0.206 | 0.208 | 0.217 | 0.209 | 0.209 | 0.206 | 0.213 | 0.210 |

`log_mcap` (test, 4365 observations):

| Metric | best v1-v3 by validation (lgbm_fe) | linear_fp | lgbm_fp | linear_fpa | lgbm_fpa | lgbm_fefpa | fpnet | fpnet_a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R² | 0.689 | 0.534 | 0.502 | 0.594 | 0.504 | 0.682 | 0.628 | 0.565 |
| MAE | 0.767 | 0.947 | 0.912 | 0.906 | 0.925 | 0.759 | 0.857 | 0.898 |

`liquidity_1m` (test, 6493 observations):

| Metric | best v1-v3 by validation (lgbm_fe) | linear_fp | lgbm_fp | linear_fpa | lgbm_fpa | lgbm_fefpa | fpnet | fpnet_a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R² | 0.957 | 0.847 | 0.858 | 0.866 | 0.863 | 0.956 | 0.893 | 0.901 |
| MAE | 0.349 | 0.848 | 0.825 | 0.788 | 0.802 | 0.357 | 0.684 | 0.656 |

`sector` (test, 4214 observations):

| Metric | best v1-v3 by validation (linear_fe3) | linear_fp | lgbm_fp | linear_fpa | lgbm_fpa | lgbm_fefpa | fpnet | fpnet_a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| macro-F1 | 0.335 | 0.222 | 0.208 | 0.223 | 0.212 | 0.289 | 0.186 | 0.229 |
| accuracy | 0.431 | 0.321 | 0.346 | 0.317 | 0.352 | 0.401 | 0.351 | 0.349 |
| ticker macro-F1 | 0.399 | 0.221 | 0.265 | 0.254 | 0.258 | 0.356 | 0.212 | 0.240 |

`sector` robustness: test macro-F1, mean ± sd over ticker-split seeds, and the mean paired change against `linear_fe3`. Source: `results/robustness.csv`.

| Model | macro-F1 | Seeds | Δ vs linear_fe3 |
|---|---:|---:|---:|
| linear_fe3 | 0.353 ± 0.031 | 5 | n/a |
| lgbm_fpa | 0.181 ± 0.027 | 5 | -0.173 |
| lgbm_fefpa | 0.247 ± 0.045 | 5 | -0.106 |
| fpnet_a | 0.206 ± 0.047 | 5 | -0.147 |

On the test split, with the best model of each family chosen by validation (main metric: R² for regression, macro-F1 for `sector`), the best v4 model against the best v1-v3 model is: `vol_1m` lgbm_fefpa 0.624 vs linear_fe3 0.617 (+0.007, v4 better); `beta_1y` fpnet_a 0.630 vs linear_fe 0.632 (-0.002, v4 not better); `log_mcap` lgbm_fefpa 0.682 vs lgbm_fe 0.689 (-0.007, v4 not better); `liquidity_1m` lgbm_fefpa 0.956 vs lgbm_fe 0.957 (-0.001, v4 not better); `sector` lgbm_fefpa 0.289 vs linear_fe3 0.335 (-0.046, v4 not better). The best v4 model is better on 1 of 5 targets, so v4 does not beat the v1-v3 benchmark on every target. Over the 5 ticker-split seeds (1, 2, 3, 4, 42), the best v4 `sector` model by mean test macro-F1 is lgbm_fefpa (0.247 ± 0.045) and the best v1-v3 model is linear_fe3 (0.353 ± 0.031); the paired mean difference is -0.106 (v4 higher in 0 of 5 seeds), so the sector gain of v4 is not positive on average. Provenance: pretrained on 4314 non-Swiss tickers with data up to 2021-12-31, 30000 steps per arm. Checksums of all Kaggle outputs are in `results/run_info.json`.

### v5: far positives for the fingerprint encoder, same-date co-movement profile for sector

Same Swiss evaluation as v4; new rows only. `linear_/lgbm_fpf` use the frozen `ssl_far` embedding, `*_fpfa` the frozen `ssl_aux_far` embedding and `lgbm_fefpfa` the `fe` features plus the `ssl_aux_far` embedding (arms pretrained with far positives: the two contrastive views are the same stock 252-756 trading days apart, so they never overlap). For `sector` only, `linear_/lgbm_fpprof` use same-date co-movement-profile features against the training tickers and `linear_/lgbm_fe3fpprof` add them to the `fe3` set. Test split, ticker split seed 42. Source: `results/metrics.csv`.

- Arm `ssl_far`: pretrained on 4314 non-Swiss tickers, data up to 2021-12-31, 30000 steps (checkpoint sha256 `21278a17ea93`).
- Arm `ssl_aux_far`: pretrained on 4314 non-Swiss tickers, data up to 2021-12-31, 30000 steps (checkpoint sha256 `de1d8cb5d6a6`).

`vol_1m` (test, 6473 observations):

| Metric | best v1-v4 by validation (linear_fe3) | linear_fpf | lgbm_fpf | linear_fpfa | lgbm_fpfa | lgbm_fefpfa |
|---|---:|---:|---:|---:|---:|---:|
| R² | 0.617 | 0.590 | 0.599 | 0.597 | 0.596 | 0.630 |
| MAE | 0.288 | 0.300 | 0.297 | 0.295 | 0.296 | 0.282 |
| QLIKE | 0.351 | 0.376 | 0.352 | 0.356 | 0.344 | 0.328 |

`beta_1y` (test, 4240 observations):

| Metric | best v1-v4 by validation (fpnet_a) | linear_fpf | lgbm_fpf | linear_fpfa | lgbm_fpfa | lgbm_fefpfa |
|---|---:|---:|---:|---:|---:|---:|
| R² | 0.630 | 0.628 | 0.626 | 0.623 | 0.614 | 0.632 |
| MAE | 0.210 | 0.209 | 0.209 | 0.210 | 0.213 | 0.206 |

`log_mcap` (test, 4365 observations):

| Metric | best v1-v4 by validation (lgbm_fefpa) | linear_fpf | lgbm_fpf | linear_fpfa | lgbm_fpfa | lgbm_fefpfa |
|---|---:|---:|---:|---:|---:|---:|
| R² | 0.682 | 0.553 | 0.561 | 0.513 | 0.603 | 0.668 |
| MAE | 0.759 | 0.929 | 0.880 | 0.927 | 0.872 | 0.776 |

`liquidity_1m` (test, 6493 observations):

| Metric | best v1-v4 by validation (lgbm_fefpa) | linear_fpf | lgbm_fpf | linear_fpfa | lgbm_fpfa | lgbm_fefpfa |
|---|---:|---:|---:|---:|---:|---:|
| R² | 0.956 | 0.829 | 0.880 | 0.867 | 0.868 | 0.957 |
| MAE | 0.357 | 0.935 | 0.748 | 0.776 | 0.775 | 0.350 |

`sector` (test, 4214 observations):

| Metric | best v1-v4 by validation (linear_fe3) | linear_fpf | lgbm_fpf | linear_fpfa | lgbm_fpfa | lgbm_fefpfa | linear_fpprof | lgbm_fpprof | linear_fe3fpprof | lgbm_fe3fpprof |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro-F1 | 0.335 | 0.222 | 0.207 | 0.230 | 0.216 | 0.282 | 0.136 | 0.132 | 0.334 | 0.249 |
| accuracy | 0.431 | 0.323 | 0.332 | 0.329 | 0.324 | 0.396 | 0.286 | 0.315 | 0.443 | 0.409 |
| ticker macro-F1 | 0.399 | 0.220 | 0.256 | 0.246 | 0.234 | 0.345 | 0.107 | 0.129 | 0.404 | 0.291 |

`sector` robustness: test macro-F1 per ticker-split seed, mean ± sd over seeds, and the mean paired change against `linear_fe3` (`lgbm_fefpa` is the best v4 model by mean test macro-F1). Source: `results/robustness.csv`.

| Model | seed 1 | seed 2 | seed 3 | seed 4 | seed 42 | mean ± sd | Δ vs linear_fe3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| linear_fe3 | 0.314 | 0.366 | 0.394 | 0.358 | 0.335 | 0.353 ± 0.031 | n/a |
| lgbm_fefpa | 0.257 | 0.236 | 0.278 | 0.175 | 0.289 | 0.247 ± 0.045 | -0.106 |
| linear_fe3fpprof | 0.303 | 0.350 | 0.379 | 0.335 | 0.334 | 0.340 ± 0.028 | -0.013 |
| lgbm_fe3fpprof | 0.280 | 0.278 | 0.341 | 0.263 | 0.249 | 0.282 ± 0.035 | -0.071 |

On the test split, with the best model of each family chosen by validation (main metric: R² for regression, macro-F1 for `sector`), the best v5 model against the best v1-v4 model is: `vol_1m` lgbm_fefpfa 0.630 vs linear_fe3 0.617 (+0.013, v5 better); `beta_1y` lgbm_fefpfa 0.632 vs fpnet_a 0.630 (+0.002, v5 better); `log_mcap` lgbm_fefpfa 0.668 vs lgbm_fefpa 0.682 (-0.014, v5 not better); `liquidity_1m` lgbm_fefpfa 0.957 vs lgbm_fefpa 0.956 (+0.001, v5 better); `sector` linear_fe3fpprof 0.334 vs linear_fe3 0.335 (-0.000, v5 not better). The best v5 model is better on 3 of 5 targets, so v5 does not beat the v1-v4 benchmark on every target.
Robustness: over the 5 ticker-split seeds the paired mean test macro-F1 difference of linear_fe3fpprof (0.340) against linear_fe3 (0.353) is -0.013 (higher in 0 of 5 seeds, seed-to-seed sd of the difference 0.008).

All metrics (validation and test, MAE/RMSE/QLIKE, accuracy, log loss): [`results/metrics.csv`](results/metrics.csv); feature gain: [`results/feature_importance.csv`](results/feature_importance.csv).
<!-- results:end -->

## Limitations

- Survivorship bias: the universe is today's SIX listing; delisted stocks are missing, so volatility, liquidity and drawdown histories are those of survivors.
- `sector` and `sharesOutstanding` are single snapshots from download time; `log_mcap` multiplies a historical split-adjusted close with today's share count, so it ignores past issuance and buybacks, and `sector` is assumed constant over time.
- yfinance is unofficial and for personal/research use; its prices contain errors (unadjusted corporate actions, bad ticks), handled only by the simple cleaning above.
- Each ticker contributes many correlated monthly observations; the grouped-target test sets contain few tickers, so those scores are noisy. Sector metrics are observation-level, and the classes are imbalanced (hence macro-F1).
- Principal components for the grouped targets are fitted on training tickers, whose own returns enter their PC factors; held-out tickers' loadings do not have that self-contribution.
- `vol_1m` windows with zero realized volatility are dropped (count in `results/run_info.json`). QLIKE for `vol_1m` uses the log-normal mean variance forecast `exp(2·pred + 2·s²)` with `s²` the validation residual variance.
- The v5 variants exist in code: pretraining with far contrastive positives (two views of the same stock at least a year apart) and a same-date comove-profile sector feature set; their benchmark rows will appear after the Kaggle pretraining run.

## How to run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.data     # download + cache data/ (git-ignored)
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.run      # features, targets, models -> results/
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.report   # rebuild the results block above
```

Set `EF_DATA_DIR` to use a data folder other than `./data`.
