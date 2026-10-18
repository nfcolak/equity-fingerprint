# Equity Fingerprint — Inferring Equity Metadata from Market Data

A stock's daily price and volume history is a fingerprint: how much it moves, how it co-moves with the market and with other stocks, how much it trades. This project asks how much of a Swiss stock's metadata can be read back from that fingerprint alone: its future volatility, beta, liquidity, its market capitalisation and its sector. It compares a deliberately simple feature set against an engineered one, with a linear model and a gradient boosted tree model, under a leakage-controlled time / ticker split.

How to read the results: the headline section scores the static targets (`sector`, `log_mcap`) once per stock with repeated issuer-grouped cross-validation over all Swiss stocks, and reads the forward targets (`vol_1m`, `beta_1y`, `liquidity_1m`) against a persistence model and the target's own noise ceiling. The per-observation benchmark of v1-v5, including the self-supervised fingerprint encoder (v4, v5), follows below it unchanged; the encoder did not beat the engineered features beyond noise. Later add-on rounds keep their own result files. v9 found one clear gain: `log_mcap` per-stock R² rises from 0.891 (ridge) to 0.922 with several past month-ends per training stock plus extra traded-value / index-review / SMI features (`ridge_md_ext (v9)` in the headline table, paired difference +0.031 [+0.014, +0.048]); its `vol_1m` and `beta_1y` gains over the previous best arms are small (+0.005 and +0.008 test R²) with intervals that reach or span zero. v10 added a longer Swiss training history, which raises `beta_1y` to 0.650 (+0.016 [+0.006, +0.027] over the same model without it, a headline row), and European reference stocks for `sector`, which add +0.024 macro-F1 over `knn_auto` with an interval spanning zero. This is consistent with the small number of Swiss stocks being the main limit for sector accuracy (descriptive, not a paired test): in the pooled benchmark the same kind of kNN scores macro-F1 0.766 on 850 European stocks against 0.579 on the Swiss ones.

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

Observations: one per (ticker, month-end trading date) from 2014-03 on, requiring at least 252 prior trading rows. Features use price and volume data up to and including the feature date only. Three fitted objects are estimated once on all training-period data (≤ 2021-12-31): the return principal components, the GJR-GARCH parameters and the v3 return-embedding basis; validation and test rows are out of sample for them, early training rows are not strictly so.

Feature sets:

- `raw` (4 features): 21-day return, 21-day close-to-close volatility, log mean volume 21d, log close.
- `fe` (superset of `raw`, 39 features): returns over 5/21/63/126/252 days; volatility 21/63/252d; Parkinson and Garman-Klass volatility 21/63d; downside volatility, skewness and kurtosis 63d; max drawdown 252d; beta, correlation and idiosyncratic volatility vs `^SSHI` over 63/252d; Amihud illiquidity 21/63d; log traded value 21/63/252d; volume trend 21d/252d; share of zero-volume and zero-return days 63d; days of history (capped); and 252-day correlations of a stock's returns with the first 6 principal components of daily returns (return structure, aimed at the sector target).

Models, per target: `baseline` (train mean, or majority class / class priors for sector), `linear_raw`, `linear_fe` (Ridge; multinomial logistic regression for sector; skewed features log-transformed), `lgbm_raw`, `lgbm_fe` (LightGBM). Ridge alpha / logistic C are chosen on validation; LightGBM uses a small grid (num_leaves × min_child_samples) with early stopping, both on validation.

Splits:

- Time targets (`vol_1m`, `beta_1y`, `liquidity_1m`): train ≤ 2021-12-31, validation 2022-01-01..2023-12-31, test ≥ 2024-01-01. Train and validation observations whose forward window ends after their split are dropped; test needs a complete forward window.
- Grouped targets (`log_mcap`, `sector`): 70/15/15 by ticker, seed 42, stratified by sector; all dates of a ticker stay in one split.

Leakage controls: features see only data up to the feature date, targets only later data (except `log_mcap`/`sector`, which are contemporaneous/static and therefore split by ticker). Winsorization (1%/99%), log scale, imputation and standardization are fitted on train only. The PCA weights are fitted on training-period returns (≤ 2021-12-31) only, and for the ticker-grouped targets on training tickers only. Hyper-parameters are tuned on validation; the test split is scored once per experiment (see Limitations for repeated reads across rounds).

## Results

<!-- results:start -->
### Headline (v6 protocol)

Static targets, one row per stock: repeated issuer-grouped stratified 5-fold cross-validation over all eligible Swiss stocks (10 repeats; share lines of one company, such as LISN/LISP, RO/ROP, SCHN/SCHP and UHR/UHRN, always stay in the same fold), hyper-parameters by inner CV on the training folds only. Mean ± sd over repeats; 95% interval from a stock bootstrap of the repeat-averaged metric. `sector` uses each stock's full daily price history; Utilities (4 stocks) is dropped because classes need at least 5 stocks. `knn` uses raw return correlations, `knn_resid` correlations of returns after removing the SPI market return, `knn_auto` lets inner CV choose among six market-adjusted correlation matrices and k (matrices use returns only, never labels). `log_mcap` is close × today's share count at the latest month-end (no historical share-count bias); `ridge_logtv21` uses only the 21-day traded value, which carries most of the signal. Source: `results/static_cv_grouped.csv`.

Earlier headline (per-ticker folds, `results/static_cv.csv`): `knn_auto` macro-F1 0.598 (now 0.578), `ridge` R² 0.892 (now 0.891). Per-ticker folds let a share line be predicted from its highly correlated sibling line, so the grouped numbers are the honest ones.

`sector` (203 stocks, 9 classes):

| Model | macro_f1 | weighted_f1 | accuracy | balanced_accuracy | macro_f1 95% interval |
|---|---:|---:|---:|---:|---:|
| baseline | 0.044 ± 0.000 | 0.097 ± 0.000 | 0.246 ± 0.000 | 0.111 ± 0.000 | [0.035, 0.052] |
| knn | 0.478 ± 0.034 | 0.567 ± 0.022 | 0.592 ± 0.017 | 0.445 ± 0.027 | [0.400, 0.533] |
| prof_logit | 0.427 ± 0.012 | 0.567 ± 0.012 | 0.622 ± 0.012 | 0.436 ± 0.011 | [0.377, 0.467] |
| femean_logit | 0.274 ± 0.015 | 0.377 ± 0.020 | 0.397 ± 0.019 | 0.278 ± 0.015 | [0.225, 0.317] |
| prof_femean_logit | 0.424 ± 0.023 | 0.557 ± 0.019 | 0.577 ± 0.017 | 0.429 ± 0.021 | [0.369, 0.466] |
| knn_resid | 0.577 ± 0.025 | 0.644 ± 0.015 | 0.655 ± 0.014 | 0.554 ± 0.023 | [0.492, 0.640] |
| knn_auto | 0.578 ± 0.017 | 0.658 ± 0.017 | 0.666 ± 0.019 | 0.573 ± 0.016 | [0.505, 0.631] |

`log_mcap` (204 stocks):

| Model | r2 | mae | r2 95% interval |
|---|---:|---:|---:|
| baseline | -0.009 ± 0.006 | 1.640 ± 0.006 | [-0.033, -0.007] |
| ridge | 0.891 ± 0.005 | 0.507 ± 0.011 | [0.859, 0.915] |
| ridge_logtv21 | 0.846 ± 0.003 | 0.616 ± 0.005 | [0.803, 0.877] |
| lgbm | 0.886 ± 0.005 | 0.537 ± 0.015 | [0.854, 0.907] |
| ridge_md_ext (v9) | 0.922 ± 0.003 | 0.440 ± 0.006 | [0.896, 0.940] |

`ridge_md_ext` (v9, same folds and protocol): ridge trained on several past month-ends of each training stock (1 to 12 dates, chosen by inner CV; converted to today by the price ratio) with extra price/volume features (index-review volume spike, traded-value structure, SMI co-movement). Paired difference to `ridge`: +0.031 R² [+0.014, +0.048]. It was picked after comparing about ten v9 arms on these same folds, so it carries a small selection optimism. Source: `results/static_v9_mcap_grouped.csv`.

Forward targets on the test split, read against two references: a persistence model (one feature, OLS fitted on train: past 21-day log vol, past 252-day beta, past 21-day log traded value) and the split-half reliability of the target (odd vs even days of the forward window, Spearman-Brown corrected). The reliability describes how noisy the target measurement is; it is not an attainable forecasting bound, because both halves share shocks that are unknowable at the forecast date. Best model = highest validation R² among the v1-v5 rows of `results/metrics.csv` (the add-on rounds below select separately). Skill = (R² − persistence R²) / (1 − persistence R²). Source: `results/ceilings.csv`, `results/metrics.csv`.

| Target | Best model (by validation) | Test R² | Persistence R² | Skill over persistence | Measurement reliability |
|---|---|---:|---:|---:|---:|
| `vol_1m` | lgbm_fefpfa | 0.630 | 0.426 | 0.35 | 0.807 |
| `beta_1y` | fpnet_a | 0.630 | 0.522 | 0.23 | 0.790 |
| `beta_1y` | avg_long2011 (v10) | 0.650 | 0.522 | 0.27 | 0.790 |
| `liquidity_1m` | lgbm_fefpfa | 0.957 | 0.949 | 0.16 | 0.992 |

`liquidity_1m` is almost fully explained by last month's traded value, so it is reported but is not a headline result.

`avg_long2011` (v10, user decision 2026-10-18): the engineered-feature linear/LightGBM average trained with extra Swiss training rows from 2011 (SMI as market before the SPI starts in 2013); validation and test rows are the same as above. It was the pre-specified primary comparison of the longer-history run, not picked by the best-by-validation rule: paired test Δ R² vs the same model without the extra rows +0.016 [+0.006, +0.027]. Source: `results/history_ext.csv`.

`beta_1y` with Welch slope-winsorised betas (stock return clipped to [-2m, 4m] around the market return m, 252-day and age-decayed) and linear/LightGBM averages. The arm marked ★ was chosen on validation before test was read; Δ is test R² minus `linear_fe` with a 95% stock-block bootstrap interval. Source: `results/forward_v7.csv`.

| Arm | Val R² | Test R² | Test Δ vs linear_fe | 95% interval |
|---|---:|---:|---:|---:|
| linear_fe | 0.662 | 0.632 | – | – |
| lgbm_fe | 0.659 | 0.603 | -0.029 | [-0.063, +0.002] |
| avg_fe | 0.672 | 0.635 | +0.003 | [-0.013, +0.018] |
| linear_fewelch | 0.670 | 0.639 | +0.007 | [-0.004, +0.019] |
| lgbm_fewelch | 0.659 | 0.624 | -0.008 | [-0.035, +0.019] |
| avg_fewelch ★ | 0.674 | 0.645 | +0.013 | [-0.001, +0.025] |

`beta_1y` with a filtered random-walk Kalman beta (noise ratio chosen by train-period likelihood), alone and with the Welch betas. The arm marked ★ was chosen on validation before test was read; Δ is test R² minus `linear_fe` with a 95% stock-block bootstrap interval. Source: `results/forward_kalman.csv`.

| Arm | Val R² | Test R² | Test Δ vs linear_fe | 95% interval |
|---|---:|---:|---:|---:|
| linear_fe | 0.662 | 0.632 | – | – |
| lgbm_fe | 0.659 | 0.603 | -0.029 | [-0.063, +0.002] |
| avg_fewelch | 0.674 | 0.645 | +0.013 | [-0.001, +0.025] |
| linear_fekalman | 0.666 | 0.630 | -0.001 | [-0.008, +0.004] |
| lgbm_fekalman | 0.685 | 0.600 | -0.032 | [-0.064, -0.002] |
| avg_fekalman ★ | 0.688 | 0.619 | -0.013 | [-0.036, +0.010] |
| linear_fewelchkalman | 0.673 | 0.639 | +0.007 | [-0.007, +0.020] |
| lgbm_fewelchkalman | 0.676 | 0.611 | -0.021 | [-0.047, +0.007] |
| avg_fewelchkalman | 0.684 | 0.638 | +0.006 | [-0.011, +0.024] |

`vol_1m` with MiniRocket features (84 fixed convolution kernels over the past 252 days of stock return, market return and traded value; biases fitted on training windows) with ridge, alone and with the engineered features. The arm marked ★ was chosen on validation before test was read; Δ is test R² minus `linear_fe` with a 95% stock-block bootstrap interval. Source: `results/forward_minirocket.csv`.

| Arm | Val R² | Test R² | Test Δ vs linear_fe | 95% interval |
|---|---:|---:|---:|---:|
| linear_fe | 0.629 | 0.609 | – | – |
| ridge_mr | 0.605 | 0.617 | +0.009 | [-0.003, +0.022] |
| ridge_mrfe ★ | 0.646 | 0.630 | +0.021 | [+0.016, +0.027] |
| lgbm_femrpca | 0.639 | 0.627 | +0.018 | [+0.010, +0.028] |
| avg_mrfe | 0.646 | 0.630 | +0.021 | [+0.016, +0.027] |

`beta_1y` with MiniRocket features (84 fixed convolution kernels over the past 252 days of stock return, market return and traded value; biases fitted on training windows) with ridge, alone and with the engineered features. The arm marked ★ was chosen on validation before test was read; Δ is test R² minus `linear_fe` with a 95% stock-block bootstrap interval. Source: `results/forward_minirocket.csv`.

| Arm | Val R² | Test R² | Test Δ vs linear_fe | 95% interval |
|---|---:|---:|---:|---:|
| linear_fe | 0.662 | 0.632 | – | – |
| ridge_mr | 0.569 | 0.566 | -0.066 | [-0.112, -0.015] |
| ridge_mrfe ★ | 0.676 | 0.653 | +0.021 | [+0.001, +0.045] |
| lgbm_femrpca | 0.669 | 0.627 | -0.005 | [-0.031, +0.020] |
| avg_mrfe | 0.676 | 0.653 | +0.021 | [+0.001, +0.045] |

### v9 add-on experiments

Each idea comes from a probe round (validation / training folds only) and was then run once under the protocols above. Forward files: every choice on validation, test read once, ★ = arm chosen on validation, Δ with a 95% stock-block bootstrap interval. Static files: same repeated CV as the headline, Δ = paired difference to the named reference (same folds), 95% stock-bootstrap interval.

`beta_1y`, residual target: the models predict the future beta minus a base beta (OLS, Welch, Vasicek-shrunk Welch, or a mix with a size-group prior) and add the base back. Source: `results/forward_v9_beta.csv`.

| Arm | Val R² | Test R² | Test Δ vs linear_fe [95%] | Test Δ vs avg_fewelch [95%] |
|---|---:|---:|---:|---:|
| linear_fe | 0.662 | 0.632 | – | -0.013 [-0.025, +0.001] |
| lgbm_fe | 0.659 | 0.603 | -0.029 [-0.063, +0.002] | -0.042 [-0.073, -0.012] |
| avg_fe | 0.672 | 0.635 | +0.003 [-0.013, +0.018] | -0.010 [-0.027, +0.006] |
| linear_fewelch | 0.670 | 0.639 | +0.007 [-0.004, +0.019] | -0.005 [-0.011, +0.001] |
| lgbm_fewelch | 0.659 | 0.624 | -0.008 [-0.035, +0.019] | -0.020 [-0.041, -0.001] |
| avg_fewelch | 0.674 | 0.645 | +0.013 [-0.001, +0.025] | – |
| linear_resols | 0.690 | 0.614 | -0.018 [-0.059, +0.012] | -0.030 [-0.068, -0.006] |
| lgbm_resols | 0.674 | 0.582 | -0.049 [-0.102, -0.011] | -0.062 [-0.120, -0.021] |
| avg_resols | 0.692 | 0.619 | -0.013 [-0.052, +0.013] | -0.026 [-0.063, -0.002] |
| linear_resw252 | 0.700 | 0.632 | +0.000 [-0.027, +0.027] | -0.012 [-0.029, +0.004] |
| lgbm_resw252 | 0.706 | 0.610 | -0.022 [-0.058, +0.011] | -0.035 [-0.063, -0.009] |
| avg_resw252 | 0.711 | 0.634 | +0.002 [-0.029, +0.029] | -0.011 [-0.032, +0.008] |
| linear_reswew | 0.694 | 0.618 | -0.014 [-0.046, +0.017] | -0.027 [-0.050, -0.006] |
| lgbm_reswew | 0.704 | 0.606 | -0.026 [-0.064, +0.013] | -0.038 [-0.069, -0.009] |
| avg_reswew | 0.710 | 0.627 | -0.005 [-0.037, +0.028] | -0.017 [-0.039, +0.005] |
| linear_resvck | 0.704 | 0.639 | +0.007 [-0.023, +0.038] | -0.006 [-0.033, +0.021] |
| lgbm_resvck | 0.710 | 0.617 | -0.015 [-0.061, +0.031] | -0.028 [-0.069, +0.012] |
| avg_resvck | 0.716 | 0.640 | +0.009 [-0.030, +0.046] | -0.004 [-0.038, +0.028] |
| linear_resmix | 0.702 | 0.645 | +0.013 [-0.008, +0.042] | +0.001 [-0.014, +0.020] |
| lgbm_resmix | 0.708 | 0.632 | -0.000 [-0.032, +0.033] | -0.013 [-0.039, +0.014] |
| avg_resmix ★ | 0.716 | 0.652 | +0.020 [-0.006, +0.049] | +0.008 [-0.011, +0.026] |
| linear_resmixhar | 0.701 | 0.643 | +0.011 [-0.012, +0.040] | -0.002 [-0.018, +0.017] |
| lgbm_resmixhar | 0.709 | 0.638 | +0.006 [-0.025, +0.039] | -0.006 [-0.030, +0.018] |
| avg_resmixhar | 0.714 | 0.652 | +0.020 [-0.006, +0.050] | +0.008 [-0.010, +0.026] |

`vol_1m` with new causal features (exponentially weighted vols, the same window one and two years earlier, month dummies, VIX and Euro Stoxx 50 vol), the HAR/GARCH set, and a stack (non-negative weights fitted on validation) of four arms. Source: `results/forward_v9_vol.csv`.

| Arm | Val R² | Test R² | Test Δ vs linear_fe [95%] | Test Δ vs ridge_mrfe [95%] |
|---|---:|---:|---:|---:|
| linear_fe | 0.629 | 0.609 | – | -0.021 [-0.027, -0.016] |
| lgbm_fe | 0.636 | 0.624 | +0.015 [+0.007, +0.024] | -0.006 [-0.015, +0.002] |
| linear_fehexp | 0.642 | 0.621 | +0.012 [+0.007, +0.018] | -0.009 [-0.014, -0.004] |
| linear_feseas | 0.644 | 0.614 | +0.005 [+0.002, +0.008] | -0.016 [-0.022, -0.012] |
| linear_femonth | 0.635 | 0.612 | +0.003 [-0.001, +0.006] | -0.018 [-0.025, -0.013] |
| linear_fevix | 0.638 | 0.613 | +0.004 [+0.002, +0.007] | -0.017 [-0.023, -0.011] |
| linear_feall | 0.657 | 0.628 | +0.020 [+0.013, +0.027] | -0.002 [-0.008, +0.004] |
| lgbm_feall | 0.662 | 0.617 | +0.009 [-0.001, +0.020] | -0.013 [-0.022, -0.004] |
| avg_feall | 0.671 | 0.634 | +0.025 [+0.018, +0.034] | +0.004 [-0.002, +0.010] |
| linear_fe3 | 0.638 | 0.617 | +0.008 [+0.003, +0.015] | -0.013 [-0.019, -0.008] |
| ridge_mrfe | 0.646 | 0.630 | +0.021 [+0.016, +0.027] | – |
| stack ★ | 0.672 | 0.635 | +0.027 [+0.019, +0.035] | +0.005 [-0.000, +0.011] |

`log_mcap` with extra price/volume features (index-review volume spike, traded-value structure, SMI co-movement), several past month-ends per training stock (`*_md`, converted to today by the price ratio) and TabPFN v2 (untuned) (204 stocks). Source: `results/static_v9_mcap.csv`.

| Model | r2 | mae | r2 95% interval | Δ r2 vs ridge [95%] |
|---|---:|---:|---:|---:|
| ridge | 0.892 ± 0.005 | 0.506 ± 0.009 | [0.862, 0.916] | – |
| lgbm | 0.886 ± 0.007 | 0.544 ± 0.018 | [0.854, 0.907] | -0.007 [-0.027, +0.015] |
| avg_ridge_lgbm | 0.898 ± 0.005 | 0.504 ± 0.009 | [0.871, 0.918] | +0.006 [-0.003, +0.016] |
| ridge_ext | 0.911 ± 0.005 | 0.476 ± 0.014 | [0.884, 0.931] | +0.018 [+0.002, +0.033] |
| lgbm_ext | 0.896 ± 0.005 | 0.517 ± 0.013 | [0.866, 0.917] | +0.004 [-0.016, +0.025] |
| ridge_md | 0.914 ± 0.002 | 0.457 ± 0.006 | [0.888, 0.934] | +0.022 [+0.013, +0.033] |
| ridge_md_ext | 0.924 ± 0.003 | 0.437 ± 0.005 | [0.898, 0.942] | +0.031 [+0.015, +0.047] |
| tabpfn | 0.910 ± 0.004 | 0.457 ± 0.008 | [0.884, 0.932] | +0.018 [+0.006, +0.032] |
| tabpfn_ext | 0.920 ± 0.003 | 0.441 ± 0.008 | [0.895, 0.938] | +0.027 [+0.013, +0.044] |
| avg_ridge_md_ext_tabpfn_ext | 0.926 ± 0.002 | 0.425 ± 0.005 | [0.903, 0.944] | +0.034 [+0.020, +0.050] |

`sector` kNN with class-prior-corrected votes (`*_prior`) and exponentially time-weighted residual correlations (`*_ew`) (203 stocks). Source: `results/static_v9_sector.csv`.

| Model | macro_f1 | weighted_f1 | accuracy | balanced_accuracy | macro_f1 95% interval | Δ macro_f1 vs knn_auto [95%] |
|---|---:|---:|---:|---:|---:|---:|
| knn_auto | 0.598 ± 0.024 | 0.665 ± 0.016 | 0.672 ± 0.016 | 0.589 ± 0.023 | [0.518, 0.652] | – |
| knn_auto_prior | 0.576 ± 0.022 | 0.651 ± 0.019 | 0.644 ± 0.022 | 0.592 ± 0.026 | [0.509, 0.629] | -0.022 [-0.048, +0.008] |
| knn_auto_ew | 0.585 ± 0.033 | 0.655 ± 0.017 | 0.665 ± 0.014 | 0.578 ± 0.034 | [0.508, 0.640] | -0.012 [-0.033, +0.007] |
| knn_auto_v9 | 0.575 ± 0.019 | 0.649 ± 0.017 | 0.644 ± 0.016 | 0.588 ± 0.021 | [0.507, 0.627] | -0.022 [-0.058, +0.017] |

`sector` with European labelled stocks (data ≤ 2021-12-31, never evaluated) as extra kNN references. `ch_le2021` is the Swiss-only control in the same ≤ 2021 space; `pooled_full` uses full-history Swiss similarities plus Swiss-European ones; `knn_auto` is copied from `results/static_cv.csv`. This uses non-Swiss labels at prediction time (user decision 2026-10-17) and is not part of the headline (203 stocks). Source: `results/sector_eu.csv`.

| Model | macro_f1 | weighted_f1 | accuracy | balanced_accuracy | macro_f1 95% interval | Δ macro_f1 vs ch_le2021 [95%] | Δ macro_f1 vs knn_resid [95%] | Δ macro_f1 vs knn_auto [95%] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| knn_auto | 0.598 ± 0.024 | 0.665 ± 0.016 | 0.672 ± 0.016 | 0.589 ± 0.023 | [0.518, 0.652] | – | – | – |
| ch_le2021 | 0.439 ± 0.020 | 0.530 ± 0.014 | 0.548 ± 0.013 | 0.429 ± 0.019 | [0.373, 0.497] | – | – | – |
| eu_only | 0.422 ± 0.013 | 0.499 ± 0.012 | 0.498 ± 0.013 | 0.402 ± 0.012 | [0.331, 0.498] | -0.017 [-0.121, +0.087] | – | – |
| pooled | 0.497 ± 0.015 | 0.568 ± 0.012 | 0.580 ± 0.013 | 0.468 ± 0.015 | [0.415, 0.555] | +0.058 [-0.005, +0.122] | – | – |
| pooled_full | 0.596 ± 0.024 | 0.660 ± 0.014 | 0.671 ± 0.011 | 0.563 ± 0.023 | [0.507, 0.659] | +0.157 [+0.075, +0.237] | +0.024 [-0.023, +0.069] | -0.001 [-0.081, +0.083] |
| pooled_full_prior | 0.616 ± 0.020 | 0.680 ± 0.010 | 0.684 ± 0.008 | 0.606 ± 0.019 | [0.541, 0.673] | +0.177 [+0.109, +0.243] | +0.044 [-0.008, +0.099] | +0.018 [-0.047, +0.084] |

### v10: European stocks, longer Swiss history

User decision 2026-10-18: European stocks (domicile matches the listing country, no Swiss issuer, cross-listings removed, prices in CHF) may be kNN references; Swiss stocks stay the only scored stocks except in the `pooled` rows. The first v10 run also used about 2,300 US stocks: as references they lowered the Swiss sector score, and pooled US/EU training rows did not help `vol_1m` or `beta_1y`, so the user removed the US stocks and the pooled forward training; the 2000-2009 history arm added nothing over 2011 and was removed too. Keeping only the European arm is therefore a choice made after seeing results. The test split was read in earlier rounds, so these test numbers are a second look. Foreign stocks come from today's index lists and listings (survivor universe).

`sector`, Swiss queries (203 stocks, issuer-grouped folds, 10 repeats). `ch_knn_auto` is the headline model on these folds; `ch_menu` the new recipe (daily or two-factor weekly residual correlations, class-prior-corrected votes) with Swiss references only; `ch_eu` adds European labelled references. Main comparison: `ch_eu` vs `ch_knn_auto`. Source: `results/multi_static.csv`.

| Arm | macro_f1 | 95% interval | Δ vs ch_knn_auto [95%] | Δ vs ch_menu [95%] |
|---|---:|---:|---:|---:|
| ch_knn_auto | 0.578 ± 0.017 | [0.505, 0.631] | – | – |
| ch_menu | 0.534 ± 0.024 | [0.467, 0.588] | -0.043 [-0.088, +0.010] | – |
| ch_eu (main) | 0.602 ± 0.014 | [0.520, 0.669] | +0.024 [-0.046, +0.099] | +0.067 [+0.005, +0.137] |

Pooled benchmark: every eligible stock (Swiss and European) is a query, issuer-grouped folds stratified by region and class, 10 repeats; scores per region. `sector`: kNN on two-factor weekly residual correlations. `log_mcap`: stocks whose close × share count is more than 0.25 log units from the reported market cap are dropped (class-specific share counts). Descriptive; not paired with the Swiss-only rows.

| Target | Model | Metric | CH | EU | ALL |
|---|---|---|---:|---:|---:|
| `sector` | all_knn | macro_f1 | 0.579 (203) | 0.766 (850) | 0.743 (1053) |
| `sector` | baseline | macro_f1 | 0.044 (203) | 0.033 (850) | 0.033 (1053) |
| `log_mcap` | all_ridge | r2 | 0.912 (204) | 0.920 (816) | 0.922 (1020) |
| `log_mcap` | all_lgbm | r2 | 0.914 (204) | 0.914 (816) | 0.918 (1020) |
| `log_mcap` | baseline | r2 | -0.155 (204) | -0.017 (816) | -0.002 (1020) |

Cells: mean over repeats (stocks).

Longer Swiss history: extra TRAIN rows before 2014-03 (`long2011`: the existing 2010+ cache with the SMI as market series before the SPI starts in 2013). Validation and test rows are identical to the control. Primary: `avg_long2011` vs `avg_ctrl` (now also a headline row). Source: `results/history_ext.csv`, `results/history_quality.csv`.

| Target | Model | Val R² | Test R² | Test Δ vs ctrl (same kind) [95%] |
|---|---|---:|---:|---:|
| `vol_1m` | linear_ctrl | 0.629 | 0.609 | – |
| `vol_1m` | lgbm_ctrl | 0.636 | 0.624 | – |
| `vol_1m` | avg_ctrl | 0.643 | 0.629 | – |
| `vol_1m` | linear_long2011 | 0.632 | 0.612 | +0.003 [+0.000, +0.006] |
| `vol_1m` | lgbm_long2011 | 0.641 | 0.627 | +0.004 [-0.001, +0.008] |
| `vol_1m` | avg_long2011 (primary) | 0.646 | 0.631 | +0.002 [-0.001, +0.005] |
| `beta_1y` | linear_ctrl | 0.662 | 0.632 | – |
| `beta_1y` | lgbm_ctrl | 0.659 | 0.603 | – |
| `beta_1y` | avg_ctrl | 0.672 | 0.635 | – |
| `beta_1y` | linear_long2011 | 0.683 | 0.636 | +0.004 [-0.006, +0.013] |
| `beta_1y` | lgbm_long2011 | 0.673 | 0.634 | +0.032 [+0.010, +0.054] |
| `beta_1y` | avg_long2011 (primary) | 0.688 | 0.650 | +0.016 [+0.006, +0.027] |

`sector` kNN on longer return histories (macro-F1, 10 repeats): knn_auto committed (ticker folds) 0.598; knn_auto_long long2011 (ticker folds) 0.603; knn_auto ctrl (issuer folds) 0.578; knn_auto_long long2011 (issuer folds) 0.588.

### Per-observation benchmark (v1-v5)

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
- Each ticker contributes many correlated monthly observations; the grouped-target test sets contain few tickers, so those scores are noisy. In that benchmark sector metrics are observation-level (the headline scores one row per stock), and the classes are imbalanced (hence macro-F1).
- Principal components for the grouped targets are fitted on training tickers, whose own returns enter their PC factors; held-out tickers' loadings do not have that self-contribution.
- `vol_1m` windows with zero realized volatility are dropped (count in `results/run_info.json`). QLIKE for `vol_1m` uses the log-normal mean variance forecast `exp(2·pred + 2·s²)` with `s²` the validation residual variance.
- The v4/v5 rows need the Kaggle fingerprint outputs: set `EF_FP_DIR` to the folder with `fp_embeddings_*.parquet`, `fp_comove_*.parquet` and the `fpnet*` prediction files before `run` (default `data/fingerprint/`); when the files are missing there, `run` silently writes the v1-v3 rows only.
- The v7/v8 add-on experiments (Welch beta, Kalman beta, MiniRocket) compare each arm with `linear_fe` only and were run after many earlier arms on the same validation and test periods, so a small test gain can partly be selection luck. Against the best earlier model chosen on validation, MiniRocket + engineered features ties for `vol_1m` and scores higher on test for `beta_1y`, where `fpnet_a` keeps the higher validation score; the Kalman beta does not help.
- Best-by-validation choices carry a winner's-curse margin: the v9 `vol_1m` stack beats `ridge_mrfe` by +0.026 R² on validation (0.672 vs 0.646) but only +0.005 [-0.000, +0.011] on test (`results/forward_v9_vol.csv`), so small gaps between the best models are within noise.
- v10 European stocks come from today's index constituent lists, so the references and the per-region benchmark rows are survivor-only, more strongly than the Swiss side. The cross-listing filter (same normalised name, or weekly CHF return correlation ≥ 0.90) also drops lines of distinct companies (Porsche SE vs Porsche AG, British Land vs Land Securities) next to true share classes (Maersk, Handelsbanken); this shrinks the reference set slightly but cannot leak a label.
- The test split is scored once per experiment file, but across the v1-v10 rounds the same test rows were read many times, and the add-on rounds were designed after earlier test results were known, so later test numbers are a second look rather than a clean hold-out.
- Forward add-on intervals (v7-v9) resample stocks only; same-date dependence between stocks is not resampled there (the v10 history rows also use a month-block bootstrap and report the wider interval). The static stock bootstrap resamples share lines of one issuer as independent stocks.
- `results/metrics.csv`, `robustness.csv` and `run_info.json` were produced at commit 0f5692a (`run_info.json`); later changes to the run path were behaviour-preserving, and the v10 `history_ext` control reproduced the `linear_fe` / `lgbm_fe` rows of `metrics.csv` (max abs diff below 1e-16).
- Code structure: the experiment modules were added round by round (v6-v10) and share a copied skeleton (folds, bootstrap, metric rows, reproduction checks); `static_cv.py` doubles as the shared library for the later static experiments. The duplicates were left as they are so that every committed result stays reproducible bit for bit.
- v10 kept only the European arm after the first run had shown that US references lowered the Swiss sector score; that is a post-hoc choice, so the `ch_eu` interval is optimistic.
- Static headline tables moved to issuer-grouped folds in v10; rows from earlier rounds that copy per-ticker numbers (`sector_eu`, `static_v9_sector`) are per-ticker and are not paired with the grouped headline.

## How to run

All commands run from the repo root; the package is used from `src/` through `PYTHONPATH` (no install step). `requirements.lock` lists the exact versions that produced the committed results (Python 3.12).

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.lock      # or requirements.txt (unpinned)
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m unittest discover -s tests   # synthetic tests only (226)
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.data     # download + cache data/ (git-ignored)
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.run      # v1-v5 benchmark -> metrics.csv, robustness.csv, run_info.json
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.static_cv  # v6: per-stock sector / log_mcap, repeated CV (per-ticker folds)
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.static_cv --grouped       # headline: issuer-grouped folds
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.ceilings   # v6: persistence and noise ceilings
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.forward_v7         # v7: Welch beta
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.forward_kalman     # v8: Kalman beta
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.forward_minirocket # v8: MiniRocket
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.extra_market       # v9: VIX / Euro Stoxx 50 closes
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.forward_v9_beta
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.forward_v9_vol
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.tabpfn_mcap        # v9: TabPFN v2 (checkpoint data/models/tabpfn-v2-regressor.ckpt or EF_TABPFN_CKPT)
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.static_v9_mcap     # v9 section (per-ticker folds)
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.static_v9_mcap --grouped  # headline ridge_md_ext row
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.static_v9_sector
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.pretrain_data      # non-Swiss price cache (v4 encoder, v9/v10 references)
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.sector_eu
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.multi_market --download-fx  # v10: CHF cross rates
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.multi_static    # v10: sector / log_mcap with European stocks
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.history_ext --arms  # v10: longer Swiss history (from 2011)
env -u PYTHONPATH PYTHONPATH=src .venv/bin/python -m equity_fingerprint.report   # rebuild the results block above
```

The v4/v5 fingerprint rows need the encoder outputs from the Kaggle GPU runs (`EF_FP_DIR`, see Limitations). The `*_oof.parquet` and `*_selection.csv` files next to each table hold the out-of-fold predictions and the hyper-parameter choices behind it. `report` leaves out a section whose result file is missing.

Set `EF_DATA_DIR` to use a data folder other than `./data`.
