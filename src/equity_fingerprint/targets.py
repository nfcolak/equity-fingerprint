"""Targets and splits.

Targets (see AGENTS.md):
  vol_1m        log annualized realized vol of daily log returns over the next 21 trading days
  beta_1y       beta vs ^SSHI over the next 252 trading days
  log_mcap      log(close x sharesOutstanding snapshot) at the feature date
  liquidity_1m  log(1 + mean daily traded value, CHF) over the next 21 trading days
  sector        yfinance sector (static label); classes with < MIN_CLASS_TICKERS tickers are dropped
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import ANN, Panel

TIME_TARGETS = ["vol_1m", "beta_1y", "liquidity_1m"]
GROUP_TARGETS = ["log_mcap", "sector"]
TARGETS = ["vol_1m", "beta_1y", "log_mcap", "liquidity_1m", "sector"]
REGRESSION = ["vol_1m", "beta_1y", "log_mcap", "liquidity_1m"]
HORIZON = {"vol_1m": 21, "beta_1y": 252, "liquidity_1m": 21}

VOL_FLOOR = 1e-3  # annualized; windows below it (zero realized vol) are flagged and dropped from vol_1m
MIN_CLASS_TICKERS = 5
SEED = 42

TRAIN_END = pd.Timestamp("2021-12-31")
VAL_START, VAL_END = pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")


def forward_targets(p: Panel, obs: pd.DataFrame) -> pd.DataFrame:
    """Time-target values and the date the forward window ends, aligned to obs (pos, ticker)."""
    r, m = p.ret, p.mkt
    cal = p.calendar
    fwd = {}
    v = r.rolling(21, min_periods=21).std().shift(-21) * np.sqrt(ANN)
    fwd["vol_1m"] = np.log(v.clip(lower=VOL_FLOOR))
    cov = r.rolling(252, min_periods=252).cov(m)
    var_m = m.rolling(252, min_periods=252).var()
    fwd["beta_1y"] = cov.div(var_m, axis=0).shift(-252)
    fwd["liquidity_1m"] = np.log1p(p.tv.rolling(21, min_periods=21).mean().shift(-21))
    out = pd.DataFrame(index=obs.index)
    ti = {t: j for j, t in enumerate(p.tickers)}
    cols = obs["ticker"].map(ti).to_numpy()
    rows = obs["pos"].to_numpy()
    for name, w in fwd.items():
        out[name] = w.to_numpy()[rows, cols]
        end = rows + HORIZON[name]
        out[f"end_{name}"] = pd.to_datetime(
            np.where(end < len(cal), cal.to_numpy()[np.minimum(end, len(cal) - 1)], np.datetime64("NaT")))
    out["vol_1m_floored"] = v.to_numpy()[rows, cols] < VOL_FLOOR
    return out


def time_split(date: pd.Series, end: pd.Series) -> pd.Series:
    """train/val/test by feature date; train and val drop observations whose forward window crosses the split end."""
    s = pd.Series(np.nan, index=date.index, dtype=object)
    s[(date <= TRAIN_END) & (end <= TRAIN_END)] = "train"
    s[(date >= VAL_START) & (date <= VAL_END) & (end <= VAL_END)] = "val"
    s[(date >= TEST_START) & end.notna()] = "test"
    return s


def ticker_split(tickers: list[str], strata: dict[str, str], seed: int = SEED) -> dict[str, str]:
    """70/15/15 by ticker, stratified by sector label where possible (seeded)."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[str]] = {}
    for t in sorted(tickers):
        groups.setdefault(strata.get(t) or "unknown", []).append(t)
    out: dict[str, str] = {}
    for key in sorted(groups):
        ts = list(groups[key])
        rng.shuffle(ts)
        n = len(ts)
        n_val = n_test = int(round(0.15 * n))
        if n >= 5:  # make sure every stratum with >= 5 tickers reaches val and test
            n_val, n_test = max(n_val, 1), max(n_test, 1)
        for k, t in enumerate(ts):
            out[t] = "test" if k < n_test else "val" if k < n_test + n_val else "train"
    return out


def sector_map(p: Panel, tickers) -> tuple[dict[str, str], list[dict]]:
    """ticker -> sector, dropping classes with fewer than MIN_CLASS_TICKERS tickers (dropped, not merged)."""
    sec = p.info["sector"].reindex(list(tickers)).dropna()
    counts = sec.value_counts()
    dropped = [{"sector": k, "n_tickers": int(v)} for k, v in counts.items() if v < MIN_CLASS_TICKERS]
    small = {d["sector"] for d in dropped}
    return {t: s for t, s in sec.items() if s not in small}, dropped


def log_mcap_target(p: Panel, obs: pd.DataFrame) -> pd.Series:
    shares = p.info["shares_outstanding"].reindex(obs["ticker"]).to_numpy()
    close = np.exp(obs["log_close"].to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log(close * np.where(shares > 0, shares, np.nan))
    return pd.Series(y, index=obs.index)
