"""v9 log_mcap, part 1: lightgbm-free feature/fold helpers and the TabPFN out-of-fold predictions.

Module split (macOS libomp rule: lightgbm and torch/tabpfn must not live in one process). ``models`` (hence
``static_cv``) imports lightgbm at module level, so this module imports neither. It holds everything both
v9 log_mcap modules share and that needs no lightgbm: the extended causal features, the (linear) Prep
copy, the log_mcap inputs and the outer folds. ``static_v9_mcap`` (which uses lightgbm) imports these
helpers from here. torch and tabpfn are imported lazily inside ``make_tabpfn_factory`` (torch first), so
importing this module never loads them.

New causal features at a month-end position ``pos`` (one definition each, data up to and including pos):
  log_rebal_spike   third Friday of Mar/Jun/Sep/Dec up to the feature date (if not a trading day: the last
                    trading day before it; needs 20 earlier rows). Event ratio = volume that day / median
                    volume of the 20 trading days before it (NaN if that median is 0 or volume missing).
                    Feature = log(max(median of the ratios of the last 20 events, 0.05)); fewer than 4
                    valid events -> NaN (Prep fills it with the training-fold median).
  VOLSTRUCT         tv_cv_63 (cv of traded value, 63d, >= 40 obs), log_tv_504 (log1p mean tv, 504d, >= 252),
                    log_tv_med_252 / log_tv_max_252 / log_tv_p10_252 (log1p of median / max / 10% quantile
                    of tv, 252d, >= 126), logtv_minus_logvol (log1p mean tv 252d, >= 126, minus log of the
                    252d annualised return sd clipped at 1e-3), log_amihud_252 (log(1e-12 + 1e6 x mean of
                    |ret| / tv, 252d, >= 126)).
  SMI               corr_smi_252, beta_smi_252 (252d rolling corr / beta of the stock on ^SSMI returns),
                    corr_smi_minus_spi_252 (that corr minus the corr with the SPI), beta_smiperp_252
                    (beta on the SMI component orthogonal to the SPI: smi - b x spi, b = rolling 252d beta
                    of smi on spi). ^SSMI comes from ``panel.smi`` (daily log returns on the panel
                    calendar) if the panel has it, else from ``data.load_index`` exactly like the SPI.
FE_EXT = FE_BASE_FEATURES + [log_rebal_spike] + VOLSTRUCT + SMI. NaNs are NOT filled here (the research
probe filled with the all-stock median before CV; that leaked): Prep fits its fill on the training fold.

TabPFN: per outer fold (KFold(5, shuffle, random_state=repeat) over the stocks of ``mcap_inputs``, the same
folds as static_cv) a TabPFNRegressor(model_path=ckpt, device="cpu", n_estimators=8, random_state=repeat,
ignore_pretraining_limits=True) is fitted on Prep(linear=True) features of the training stocks (labels of the
training stocks only) for the feature sets ``base`` (FE_BASE_FEATURES) and ``ext`` (FE_EXT). The regressor
factory is injectable. Output ``static_v9_tabpfn_oof.parquet``: feature_set, repeat, ticker, y_pred.
Checkpoint: env ``EF_TABPFN_CKPT`` or ``data/models/tabpfn-v2-regressor.ckpt``.
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .data import load_index
from .features import ANN, FE_BASE_FEATURES, LOG_FEATURES, base_features, build_panel, month_end_positions
from .targets import log_mcap_target

SPIKE_MONTHS = (3, 6, 9, 12)
SPIKE_EVENTS = 20
SPIKE_MIN_EVENTS = 4
SPIKE_WINDOW = 20
SPIKE_FLOOR = 0.05
SPIKE_FEATURE = "log_rebal_spike"
VOLSTRUCT = ["tv_cv_63", "log_tv_504", "log_tv_med_252", "log_tv_max_252", "log_tv_p10_252",
             "logtv_minus_logvol", "log_amihud_252"]
SMI = ["corr_smi_252", "beta_smi_252", "corr_smi_minus_spi_252", "beta_smiperp_252"]
EXT_FEATURES = [SPIKE_FEATURE] + VOLSTRUCT + SMI
FE_EXT = FE_BASE_FEATURES + EXT_FEATURES
FEATURE_SETS = {"base": FE_BASE_FEATURES, "ext": FE_EXT}
TABPFN_FILE = "static_v9_tabpfn_oof.parquet"
DEFAULT_CKPT = Path("data/models/tabpfn-v2-regressor.ckpt")


# --------------------------------------------------------------------------- features
def third_friday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year, month, 1)
    return first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)


def spike_events(calendar: pd.DatetimeIndex, pos: int) -> list[int]:
    """Calendar positions of the rebalance-day events up to the feature date (the last SPIKE_EVENTS of
    them). A third Friday that is not a trading day maps to the last trading day before it."""
    out = []
    for yy in range(calendar[0].year, calendar[pos].year + 1):
        for mm in SPIKE_MONTHS:
            tf = third_friday(yy, mm)
            if tf > calendar[pos]:
                continue
            e = int(calendar.searchsorted(tf, side="right")) - 1
            if e >= SPIKE_WINDOW:
                out.append(e)
    return out[-SPIKE_EVENTS:]


def rebal_spike(volume: np.ndarray, calendar: pd.DatetimeIndex, pos: int) -> np.ndarray:
    """log_rebal_spike for every column of ``volume`` (rows = calendar) at calendar position pos."""
    ev = spike_events(calendar, pos)
    n = volume.shape[1]
    if not ev:
        return np.full(n, np.nan)
    ratios = np.full((len(ev), n), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, e in enumerate(ev):
            med = np.nanmedian(volume[e - SPIKE_WINDOW:e], axis=0)
            ratios[k] = np.where(med > 0, volume[e] / np.where(med > 0, med, 1.0), np.nan)
        cnt = np.isfinite(ratios).sum(0)
        med_ratio = np.nanmedian(ratios, axis=0)
    return np.where(cnt >= SPIKE_MIN_EVENTS, np.log(np.maximum(med_ratio, SPIKE_FLOOR)), np.nan)


def smi_returns(p) -> pd.Series:
    """Daily log returns of ^SSMI on the panel calendar."""
    if getattr(p, "smi", None) is not None:
        return pd.Series(p.smi).reindex(p.calendar)
    ix = load_index(None)
    s = ix[ix["ticker"] == "^SSMI"].set_index("date")["close"].sort_index()
    s = s[~s.index.duplicated()]
    s = s.reindex(p.calendar.union(s.index)).ffill().reindex(p.calendar)
    return np.log(s).diff()


def ext_wide(p) -> dict[str, pd.DataFrame]:
    """VOLSTRUCT and SMI as wide (calendar x ticker) frames; every value uses data up to its own row."""
    r, tv, mkt = p.ret, p.tv, p.mkt
    w = {}
    tv63 = tv.rolling(63, min_periods=40)
    w["tv_cv_63"] = tv63.std() / tv63.mean()
    w["log_tv_504"] = np.log1p(tv.rolling(504, min_periods=252).mean())
    t252 = tv.rolling(252, min_periods=126)
    w["log_tv_med_252"] = np.log1p(t252.median())
    w["log_tv_max_252"] = np.log1p(t252.max())
    w["log_tv_p10_252"] = np.log1p(t252.quantile(0.1))
    v252 = r.rolling(252, min_periods=252).std() * np.sqrt(ANN)
    w["logtv_minus_logvol"] = np.log1p(t252.mean()) - np.log(v252.clip(lower=1e-3))
    am = r.abs() / tv.where(tv > 0)
    w["log_amihud_252"] = np.log(1e-12 + am.rolling(252, min_periods=126).mean() * 1e6)
    ms = smi_returns(p)
    cm = r.rolling(252, min_periods=252).corr(ms)
    cs = r.rolling(252, min_periods=252).corr(mkt)
    w["corr_smi_252"] = cm
    w["beta_smi_252"] = r.rolling(252, min_periods=252).cov(ms).div(ms.rolling(252).var(), axis=0)
    w["corr_smi_minus_spi_252"] = cm - cs
    b = ms.rolling(252, min_periods=252).cov(mkt) / mkt.rolling(252, min_periods=252).var()
    perp = ms - b * mkt
    w["beta_smiperp_252"] = r.rolling(252, min_periods=252).cov(perp).div(perp.rolling(252).var(), axis=0)
    return w


def ext_features(p, base: pd.DataFrame) -> pd.DataFrame:
    """EXT_FEATURES for the rows of ``base`` (needs columns pos, ticker); same index as ``base``."""
    col = {t: j for j, t in enumerate(p.tickers)}
    ti = base["ticker"].map(col).to_numpy()
    ps = base["pos"].to_numpy()
    out = pd.DataFrame(index=base.index)
    vol = p.volume.to_numpy(float)
    spike = {int(q): rebal_spike(vol, p.calendar, int(q)) for q in np.unique(ps)}
    out[SPIKE_FEATURE] = [spike[int(q)][j] for q, j in zip(ps, ti)]
    for name, wide in ext_wide(p).items():
        out[name] = wide.to_numpy(float)[ps, ti]
    return out.replace([np.inf, -np.inf], np.nan)[EXT_FEATURES]


# --------------------------------------------------------------------------- inputs, folds, Prep
def mcap_inputs_v9(p):
    """Same eligibility and order as ``static_cv.mcap_inputs`` (latest month-end, FE_BASE complete, finite
    target), plus the extended frame. Returns tickers, y, X_base, X_ext, date."""
    pos = month_end_positions(p)[-1:]
    base = base_features(p, pos)
    y = log_mcap_target(p, base)
    X = base[FE_BASE_FEATURES].replace([np.inf, -np.inf], np.nan)
    ok = np.isfinite(y.to_numpy()) & X.notna().all(axis=1).to_numpy()
    ext = pd.concat([X, ext_features(p, base)], axis=1)[FE_EXT]
    return (base.loc[ok, "ticker"].tolist(), y[ok].to_numpy(), X[ok].reset_index(drop=True),
            ext[ok].reset_index(drop=True), base.loc[ok, "date"].iloc[0])


def outer_folds(n: int, repeat: int):
    return list(KFold(5, shuffle=True, random_state=repeat).split(np.zeros(n)))


class LinearPrep:
    """Copy of ``models.Prep(cols, linear=True)`` (models imports lightgbm, this module must not): winsorize
    1%/99%, log1p-transform LOG_FEATURES, impute the median, standardise; training-fold statistics only.
    Output is made finite (an all-NaN training column becomes 0)."""

    def __init__(self, cols):
        self.cols = list(cols)

    def _log(self, X):
        X = X.copy()
        for c in self.cols:
            if c in LOG_FEATURES:
                X[c] = np.log1p(X[c].clip(lower=0) / self.scale_[c])
        return X

    def fit(self, X: pd.DataFrame) -> "LinearPrep":
        X = X[self.cols].astype(float)
        self.scale_ = {}
        for c in self.cols:
            pos = X[c][X[c] > 0]
            self.scale_[c] = float(pos.median()) if len(pos) else 1.0
        X = self._log(X)
        self.lo_, self.hi_ = X.quantile(0.01), X.quantile(0.99)
        Xc = X.clip(self.lo_, self.hi_, axis=1)
        self.med_ = Xc.median()
        self.mu_ = Xc.fillna(self.med_).mean()
        self.sd_ = Xc.fillna(self.med_).std().replace(0, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = self._log(X[self.cols].astype(float))
        X = X.clip(self.lo_, self.hi_, axis=1)
        return np.nan_to_num(((X.fillna(self.med_) - self.mu_) / self.sd_).to_numpy())


# --------------------------------------------------------------------------- TabPFN
def make_tabpfn_factory(ckpt=None) -> Callable[[int], object]:
    """factory(repeat) -> TabPFNRegressor. torch is imported first, tabpfn lazily (never in tests)."""
    import torch  # noqa: F401  (load before tabpfn)
    from tabpfn import TabPFNRegressor
    path = str(ckpt or os.environ.get("EF_TABPFN_CKPT") or DEFAULT_CKPT)

    def factory(repeat: int):
        return TabPFNRegressor(model_path=path, device="cpu", n_estimators=8, random_state=repeat,
                               ignore_pretraining_limits=True)
    return factory


def tabpfn_oof(X_base: pd.DataFrame, X_ext: pd.DataFrame, y, tickers, repeats: int,
               factory: Callable[[int], object], log=None) -> pd.DataFrame:
    """Out-of-fold TabPFN predictions: one row per (feature_set, repeat, stock). Only y[train] is read."""
    y = np.asarray(y, float)
    frames = {"base": X_base, "ext": X_ext}
    parts = []
    for r in range(repeats):
        pred = {fs: np.full(len(y), np.nan) for fs in frames}
        for tr, te in outer_folds(len(y), r):
            for fs, X in frames.items():
                prep = LinearPrep(FEATURE_SETS[fs]).fit(X.iloc[tr])
                m = factory(r)
                m.fit(prep.transform(X.iloc[tr]), y[tr])
                pred[fs][te] = np.asarray(m.predict(prep.transform(X.iloc[te])), float)
        for fs in frames:
            parts.append(pd.DataFrame({"feature_set": fs, "repeat": r, "ticker": list(tickers), "y_pred": pred[fs]}))
        if log:
            log(r)
    return pd.concat(parts, ignore_index=True)


def run(repeats: int = 10, out_dir="results", panel=None, factory=None) -> pd.DataFrame:
    t0 = time.time()
    p = panel or build_panel()
    tickers, y, Xb, Xe, date = mcap_inputs_v9(p)
    print(f"log_mcap tabpfn: {len(tickers)} stocks at {pd.Timestamp(date).date()}", flush=True)
    factory = factory or make_tabpfn_factory()
    df = tabpfn_oof(Xb, Xe, y, tickers, repeats, factory,
                    log=lambda r: print(f"  repeat {r} done ({time.time() - t0:.0f}s)", flush=True))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / TABPFN_FILE, index=False)
    print(f"done in {time.time() - t0:.0f}s")
    return df


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args(argv)
    run(a.repeats, a.out_dir)


if __name__ == "__main__":
    main()
