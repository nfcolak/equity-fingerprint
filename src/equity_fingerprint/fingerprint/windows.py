"""Daily input channels and 252-day windows for the fingerprint encoder.

Library module (no CLI). Channels per ticker and day, all causal (a day uses data up to that day):
  ret        log adj_close return (|r| > ``ARTIFACT_LOGRET`` -> 0, like the v1-v3 features)
  hl_range   log(high / low)
  gap        log(open / previous close)
  logvol_rel log1p(volume) minus its trailing 252-day median (including that day)
  mkt_ret    log return of the ticker's market index
  zero_vol   1 if volume == 0
NaN marks days a ticker has no data. ``window_batch`` robust-scales a window, sets NaN to 0 and appends
an availability channel (1 where the day had data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import ARTIFACT_LOGRET

CHANNELS = ["ret", "hl_range", "gap", "logvol_rel", "mkt_ret", "zero_vol"]
WINDOW = 252
PATCH = 21
CLIP = 10.0  # scaled values are clipped to +-CLIP robust units (bad ticks must not dominate MSE)
MIN_VOL_ROWS = 20


def _logret(x: pd.DataFrame) -> pd.DataFrame:
    """Log change vs the previous valid value (a holiday gap gives a multi-day return)."""
    lg = np.log(x.where(x > 0))
    r = lg.ffill().diff().where(lg.notna())
    return r.mask(r.abs() > ARTIFACT_LOGRET, 0.0)


def channels_from_wide(adj, close, open_, high, low, volume, mkt_close) -> np.ndarray:
    """date x ticker frames on one calendar -> float32 [T, N, len(CHANNELS)]."""
    assert all(f.shape == adj.shape for f in (close, open_, high, low, volume, mkt_close))
    mkt_close = pd.DataFrame(mkt_close.to_numpy(), index=adj.index, columns=adj.columns)
    valid = adj.notna() & (adj > 0)

    ret = _logret(adj)
    ok = (high > 0) & (low > 0) & (high >= low)
    hl = np.log(high.where(ok) / low.where(ok))
    prev = close.where(close > 0).ffill().shift(1)
    gap = np.log(open_.where(open_ > 0) / prev)
    gap = gap.mask(gap.abs() > ARTIFACT_LOGRET, 0.0)
    lv = np.log1p(volume.clip(lower=0))
    logvol_rel = lv - lv.rolling(WINDOW, min_periods=MIN_VOL_ROWS).median()
    mkt = _logret(mkt_close)
    zero = (volume == 0).astype(float).where(volume.notna())

    frames = [ret, hl, gap, logvol_rel, mkt, zero]
    return np.stack([f.where(valid).to_numpy(np.float32) for f in frames], axis=-1)


def swiss_channels(panel):
    """Channels for the Swiss benchmark panel (market = ^SSHI, taken from ``panel.mkt``)."""
    mkt = np.exp(panel.mkt.cumsum()).to_numpy()
    mkt_close = pd.DataFrame(np.repeat(mkt[:, None], panel.adj.shape[1], axis=1),
                             index=panel.adj.index, columns=panel.adj.columns)
    ch = channels_from_wide(panel.adj, panel.close, panel.open, panel.high, panel.low, panel.volume,
                            mkt_close)
    return ch, list(panel.adj.columns)


def window_batch(ch, t_end, j, scaler) -> np.ndarray:
    """Windows ending at rows ``t_end`` (inclusive) of ticker columns ``j`` -> [B, WINDOW, C + 1]."""
    t_end, j = np.asarray(t_end), np.asarray(j)
    rows = t_end[:, None] + np.arange(-WINDOW + 1, 1)
    inside = rows >= 0
    x = ch[np.where(inside, rows, 0), j[:, None]]  # fancy indexing copies
    x[~inside] = np.nan
    avail = np.isfinite(x[..., :2]).any(-1)
    med = np.asarray(scaler["median"], np.float32)
    iqr = np.asarray(scaler["iqr"], np.float32)
    x = np.clip(np.nan_to_num((x - med) / iqr, nan=0.0, posinf=CLIP, neginf=-CLIP), -CLIP, CLIP)
    return np.concatenate([x, avail[..., None].astype(np.float32)], axis=-1).astype(np.float32)


def window_tensor(ch, t_end, j, scaler) -> np.ndarray:
    """One window: rows t_end-WINDOW+1 .. t_end of ticker column j -> [WINDOW, len(CHANNELS) + 1]."""
    return window_batch(ch, [t_end], [j], scaler)[0]
