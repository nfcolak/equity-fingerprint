"""Panel construction and feature engineering.

One observation per (ticker, month-end trading date) from 2014-03 on, needing at least 252 prior
trading rows. Every feature at date t uses data up to and including t only.

Cleaning (documented in the README):
  * returns are log changes of ``adj_close``; a daily |log return| > ``ARTIFACT_LOGRET`` is treated as
    an unadjusted corporate-action artifact (reverse split, rights issue, bad tick) and set to 0.
  * tickers whose ``Ticker.info`` currency is not CHF are excluded (traded value / market cap are in CHF).
  * missing trading days inside a ticker's listing span are forward filled (flat price, zero volume).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import load_index, load_info, load_prices

ANN = 252
OBS_START = pd.Timestamp("2014-03-01")
MIN_PRIOR_ROWS = 252
ARTIFACT_LOGRET = 0.7
HISTORY_CAP = 2500
PC_K = 6
PC_WINDOW = 252
TRAIN_END = pd.Timestamp("2021-12-31")
MARKET = "^SSHI"

RAW_FEATURES = ["ret_21", "vol_21", "log_volume_21", "log_close"]

FE_BASE_FEATURES = (
    ["ret_5", "ret_21", "ret_63", "ret_126", "ret_252"]
    + ["vol_21", "vol_63", "vol_252"]
    + ["parkinson_21", "parkinson_63", "gk_21", "gk_63"]
    + ["downside_vol_63", "skew_63", "kurt_63", "max_dd_252"]
    + ["beta_63", "beta_252", "corr_63", "corr_252", "idio_vol_63", "idio_vol_252"]
    + ["amihud_21", "amihud_63"]
    + ["log_tv_21", "log_tv_63", "log_tv_252"]
    + ["log_volume_21", "vol_trend", "zero_vol_share_63", "zero_ret_share_63", "history_days"]
    + ["log_close"]
)
PC_FEATURES = [f"pc{j + 1}_load" for j in range(PC_K)]
FE_FEATURES = FE_BASE_FEATURES + PC_FEATURES

# skewed, non-negative features: log-transformed for the linear models
LOG_FEATURES = [
    "vol_21", "vol_63", "vol_252", "parkinson_21", "parkinson_63", "gk_21", "gk_63",
    "downside_vol_63", "idio_vol_63", "idio_vol_252", "amihud_21", "amihud_63",
]

LABELS = {
    "ret_5": "Return 5d", "ret_21": "Return 21d", "ret_63": "Return 63d", "ret_126": "Return 126d",
    "ret_252": "Return 252d", "vol_21": "Volatility 21d", "vol_63": "Volatility 63d",
    "vol_252": "Volatility 252d", "parkinson_21": "Parkinson vol 21d", "parkinson_63": "Parkinson vol 63d",
    "gk_21": "Garman-Klass vol 21d", "gk_63": "Garman-Klass vol 63d", "downside_vol_63": "Downside vol 63d",
    "skew_63": "Skewness 63d", "kurt_63": "Kurtosis 63d", "max_dd_252": "Max drawdown 252d",
    "beta_63": "Beta vs SPI 63d", "beta_252": "Beta vs SPI 252d", "corr_63": "Correlation with SPI 63d",
    "corr_252": "Correlation with SPI 252d", "idio_vol_63": "Idiosyncratic vol 63d",
    "idio_vol_252": "Idiosyncratic vol 252d", "amihud_21": "Amihud illiquidity 21d",
    "amihud_63": "Amihud illiquidity 63d", "log_tv_21": "Log traded value 21d",
    "log_tv_63": "Log traded value 63d", "log_tv_252": "Log traded value 252d",
    "log_volume_21": "Log mean volume 21d", "vol_trend": "Volume trend 21d/252d",
    "zero_vol_share_63": "Zero-volume days 63d", "zero_ret_share_63": "Zero-return days 63d",
    "history_days": "Days of history", "log_close": "Log close",
    **{f"pc{j + 1}_load": f"Loading on return PC{j + 1}" for j in range(PC_K)},
}


# --------------------------------------------------------------------------- panel
class Panel:
    """Wide (date x ticker) frames on a common trading calendar."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def build_panel(data_dir=None) -> Panel:
    return panel_from_frames(load_prices(data_dir), load_index(data_dir), load_info(data_dir))


def panel_from_frames(prices: pd.DataFrame, index: pd.DataFrame, info: pd.DataFrame, chf_only: bool = True,
                      market: str = MARKET, rate: pd.DataFrame | None = None) -> Panel:
    """Panel from long price / index / info frames (the steps of ``build_panel``).

    chf_only: keep only tickers whose info currency is CHF (the Swiss benchmark); otherwise keep all.
    market: index ticker whose log returns become ``mkt``.
    rate: optional (date x ticker) local->CHF factors; close/open/high/low (and so traded value) are
    multiplied by it on the panel calendar; returns stay in local currency (from ``adj_close``).
    """
    currency = info.set_index("ticker")["currency"]
    have = sorted(prices["ticker"].unique())
    keep = [t for t in have if currency.get(t) == "CHF"] if chf_only else have
    excluded_currency = {t: (None if pd.isna(currency.get(t)) else str(currency.get(t)))
                         for t in have if t not in keep}
    prices = prices[prices["ticker"].isin(keep)]

    def wide(col):
        return prices.pivot(index="date", columns="ticker", values=col).sort_index()

    adj, close = wide("adj_close"), wide("close")
    opn, high, low, vol = wide("open"), wide("high"), wide("low"), wide("volume")
    cal = adj.index
    valid = adj.notna()
    span = (valid.cumsum() > 0) & (valid[::-1].cumsum()[::-1] > 0)
    n_filled = int((span & ~valid).sum().sum())

    adj = adj.ffill().where(span)
    close = close.ffill().where(span)
    miss = span & ~valid
    opn = opn.where(~miss, close).where(span)
    high = high.where(~miss, close).where(span)
    low = low.where(~miss, close).where(span)
    vol = vol.fillna(0.0).where(span)

    if rate is not None:  # local -> CHF on the panel calendar (prices only; returns stay local)
        R = rate.reindex(index=cal, columns=adj.columns)
        close, opn, high, low = close * R, opn * R, high * R, low * R

    # OHLC sanity for range-based vols: invalid bars -> NaN, clip H/L to contain O and C
    bad = (opn <= 0) | (high <= 0) | (low <= 0) | (high < low)
    hi = np.maximum(high, np.maximum(opn, close)).where(~bad)
    lo = np.minimum(low, np.minimum(opn, close)).where(~bad)
    opn_ok = opn.where(~bad)

    r = np.log(adj).diff()
    artifact = r.abs() > ARTIFACT_LOGRET
    n_artifacts = int(artifact.sum().sum())
    r = r.mask(artifact, 0.0)

    im = index[index["ticker"] == market].set_index("date")["close"].sort_index()
    im = im[~im.index.duplicated()].reindex(cal.union(im.index)).ffill().reindex(cal)
    m = np.log(im).diff()

    rows = span.cumsum().where(span)  # trading rows so far (incl. today)
    tv = (close * vol)

    return Panel(calendar=cal, tickers=list(adj.columns), adj=adj, close=close, open=opn_ok, high=hi,
                 low=lo, volume=vol, tv=tv, ret=r, mkt=m, rows=rows, span=span,
                 excluded_currency=excluded_currency, n_filled_days=n_filled,
                 n_artifact_returns=n_artifacts, info=info.set_index("ticker"))


def month_end_positions(panel: Panel) -> np.ndarray:
    cal = panel.calendar
    s = pd.Series(np.arange(len(cal)), index=cal)
    ym = cal.to_period("M")
    last = s.groupby(ym).last()
    # drop the (possibly incomplete) final month
    last = last.iloc[:-1]
    pos = last.to_numpy()
    return pos[cal[pos] >= OBS_START]


# --------------------------------------------------------------------------- features
def _rmean(x: pd.DataFrame, w: int, min_periods: int | None = None) -> pd.DataFrame:
    return x.rolling(w, min_periods=w if min_periods is None else min_periods).mean()


def _beta_block(r: pd.DataFrame, m: pd.Series, w: int):
    cov = r.rolling(w, min_periods=w).cov(m)
    var_m = m.rolling(w, min_periods=w).var()
    var_i = r.rolling(w, min_periods=w).var()
    beta = cov.div(var_m, axis=0)
    denom = np.sqrt(var_i.mul(var_m, axis=0))
    corr = (cov / denom).where(var_i > 1e-14, 0.0)
    beta = beta.where(var_i > 1e-14, 0.0)
    idio = np.sqrt(((var_i - beta ** 2 * var_m.to_numpy()[:, None]).clip(lower=0)) * ANN)
    return beta, corr, idio


def _range_vols(p: Panel, w: int):
    lh = np.log(p.high / p.low)
    co = np.log(p.close / p.open)
    pk = (lh ** 2) / (4 * np.log(2))
    gk = 0.5 * lh ** 2 - (2 * np.log(2) - 1) * co ** 2
    mp = max(w // 2, 5)
    park = np.sqrt(_rmean(pk, w, mp).clip(lower=0) * ANN)
    gkv = np.sqrt(_rmean(gk, w, mp).clip(lower=0) * ANN)
    return park, gkv


def _max_drawdown(p: Panel, pos: np.ndarray, w: int = 252) -> np.ndarray:
    cum = p.ret.fillna(0.0).cumsum().to_numpy()
    out = np.full((len(pos), cum.shape[1]), np.nan)
    for k, i in enumerate(pos):
        seg = np.exp(cum[i - w:i + 1])
        out[k] = (seg / np.maximum.accumulate(seg, axis=0)).min(axis=0) - 1.0
    return out


def base_features(p: Panel, pos: np.ndarray) -> pd.DataFrame:
    """All non-PC features for the observation positions; long frame, one row per (date, ticker)."""
    r, m = p.ret, p.mkt
    cum = r.fillna(0.0).cumsum()
    W: dict[str, pd.DataFrame | np.ndarray] = {}
    for h in (5, 21, 63, 126, 252):
        W[f"ret_{h}"] = cum - cum.shift(h)
    for w in (21, 63, 252):
        W[f"vol_{w}"] = r.rolling(w, min_periods=w).std() * np.sqrt(ANN)
    for w in (21, 63):
        W[f"parkinson_{w}"], W[f"gk_{w}"] = _range_vols(p, w)
    W["downside_vol_63"] = np.sqrt(_rmean(r.clip(upper=0) ** 2, 63) * ANN)
    W["skew_63"] = r.rolling(63, min_periods=63).skew()
    W["kurt_63"] = r.rolling(63, min_periods=63).kurt()
    for w in (63, 252):
        b, c, iv = _beta_block(r, m, w)
        W[f"beta_{w}"], W[f"corr_{w}"], W[f"idio_vol_{w}"] = b, c, iv
    amihud_term = (r.abs() / p.tv.where(p.tv > 0)) * 1e6
    for w in (21, 63):
        W[f"amihud_{w}"] = amihud_term.rolling(w, min_periods=1).mean()
    for w in (21, 63, 252):
        W[f"log_tv_{w}"] = np.log1p(_rmean(p.tv, w))
    mv21 = _rmean(p.volume, 21)
    W["log_volume_21"] = np.log1p(mv21)
    W["vol_trend"] = np.log1p(mv21) - np.log1p(_rmean(p.volume, 252))
    W["zero_vol_share_63"] = _rmean((p.volume == 0).astype(float).where(p.span), 63)
    W["zero_ret_share_63"] = _rmean((r == 0).astype(float).where(r.notna()), 63)
    W["history_days"] = p.rows.clip(upper=HISTORY_CAP)
    W["log_close"] = np.log(p.close)

    cal = p.calendar
    nt = len(p.tickers)
    data = {}
    for name, w in W.items():
        data[name] = w.iloc[pos].to_numpy().ravel()
    data["max_dd_252"] = _max_drawdown(p, pos).ravel()
    df = pd.DataFrame(data)
    df.insert(0, "ticker", np.tile(np.array(p.tickers, dtype=object), len(pos)))
    df.insert(0, "date", np.repeat(cal[pos].to_numpy(), nt))
    df.insert(0, "pos", np.repeat(pos, nt))
    keep = (p.rows.iloc[pos].to_numpy().ravel() >= MIN_PRIOR_ROWS + 1)
    df = df[keep & df[RAW_FEATURES].notna().all(axis=1).to_numpy()]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- return-structure (PCA) features
def pc_factors(p: Panel, fit_tickers: list[str], train_end=TRAIN_END, k: int = PC_K) -> pd.DataFrame:
    """Daily factor series: first k PCs of standardized daily returns.

    Weights are fitted by SVD on ``fit_tickers`` and dates <= ``train_end`` only; the resulting
    factor series is then evaluated on all dates from the same tickers' contemporaneous returns.
    """
    R = p.ret[[t for t in fit_tickers if t in p.ret.columns]]
    train = R.index <= train_end
    Rt = R[train]
    keep = Rt.columns[Rt.notna().sum() >= MIN_PRIOR_ROWS]
    sd = Rt[keep].std().replace(0, np.nan)
    keep = sd.dropna().index
    Z = (R[keep] / sd[keep]).clip(-6, 6).fillna(0.0)
    _, _, vt = np.linalg.svd(Z[train].to_numpy(), full_matrices=False)
    F = pd.DataFrame(Z.to_numpy() @ vt[:k].T, index=R.index, columns=PC_FEATURES)
    return F / F[train].std()


def pc_loadings(p: Panel, F: pd.DataFrame, pos: np.ndarray) -> pd.DataFrame:
    """Rolling PC_WINDOW-day correlation of each ticker's returns with each PC factor (observation rows)."""
    cal = p.calendar
    nt = len(p.tickers)
    data = {}
    for name in PC_FEATURES:
        c = p.ret.rolling(PC_WINDOW, min_periods=PC_WINDOW).corr(F[name])
        data[name] = c.iloc[pos].to_numpy().ravel()
    df = pd.DataFrame(data)
    df.insert(0, "ticker", np.tile(np.array(p.tickers, dtype=object), len(pos)))
    df.insert(0, "pos", np.repeat(pos, nt))
    return df


# --------------------------------------------------------------------------- v2: sector co-movement and size proxies
SIZE_FEATURES = ["tv21_rel", "tv252_rel", "tv21_rank", "tv252_rank", "close_rank"]
SECTOR_WINDOW = 252

LABELS.update({
    "tv21_rel": "Log traded value 21d minus date median", "tv252_rel": "Log traded value 252d minus date median",
    "tv21_rank": "Rank of log traded value 21d (date)", "tv252_rank": "Rank of log traded value 252d (date)",
    "close_rank": "Rank of log close (date)",
})


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s.lower()).strip("_")


def size_features(base: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional size proxies at each observation date (uses only that date's data, no labels)."""
    g = base.groupby("date")
    out = pd.DataFrame(index=base.index)
    out["tv21_rel"] = base["log_tv_21"] - g["log_tv_21"].transform("median")
    out["tv252_rel"] = base["log_tv_252"] - g["log_tv_252"].transform("median")
    out["tv21_rank"] = g["log_tv_21"].rank(pct=True)
    out["tv252_rank"] = g["log_tv_252"].rank(pct=True)
    out["close_rank"] = g["log_close"].rank(pct=True)
    return out


def _window_stats(X: np.ndarray, Y: np.ndarray, pos: np.ndarray, w: int):
    """Column-wise correlation and beta of X on Y over the w rows ending at each position (inclusive)."""
    corr = np.full((len(pos), X.shape[1]), np.nan)
    beta = np.full_like(corr, np.nan)
    for k, i in enumerate(pos):
        x, y = X[i - w + 1:i + 1], Y[i - w + 1:i + 1]
        xc, yc = x - x.mean(0), y - y.mean(0)
        cov = (xc * yc).sum(0) / (w - 1)
        vx, vy = (xc ** 2).sum(0) / (w - 1), (yc ** 2).sum(0) / (w - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            c = cov / np.sqrt(vx * vy)
            b = cov / vy
        dead = vx <= 1e-14
        corr[k], beta[k] = np.where(dead, 0.0, c), np.where(dead, 0.0, b)
    return corr, beta


def sector_features(p: Panel, pos: np.ndarray, train_tickers: list[str], sec_map: dict[str, str]):
    """Rolling 252d correlation / beta of each ticker with an equal-weight return index per sector.

    Each index uses only ``train_tickers`` (with their sector labels from ``sec_map``). For a training
    ticker its own sector index excludes itself (leave-one-out); val/test tickers never enter any index.
    Returns (frame keyed by pos, ticker, list of feature names).
    """
    R = p.ret
    Rv = R.to_numpy()
    Rf = np.nan_to_num(Rv, nan=0.0)
    N = (~np.isnan(Rv)).astype(float)
    ti = {t: j for j, t in enumerate(p.tickers)}
    nt = len(p.tickers)
    classes = sorted({sec_map[t] for t in train_tickers if t in sec_map and t in ti})
    data, names = {}, []
    for k in classes:
        mem = [ti[t] for t in train_tickers if sec_map.get(t) == k and t in ti]
        s_sum, s_cnt = Rf[:, mem].sum(1), N[:, mem].sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            S = np.where(s_cnt > 0, s_sum / s_cnt, np.nan)
            cnt_loo = s_cnt[:, None] - N[:, mem]
            loo = np.where(cnt_loo > 0, (s_sum[:, None] - Rf[:, mem]) / cnt_loo, np.nan)
        Y = np.repeat(S[:, None], nt, axis=1)
        Y[:, mem] = loo
        corr, beta = _window_stats(Rv, Y, pos, SECTOR_WINDOW)
        data[f"sec_corr_{_slug(k)}"], data[f"sec_beta_{_slug(k)}"] = corr.ravel(), beta.ravel()
        names += [f"sec_corr_{_slug(k)}", f"sec_beta_{_slug(k)}"]
        LABELS[f"sec_corr_{_slug(k)}"] = f"Correlation with {k} index (252d)"
        LABELS[f"sec_beta_{_slug(k)}"] = f"Beta vs {k} index (252d)"
    df = pd.DataFrame(data)
    df.insert(0, "ticker", np.tile(np.array(p.tickers, dtype=object), len(pos)))
    df.insert(0, "pos", np.repeat(pos, nt))
    return df, names
