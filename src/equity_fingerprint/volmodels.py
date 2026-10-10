"""v3 volatility reference features: HAR inputs, GJR-GARCH(1,1) forecast, market volatility state.

Every value at feature date t uses returns up to and including t only. The only quantities that look past
the date are the GARCH *parameters*, which are fitted once on returns up to ``TRAIN_END`` (2021-12-31);
the conditional variance itself is filtered forward causally (each date sees only past returns).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

from .features import ANN, LABELS, TRAIN_END, Panel

HAR_COLS = ["har_rv1", "har_rv5", "har_rv22"]
GARCH_COLS = ["garch_vol21", "garch_persist"]
MKT_COLS = ["mkt_vol21", "mkt_vol252", "mkt_vol_ratio"]

HAR_WINDOWS = (1, 5, 22)
VOL_FLOOR = 0.005       # annualised vol floor before the log (a zero-return window has no defined log)
MIN_TRAIN_RET = 500     # fewer train-period returns than this -> pooled GARCH parameters
HORIZON = 21
SCALE = 100.0           # returns in percent inside the optimiser (better conditioned)
P_MAX = 0.9999          # stationarity bound on alpha + gamma/2 + beta

LABELS.update({
    "har_rv1": "HAR: log realised vol 1d", "har_rv5": "HAR: log realised vol 5d",
    "har_rv22": "HAR: log realised vol 22d",
    "garch_vol21": "GJR-GARCH: log mean vol forecast, next 21d", "garch_persist": "GJR-GARCH: persistence",
    "mkt_vol21": "SPI log realised vol 21d", "mkt_vol252": "SPI log realised vol 252d",
    "mkt_vol_ratio": "SPI log vol 21d minus 252d",
})

# filled by vol_features: how many tickers got their own GARCH fit vs pooled parameters
LAST_FIT: dict = {}


def _log_rv(r2: pd.DataFrame | pd.Series, w: int):
    """Log annualised realised vol from squared daily log returns over the last w days."""
    rv = np.sqrt(r2.rolling(w, min_periods=w).mean() * ANN)
    return np.log(rv.clip(lower=VOL_FLOOR))


# --------------------------------------------------------------------------- GJR-GARCH(1,1)
def _filter(params, r: np.ndarray) -> np.ndarray:
    """One-step-ahead variance s2[k] = Var(r[k] | r[:k]) for k = 0..n (length n + 1), GJR recursion.

    s2[k+1] = omega + (alpha + gamma * 1[r_k < 0]) r_k^2 + beta s2[k], started at the unconditional variance.
    """
    omega, alpha, gamma, beta = params
    p = alpha + gamma / 2 + beta
    s2_0 = omega / (1 - p)
    u = omega + (alpha + gamma * (r < 0)) * r ** 2
    # y[k] = s2[k+1]: y[k] = beta*y[k-1] + u[k], with y[-1] = s2[0] -> zi = beta*s2_0
    y, _ = lfilter([1.0], [1.0, -beta], u, zi=[beta * s2_0])
    return np.concatenate([[s2_0], y])


def _nll(params, series: list[np.ndarray]) -> float:
    tot = 0.0
    for r in series:
        s2 = _filter(params, r)[:-1]
        if not np.all(np.isfinite(s2)) or s2.min() <= 0:
            return 1e12
        tot += 0.5 * np.sum(np.log(s2) + r ** 2 / s2)
    return tot / sum(len(r) for r in series)


def _fit(series: list[np.ndarray]):
    """Gaussian MLE of (omega, alpha, gamma, beta); returns (params, success)."""
    v = float(np.mean(np.concatenate(series) ** 2))
    cons = [{"type": "ineq", "fun": lambda x: P_MAX - (x[1] + x[2] / 2 + x[3])}]
    bnds = [(1e-8 * v, 10 * v), (1e-6, 1.0), (0.0, 1.0), (1e-6, 1.0)]
    best = None
    for a0, g0, b0 in ((0.03, 0.05, 0.90), (0.08, 0.05, 0.80)):
        x0 = [v * (1 - (a0 + g0 / 2 + b0)), a0, g0, b0]
        res = minimize(_nll, x0, args=(series,), method="SLSQP", bounds=bnds, constraints=cons,
                       options={"maxiter": 300, "ftol": 1e-10})
        ok = bool(res.success) and np.all(np.isfinite(res.x)) and res.x[1] + res.x[2] / 2 + res.x[3] < 1
        if ok and (best is None or res.fun < best[0]):
            best = (res.fun, res.x)
    return (best[1], True) if best is not None else (None, False)


def garch_features(p: Panel) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide (date x ticker) frames: log annualised mean vol forecast over the next 21 days, and persistence."""
    R = p.ret
    cal = R.index
    train = (cal <= TRAIN_END)
    tr_series = {}
    for t in R.columns:
        x = R[t][train].dropna().to_numpy() * SCALE
        tr_series[t] = x

    pooled, ok = _fit([x for x in tr_series.values() if len(x) > 0])
    if not ok:
        raise RuntimeError("pooled GJR-GARCH fit failed")
    own, n_own, n_fail = {}, 0, 0
    for t, x in tr_series.items():
        if len(x) >= MIN_TRAIN_RET:
            par, good = _fit([x])
            if good:
                own[t] = par
                n_own += 1
                continue
            n_fail += 1
    n_pool = len(R.columns) - n_own
    LAST_FIT.update(n_tickers=len(R.columns), n_per_ticker=n_own, n_pooled=n_pool,
                    n_pooled_short_history=n_pool - n_fail, n_pooled_fit_failed=n_fail,
                    pooled_params=[float(v) for v in pooled])
    print(f"[garch] GJR-GARCH fits: {n_own} per-ticker, {n_pool} pooled "
          f"({n_pool - n_fail} with < {MIN_TRAIN_RET} train returns, {n_fail} optimiser failures)")

    ks = np.arange(HORIZON)
    vol = pd.DataFrame(np.nan, index=cal, columns=R.columns)
    pers = pd.DataFrame(np.nan, index=cal, columns=R.columns)
    for t in R.columns:
        par = own.get(t, pooled)
        omega, alpha, gamma, beta = par
        pp = alpha + gamma / 2 + beta
        s = R[t]
        valid = s.notna().to_numpy()
        if not valid.any():
            continue
        i0 = int(np.argmax(valid))
        i1 = len(valid) - int(np.argmax(valid[::-1]))
        r = s.iloc[i0:i1].fillna(0.0).to_numpy() * SCALE
        s2_next = _filter(par, r)[1:]          # variance of day k+1 given returns up to day k
        v_inf = omega / (1 - pp)
        avg = np.mean(pp ** ks)                # mean over k = 0..20 of persistence^k
        vbar = v_inf + (s2_next - v_inf) * avg  # mean variance forecast over days t+1..t+21
        vol.iloc[i0:i1, vol.columns.get_loc(t)] = np.log(np.sqrt(vbar * ANN) / SCALE)
        pers.iloc[i0:i1, pers.columns.get_loc(t)] = pp
    return vol, pers


# --------------------------------------------------------------------------- public entry point
def vol_features(panel: Panel, obs: pd.DataFrame) -> pd.DataFrame:
    """HAR, GARCH and market-vol features aligned to the rows of ``obs`` (keys ``pos``, ``ticker``)."""
    r2 = panel.ret ** 2
    wide = {f"har_rv{w}": _log_rv(r2, w) for w in HAR_WINDOWS}
    wide["garch_vol21"], wide["garch_persist"] = garch_features(panel)

    pos = obs["pos"].to_numpy()
    cols = panel.ret.columns.get_indexer(obs["ticker"])
    out = pd.DataFrame({"pos": pos, "ticker": obs["ticker"].to_numpy()}, index=obs.index)
    for name, w in wide.items():
        out[name] = w.to_numpy()[pos, cols]

    m2 = panel.mkt ** 2
    m21, m252 = _log_rv(m2, 21), _log_rv(m2, 252)
    out["mkt_vol21"] = m21.to_numpy()[pos]
    out["mkt_vol252"] = m252.to_numpy()[pos]
    out["mkt_vol_ratio"] = out["mkt_vol21"] - out["mkt_vol252"]
    return out[["pos", "ticker"] + HAR_COLS + GARCH_COLS + MKT_COLS]
