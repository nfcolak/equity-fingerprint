"""Causal Kalman-filter beta (random-walk state space), vectorised across stocks.

Model per stock (daily returns ``r``, market return ``m`` = ``panel.mkt``)::

    r_t = alpha_t + beta_t * z_t + e_t,   z_t = m_t / scale,      e_t ~ N(0, s2)
    alpha_t = alpha_{t-1} + u_t,          var(u) = QA * s2        (alpha: tiny random walk, QA fixed)
    beta*_t = beta*_{t-1} + w_t,          var(w) = q  * s2        (beta* = beta * scale)

* ``scale`` is the standard deviation of the market return over dates <= ``train_end`` (so ``q`` is
  dimension-free: a drift of ~0.2 beta/year corresponds to q of order 1e-4).
* The state covariance is kept in units of ``s2``. ``s2`` is estimated causally per stock as the running mean of
  the normalised squared one-step innovations ``v^2 / S``, started from the OLS residual variance (``ss / n`` with n = 250 = 252 - 2 degrees of freedom).
* Initialisation: per stock, at the date of its 252nd valid (r and m observed) day, from the OLS fit over those 252
  days (alpha, beta, ``(X'X)^-1`` as covariance). Output is NaN before that date.
* Missing days (r or m NaN): prediction step only (state carried, covariance grows).
* Output at date t is the *filtered* estimate given data up to and including t: ``beta`` and its posterior sd
  (``sqrt(P11 * s2) / scale``). Nothing after t is used, so the value at t does not change if later data change.
* ``q`` is one value for all stocks, chosen from ``Q_GRID`` by the summed one-step-ahead predictive log-likelihood
  on dates <= ``train_end`` only (``select_q`` truncates the data to that date before filtering).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Q_GRID = (1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
QA = 1e-7  # alpha random-walk variance in units of s2 (alpha is nearly constant)
TRAIN_END = "2021-12-31"
INIT = 252


def _init_states(R: np.ndarray, M: np.ndarray, z: np.ndarray):
    """Per stock: init index t0 (252nd valid day) and OLS state. Returns arrays (t0, a, b, P00, P01, P11, ss)."""
    T, N = R.shape
    t0 = np.full(N, -1)
    a = np.full(N, np.nan); b = a.copy(); P00 = a.copy(); P01 = a.copy(); P11 = a.copy(); ss = a.copy()
    for j in range(N):
        ok = np.flatnonzero(np.isfinite(R[:, j]) & np.isfinite(z))
        if len(ok) < INIT:
            continue
        idx = ok[:INIT]
        X = np.column_stack([np.ones(INIT), z[idx]])
        y = R[idx, j]
        XtX = X.T @ X
        if np.linalg.cond(XtX) > 1e12:
            continue
        inv = np.linalg.inv(XtX)
        coef = inv @ (X.T @ y)
        res = y - X @ coef
        t0[j] = idx[-1]
        a[j], b[j] = coef
        P00[j], P01[j], P11[j] = inv[0, 0], inv[0, 1], inv[1, 1]
        ss[j] = float(res @ res)
    return t0, a, b, P00, P01, P11, ss


def kalman_filter(ret: pd.DataFrame, mkt: pd.Series, q: float, scale: float):
    """Run the filter for one ``q``. Returns (beta, beta_sd, loglik) with beta/beta_sd DataFrames like ``ret``
    and loglik a DataFrame of per-day per-stock predictive log-likelihoods (NaN where no update happened)."""
    R = ret.to_numpy(float)
    M = mkt.reindex(ret.index).to_numpy(float)
    z = M / scale
    T, N = R.shape
    t0, a, b, P00, P01, P11, ss = _init_states(R, M, z)
    n = np.full(N, INIT - 2.0)
    BETA = np.full((T, N), np.nan); SD = BETA.copy(); LL = BETA.copy()
    live = t0 >= 0
    for t in range(T):
        at0 = live & (t0 == t)
        if at0.any():  # state was initialised from OLS over the first 252 valid days, ending today
            s2 = ss[at0] / n[at0]
            BETA[t, at0] = b[at0] / scale
            SD[t, at0] = np.sqrt(P11[at0] * s2) / scale
        act = live & (t0 < t)
        if not act.any():
            continue
        P00[act] += QA
        P11[act] += q
        zt, rt = z[t], R[t]
        upd = act & np.isfinite(rt) & np.isfinite(zt)
        if upd.any():
            u = upd
            Px0 = P00[u] + P01[u] * zt
            Px1 = P01[u] + P11[u] * zt
            S = 1.0 + Px0 + Px1 * zt
            v = rt[u] - a[u] - b[u] * zt
            s2 = ss[u] / n[u]
            LL[t, u] = -0.5 * (np.log(2 * np.pi * s2 * S) + v * v / (s2 * S))
            a[u] += Px0 / S * v
            b[u] += Px1 / S * v
            P00[u] -= Px0 * Px0 / S
            P01[u] -= Px0 * Px1 / S
            P11[u] -= Px1 * Px1 / S
            ss[u] += v * v / S
            n[u] += 1.0
        s2 = ss[act] / n[act]
        BETA[t, act] = b[act] / scale
        SD[t, act] = np.sqrt(np.maximum(P11[act], 0) * s2) / scale
    mk = lambda A: pd.DataFrame(A, index=ret.index, columns=ret.columns)  # noqa: E731
    return mk(BETA), mk(SD), mk(LL)


def market_scale(mkt: pd.Series, train_end: str = TRAIN_END) -> float:
    return float(mkt.loc[:train_end].std())


def select_q(ret: pd.DataFrame, mkt: pd.Series, grid=Q_GRID, train_end: str = TRAIN_END):
    """q maximising the summed predictive log-likelihood on dates <= train_end. Only data up to train_end is used
    (data is truncated first). Returns (best_q, {q: loglik})."""
    r, m = ret.loc[:train_end], mkt.loc[:train_end]
    scale = market_scale(m, train_end)
    scores = {}
    for q in grid:
        scores[q] = float(np.nansum(kalman_filter(r, m, q, scale)[2].to_numpy()))
    return max(scores, key=scores.get), scores


def kalman_beta(ret: pd.DataFrame, mkt: pd.Series, q: float | None = None, train_end: str = TRAIN_END,
                grid=Q_GRID):
    """Filtered beta and posterior sd for every date; q chosen on <= train_end if not given.
    Returns (beta, beta_sd, q, scores)."""
    scores = {}
    if q is None:
        q, scores = select_q(ret, mkt, grid, train_end)
    beta, sd, _ = kalman_filter(ret, mkt, q, market_scale(mkt, train_end))
    return beta, sd, q, scores
