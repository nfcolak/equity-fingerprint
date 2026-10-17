"""Multivariate MiniRocket transform (Dempster, Schmidt, Webb 2021, arXiv 2012.08791) in numpy + numba.

No sktime dependency. ``MiniRocket(num_features, seed).fit(X_train).transform(X)`` with ``X`` of shape
(n, channels, length) returns float32 PPV features of shape (n, n_features_).

Algorithm (as in sktime's ``MiniRocketMultivariate``):

* 84 fixed kernels of length 9; weights -1 everywhere except 2 at three of the nine positions
  (``C(9,3) = 84`` choices). Every kernel sums to zero, so a feature is invariant to the level of a series.
* Dilations ``unique(int(2 ** linspace(0, log2((L - 1) / 8), min(per_kernel, 32))))``, i.e. exponentially
  spaced up to ``(L - 1) / 8``. ``per_kernel = num_features // 84``; the features of a kernel are spread over
  the dilations in proportion to how often each (rounded) dilation occurs. Zero padding is used for every
  second dilation (``index % 2``), as in the reference code.
* Multivariate: every (dilation, kernel) combination convolves a random subset of channels and sums the
  convolution outputs. The subset size is ``floor(2 ** U(0, log2(min(C, 9) + 1)))`` and the channels are a
  random permutation prefix.
* Biases: for each (dilation, kernel) combination one randomly drawn *fit* example is convolved and the bias
  values are quantiles of its output at a low-discrepancy sequence (``frac(k * golden ratio)``).
  They therefore depend on the fit data (and the seed) only, never on the data passed to ``transform``.
* Feature = PPV, the proportion of convolution outputs (over the valid positions) that exceed the bias.

Deterministic for a given seed and fit data. Every output position of a feature uses input samples at or
before ``t + 4 * dilation`` of the window only, and windows are processed independently, so a window's
features depend on that window alone (the causality of the caller's windows is therefore preserved).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from numba import njit, prange

N_KERNELS = 84
KERNEL_IDX = np.array(list(combinations(range(9), 3)), dtype=np.int64)  # (84, 3): positions that get weight 2
GOLDEN = (np.sqrt(5.0) + 1.0) / 2.0
MAX_DILATIONS_PER_KERNEL = 32


@njit(cache=True)
def _conv(x, chans, nch, idx, d, padding, out):
    """Sum over ``chans[:nch]`` of the dilated kernel output; returns the valid range [lo, hi)."""
    length = x.shape[1]
    if padding:
        lo, hi = 0, length
    else:
        lo, hi = 4 * d, length - 4 * d
    for t in range(lo, hi):
        s_all = 0.0
        s_idx = 0.0
        for ci in range(nch):
            c = chans[ci]
            for j in range(9):
                u = t + (j - 4) * d
                if u >= 0 and u < length:
                    s_all += x[c, u]
            for q in range(3):
                u = t + (idx[q] - 4) * d
                if u >= 0 and u < length:
                    s_idx += x[c, u]
        out[t] = 3.0 * s_idx - s_all  # -1 on all nine taps, +2 on three of them
    return lo, hi


@njit(parallel=True, cache=True)
def _transform(X, dil, pad, kidx, combo_dil, combo_k, chan_n, chan_tab, feat_start, feat_n, bias, out):
    n, _, length = X.shape
    for i in prange(n):
        buf = np.empty(length)
        for cb in range(combo_dil.shape[0]):
            di = combo_dil[cb]
            lo, hi = _conv(X[i], chan_tab[cb], chan_n[cb], kidx[combo_k[cb]], dil[di], pad[di], buf)
            srt = np.sort(buf[lo:hi])
            m = hi - lo
            for f in range(feat_n[cb]):
                b = bias[feat_start[cb] + f]
                out[i, feat_start[cb] + f] = (m - np.searchsorted(srt, b, side="right")) / m


@njit(cache=True)
def _conv_py(x, chans, nch, idx, d, padding):
    out = np.zeros(x.shape[1])
    lo, hi = _conv(x, chans, nch, idx, d, padding, out)
    return out[lo:hi]


def _dilations(length: int, num_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Dilations and the number of features per (kernel, dilation), summing to (num_features // 84) per kernel."""
    per_kernel = max(num_features // N_KERNELS, 1)
    true_max = min(per_kernel, MAX_DILATIONS_PER_KERNEL)
    multiplier = per_kernel / true_max
    max_exp = np.log2((length - 1) / 8)
    dil, counts = np.unique(np.floor(2 ** np.linspace(0, max_exp, true_max)).astype(np.int64), return_counts=True)
    nf = (counts * multiplier).astype(np.int64)
    rest = per_kernel - int(nf.sum())
    i = 0
    while rest > 0:  # hand the remainder to the dilations in turn, as the reference does
        nf[i % len(nf)] += 1
        rest -= 1
        i += 1
    return dil, nf


def _quantiles(n: int) -> np.ndarray:
    return (np.arange(1, n + 1) * GOLDEN) % 1.0


class MiniRocket:
    """Multivariate MiniRocket. ``num_features`` is a target; the realised count is ``n_features_``."""

    def __init__(self, num_features: int = 5000, seed: int = 42):
        self.num_features, self.seed = int(num_features), int(seed)

    def fit(self, X: np.ndarray) -> "MiniRocket":
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError("X must have shape (n, channels, length)")
        n, c, length = X.shape
        if length < 9:
            raise ValueError("length must be >= 9")
        rng = np.random.default_rng(self.seed)
        dil, nf = _dilations(length, self.num_features)
        self.dilations_, self.feat_per_dilation_ = dil, nf
        self.padding_ = (np.arange(len(dil)) % 2).astype(np.int64)
        n_combo = len(dil) * N_KERNELS
        self.combo_dil_ = np.repeat(np.arange(len(dil)), N_KERNELS).astype(np.int64)
        self.combo_k_ = np.tile(np.arange(N_KERNELS), len(dil)).astype(np.int64)
        self.feat_n_ = nf[self.combo_dil_].astype(np.int64)
        self.feat_start_ = np.concatenate([[0], np.cumsum(self.feat_n_)[:-1]]).astype(np.int64)
        self.n_features_ = int(self.feat_n_.sum())
        # channel subsets
        max_ch = min(c, 9)
        self.chan_n_ = np.floor(2 ** rng.uniform(0, np.log2(max_ch + 1), n_combo)).astype(np.int64).clip(1, max_ch)
        self.chan_tab_ = np.stack([rng.permutation(c)[:max_ch] for _ in range(n_combo)]).astype(np.int64)
        # biases: one random fit example per combination, quantiles of its convolution output
        ex = rng.integers(0, n, n_combo)
        bias = np.empty(self.n_features_)
        for cb in range(n_combo):
            di = self.combo_dil_[cb]
            conv = _conv_py(X[ex[cb]], self.chan_tab_[cb], self.chan_n_[cb], KERNEL_IDX[self.combo_k_[cb]],
                            dil[di], self.padding_[di])
            q = _quantiles(self.feat_n_[cb])
            bias[self.feat_start_[cb]:self.feat_start_[cb] + self.feat_n_[cb]] = np.quantile(conv, q)
        self.bias_ = bias
        self.length_, self.n_channels_ = length, c
        return self

    def transform(self, X: np.ndarray, batch: int = 4096) -> np.ndarray:
        if X.ndim != 3 or X.shape[1] != self.n_channels_ or X.shape[2] != self.length_:
            raise ValueError(f"expected shape (n, {self.n_channels_}, {self.length_}), got {X.shape}")
        out = np.empty((X.shape[0], self.n_features_), dtype=np.float32)
        for s in range(0, X.shape[0], batch):
            xb = np.ascontiguousarray(X[s:s + batch], dtype=np.float64)
            _transform(xb, self.dilations_, self.padding_, KERNEL_IDX, self.combo_dil_, self.combo_k_,
                       self.chan_n_, self.chan_tab_, self.feat_start_, self.feat_n_, self.bias_,
                       out[s:s + batch])
        return out
