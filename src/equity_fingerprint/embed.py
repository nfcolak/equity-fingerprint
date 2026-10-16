"""v3 sector features: which TRAIN stocks does a stock move with?

At every observation position the return correlation of each ticker with every train ticker is taken
over the 252 trading days ending at the position (inclusive), one correlation matrix per position.
From it:
  * kNN profile    prof_knn_<class>: mean of the top-M correlations with train tickers of that class;
                   prof_vote_<class>: correlation-weighted vote share of the class among the top-K
                   most correlated train tickers overall.
  * return embedding: the correlation vector with the train tickers is projected on the first D
                   eigenvectors of the train-period (<= 2021-12-31) correlation matrix of the train
                   tickers (prof_emb_<j>); prof_cos_<class> is the cosine similarity of the (cross-
                   sectionally centred) embedding with the class centroid of train-ticker embeddings.
Leakage: labels enter only through class membership of TRAIN tickers. A train ticker never correlates
with itself (diagonal removed) and is removed from its own class centroid (leave-one-out); val/test
tickers are never reference tickers. The eigen-basis is unsupervised and fitted on train tickers and
train-period dates only (same convention as the v1 PCA factors).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import LABELS, MIN_PRIOR_ROWS, TRAIN_END, Panel, _slug

WINDOW = 252
MIN_OVERLAP = 126  # minimum common non-missing days for a pair correlation

# Chosen on validation macro-F1 of linear_fe3 (seed 42) over m in {3,5,10}, K in {10,20}, d in {8,16}
# (best val macro-F1 0.380; test never used for the choice).
KNN_M = 10    # top-m correlations averaged per class
VOTE_K = 10   # top-K train neighbours that vote
EMB_D = 16    # embedding dimension


def _corr_to_train(X: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Correlation of every column of X (w x nt, NaN = missing) with the train columns; NaN if undefined."""
    M = ~np.isnan(X)
    cnt = M.sum(0)
    Xf = np.where(M, X, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        Xc = (Xf - Xf.sum(0) / cnt) * M
        sd = np.sqrt((Xc ** 2).sum(0) / (cnt - 1))
        ok = (cnt >= MIN_OVERLAP) & (sd > 1e-8)
        Z = np.where(ok, Xc / sd, 0.0)
        Mf = (M & ok).astype(float)
        n_ij = Mf.T @ Mf[:, train_idx]
        C = (Z.T @ Z[:, train_idx]) / (n_ij - 1)
    C = np.where((n_ij >= MIN_OVERLAP) & ok[:, None] & ok[train_idx][None, :], C, np.nan)
    return np.clip(C, -1.0, 1.0)


def _embedding_basis(p: Panel, ref: list[str], d: int) -> np.ndarray:
    """Top-d eigenvectors (n_ref x d) of the train-period correlation matrix of the reference tickers."""
    R = p.ret[ref]
    Rt = R[R.index <= TRAIN_END]
    sd = Rt.std().replace(0, np.nan)
    Z = ((Rt - Rt.mean()) / sd).clip(-6, 6).fillna(0.0).to_numpy()
    C = Z.T @ Z / (len(Z) - 1)
    w, V = np.linalg.eigh(C)
    return V[:, np.argsort(w)[::-1][:d]]


def sector_profile_features(p: Panel, pos: np.ndarray, train_tickers: list[str], sec_map: dict[str, str],
                            m: int = KNN_M, K: int = VOTE_K, d: int = EMB_D):
    """Co-movement-with-training-stocks features; returns (frame keyed by pos, ticker, feature names)."""
    ti = {t: j for j, t in enumerate(p.tickers)}
    ref = [t for t in train_tickers if t in ti and t in sec_map and p.ret[t].notna().sum() >= MIN_PRIOR_ROWS]
    ref_idx = np.array([ti[t] for t in ref])
    classes = sorted({sec_map[t] for t in ref})
    cls_of = np.array([classes.index(sec_map[t]) for t in ref])
    onehot = np.eye(len(classes))[cls_of]                      # n_ref x C
    V = _embedding_basis(p, ref, d)                              # n_ref x d
    Rv = p.ret.to_numpy()
    nt, nref, nc = len(p.tickers), len(ref), len(classes)
    # position of each ticker inside the reference set (-1: not a reference ticker)
    self_col = np.full(nt, -1)
    self_col[ref_idx] = np.arange(nref)

    knn = np.full((len(pos), nt, nc), np.nan)
    vote = np.full((len(pos), nt, nc), np.nan)
    emb = np.full((len(pos), nt, d), np.nan)
    cos = np.full((len(pos), nt, nc), np.nan)
    kk = min(K, nref - 1)
    for a, i in enumerate(pos):
        C = _corr_to_train(Rv[i - WINDOW + 1:i + 1], ref_idx)   # nt x nref
        C[ref_idx, np.arange(nref)] = np.nan                    # a train ticker never uses itself
        valid = ~np.isnan(C)
        Cf = np.where(valid, C, -np.inf)
        # kNN profile: mean of the top-m correlations within each class
        for c in range(nc):
            S = -np.sort(-Cf[:, cls_of == c], axis=1)[:, :m]
            ok = np.isfinite(S)
            n = ok.sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                knn[a, :, c] = np.where(n > 0, np.where(ok, S, 0.0).sum(1) / n, np.nan)
        # weighted vote among the top-K train tickers overall
        top = np.argpartition(-Cf, kk - 1, axis=1)[:, :kk]
        wts = np.take_along_axis(np.where(valid, np.clip(C, 0, None), 0.0), top, axis=1)
        tot = wts.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            vote[a] = np.where(tot[:, None] > 0, np.einsum("nk,nkc->nc", wts, onehot[top]) / tot[:, None], np.nan)
        # out-of-sample embedding from this window, then cosine to (leave-one-out) class centroids
        E = np.nan_to_num(C, nan=0.0) @ V                       # nt x d
        has = valid.sum(1) >= MIN_OVERLAP // 2
        ref_E = E[ref_idx]
        mu = ref_E.mean(0)
        Ec = E - mu
        csum = onehot.T @ (ref_E - mu)                          # C x d
        cn = onehot.sum(0)
        for c in range(nc):
            cent = np.broadcast_to(csum[c], (nt, d)).copy()
            n_c = np.full(nt, cn[c])
            mine = (self_col >= 0) & (cls_of[np.maximum(self_col, 0)] == c)
            cent[mine] -= Ec[mine]                              # leave-one-out centroid
            n_c[mine] -= 1
            den = np.linalg.norm(Ec, axis=1) * np.linalg.norm(cent, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                cos[a, :, c] = np.where((n_c > 0) & (den > 1e-12), (Ec * cent).sum(1) / den, np.nan)
        E = np.where(has[:, None], E, np.nan)
        cos[a][~has] = np.nan
        emb[a] = E

    data, names = {}, []
    for c, k in enumerate(classes):
        s = _slug(k)
        for key, arr, lab in (("knn", knn, f"Mean top-{m} correlation with train {k} stocks"),
                              ("vote", vote, f"Neighbour vote share of {k} (top {K})"),
                              ("cos", cos, f"Embedding cosine to {k} centroid")):
            data[f"prof_{key}_{s}"] = arr[:, :, c].ravel()
            LABELS[f"prof_{key}_{s}"] = lab
            names.append(f"prof_{key}_{s}")
    for j in range(d):
        data[f"prof_emb_{j + 1}"] = emb[:, :, j].ravel()
        LABELS[f"prof_emb_{j + 1}"] = f"Return embedding dim {j + 1}"
        names.append(f"prof_emb_{j + 1}")
    df = pd.DataFrame(data)
    df.insert(0, "ticker", np.tile(np.array(p.tickers, dtype=object), len(pos)))
    df.insert(0, "pos", np.repeat(pos, nt))
    return df, names


def fp_profile_features(comove: pd.DataFrame, train_tickers: list[str], sec_map: dict[str, str],
                        m: int = KNN_M, K: int = VOTE_K):
    """Co-movement features from learned comove vectors; returns (frame keyed by pos, ticker, feature names).

    ``comove`` has columns pos, ticker, [date,] fc_0..fc_<n-1> (unit-norm rows). On each ``pos`` the references
    are the labelled TRAIN tickers with a vector at that pos; cosine to them gives fpprof_knn_<class> (mean of
    the top-m per class, NaN if the class has no reference that day) and fpprof_vote_<class> (clipped-cosine
    weighted vote among the top-K references overall). A ticker is excluded from its own references.
    """
    fc = [c for c in comove.columns if c.startswith("fc_")]
    ref_set = {t for t in train_tickers if t in sec_map}
    classes = sorted({sec_map[t] for t in set(comove["ticker"]) & ref_set})
    nc = len(classes)
    cidx = {k: c for c, k in enumerate(classes)}
    pos_l, tick_l, knn_l, vote_l = [], [], [], []
    for pos, g in comove.groupby("pos", sort=True):
        tick = g["ticker"].to_numpy()
        Z = g[fc].to_numpy(np.float64)
        isref = np.array([t in ref_set for t in tick])
        n = len(g)
        knn = np.full((n, nc), np.nan)
        vote = np.full((n, nc), np.nan)
        if isref.any():
            rt = tick[isref]
            cls_of = np.array([cidx[sec_map[t]] for t in rt])
            S = Z @ Z[isref].T                                        # n x nref
            S[tick[:, None] == rt[None, :]] = np.nan                  # a ticker never references itself
            valid = ~np.isnan(S)
            Sf = np.where(valid, S, -np.inf)
            for c in range(nc):
                top = -np.sort(-Sf[:, cls_of == c], axis=1)[:, :m]
                ok = np.isfinite(top)
                cnt = ok.sum(1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    knn[:, c] = np.where(cnt > 0, np.where(ok, top, 0.0).sum(1) / cnt, np.nan)
            top = np.argsort(-Sf, axis=1)[:, :min(K, S.shape[1])]
            wts = np.take_along_axis(np.where(valid, np.clip(S, 0, None), 0.0), top, axis=1)
            tot = wts.sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                vote = np.where(tot[:, None] > 0,
                                np.einsum("nk,nkc->nc", wts, np.eye(nc)[cls_of][top]) / tot[:, None], np.nan)
        pos_l.append(np.full(n, pos))
        tick_l.append(tick)
        knn_l.append(knn)
        vote_l.append(vote)
    knn_a = np.concatenate(knn_l) if knn_l else np.empty((0, nc))
    vote_a = np.concatenate(vote_l) if vote_l else np.empty((0, nc))
    data, names = {}, []
    for c, k in enumerate(classes):
        s = _slug(k)
        for key, arr, lab in (("knn", knn_a, f"Mean top-{m} comove cosine to train {k} stocks"),
                              ("vote", vote_a, f"Comove neighbour vote share of {k} (top {K})")):
            data[f"fpprof_{key}_{s}"] = arr[:, c]
            LABELS[f"fpprof_{key}_{s}"] = lab
            names.append(f"fpprof_{key}_{s}")
    df = pd.DataFrame(data)
    df.insert(0, "ticker", np.concatenate(tick_l) if tick_l else np.array([], dtype=object))
    df.insert(0, "pos", np.concatenate(pos_l) if pos_l else np.array([], dtype=int))
    return df, names
