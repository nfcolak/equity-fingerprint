"""Preprocessing, models and metrics.

Models per target: baseline, linear_raw, linear_fe, lgbm_raw, lgbm_fe (+ linear_fe2, lgbm_fe2 for the
grouped targets sector / log_mcap when ``fe2_cols`` is given).
Everything is fitted on train; hyper-parameters / early stopping use validation; test is scored once.
"""
from __future__ import annotations

import itertools
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error, r2_score

from .features import LOG_FEATURES

warnings.filterwarnings("ignore", message=".*eval_set.*")
SEED = 42
RIDGE_ALPHAS = [0.01, 0.1, 1, 10, 100, 1000]
LOGIT_CS = [0.01, 0.1, 1, 10]
LGBM_GRID = [dict(num_leaves=nl, min_child_samples=mcs)
             for nl, mcs in itertools.product([8, 31], [20, 100, 400])]
MODELS = ["baseline", "linear_raw", "linear_fe", "lgbm_raw", "lgbm_fe"]
MODELS_V2 = MODELS + ["linear_fe2", "lgbm_fe2"]


# --------------------------------------------------------------------------- preprocessing
class Prep:
    """Winsorize (1%/99%), optionally log-transform skewed features, impute and scale: train statistics only."""

    def __init__(self, cols: list[str], linear: bool):
        self.cols, self.linear = list(cols), linear

    def _log(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.cols:
            if c in LOG_FEATURES:
                X[c] = np.log1p(X[c].clip(lower=0) / self.scale_[c])
        return X

    def fit(self, X: pd.DataFrame) -> "Prep":
        X = X[self.cols].astype(float)
        self.scale_ = {}
        for c in self.cols:
            pos = X[c][X[c] > 0]
            self.scale_[c] = float(pos.median()) if len(pos) else 1.0
        if self.linear:
            X = self._log(X)
        self.lo_, self.hi_ = X.quantile(0.01), X.quantile(0.99)
        Xc = X.clip(self.lo_, self.hi_, axis=1)
        self.med_ = Xc.median()
        self.mu_ = Xc.fillna(self.med_).mean()
        self.sd_ = Xc.fillna(self.med_).std().replace(0, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.cols].astype(float)
        if self.linear:
            X = self._log(X)
        X = X.clip(self.lo_, self.hi_, axis=1)
        if self.linear:
            X = (X.fillna(self.med_) - self.mu_) / self.sd_
        return X


# --------------------------------------------------------------------------- metrics
def regression_metrics(y, pred, with_qlike: bool, resid_var: float = 0.0) -> dict[str, float]:
    out = {"r2": r2_score(y, pred), "mae": mean_absolute_error(y, pred),
           "rmse": float(np.sqrt(mean_squared_error(y, pred)))}
    if with_qlike:
        # y, pred are log annualized vol. QLIKE on variance; the variance forecast is the log-normal mean
        # exp(2*pred + 2*resid_var) (resid_var = variance of validation residuals), not exp(2*pred).
        ratio = np.exp(2 * (np.asarray(y) - np.asarray(pred)) - 2 * resid_var)
        out["qlike"] = float(np.mean(ratio - np.log(ratio) - 1))
    return out


def classification_metrics(y, proba, classes) -> dict[str, float]:
    pred = np.asarray(classes)[np.argmax(proba, axis=1)]
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "log_loss": log_loss(y, np.clip(proba, 1e-12, 1), labels=list(classes))}


# --------------------------------------------------------------------------- fitting
def _lgbm_params(extra: dict) -> dict:
    return dict(learning_rate=0.05, n_estimators=3000, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=5, verbose=-1, **extra)


def fit_lgbm(Xtr, ytr, Xva, yva, classification: bool):
    best = None
    for g in LGBM_GRID:
        cls = lgb.LGBMClassifier if classification else lgb.LGBMRegressor
        m = cls(**_lgbm_params(g))
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        score = next(iter(m.best_score_["valid_0"].values()))  # l2 / multi_logloss: lower is better
        if best is None or score < best[0]:
            best = (score, m, {**g, "best_iteration": int(m.best_iteration_)})
    return best[1], best[2]


def fit_ridge(Xtr, ytr, Xva, yva):
    best = None
    for a in RIDGE_ALPHAS:
        m = Ridge(alpha=a).fit(Xtr, ytr)
        s = mean_squared_error(yva, m.predict(Xva))
        if best is None or s < best[0]:
            best = (s, m, {"alpha": a})
    return best[1], best[2]


def fit_logit(Xtr, ytr, Xva, yva):
    best = None
    for c in LOGIT_CS:
        m = LogisticRegression(C=c, max_iter=1000).fit(Xtr, ytr)
        s = log_loss(yva, m.predict_proba(Xva), labels=m.classes_)
        if best is None or s < best[0]:
            best = (s, m, {"C": c})
    return best[1], best[2]


def ticker_level_metrics(part: pd.DataFrame, y_col: str, proba, classes) -> dict[str, float]:
    """Static labels: average class probabilities over a ticker's observations, argmax, score per ticker."""
    P = pd.DataFrame(np.asarray(proba), index=part["ticker"].to_numpy()).groupby(level=0).mean()
    truth = part.groupby("ticker")[y_col].first().reindex(P.index)
    pred = np.asarray(classes)[np.argmax(P.to_numpy(), axis=1)]
    return {"ticker_accuracy": accuracy_score(truth, pred),
            "ticker_macro_f1": f1_score(truth, pred, average="macro"), "_n": len(P)}


def run_target(name: str, df: pd.DataFrame, y_col: str, raw_cols: list[str], fe_cols: list[str],
               classification: bool, with_qlike: bool = False, fe2_cols: list[str] | None = None,
               extra_sets: list[tuple[str, list[str]]] | None = None):
    """Fit all models on df[split=='train']; return (metric rows, fitted lgbm models + test frames, hp, classes)."""
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    classes = np.array(sorted(tr[y_col].unique())) if classification else None
    rows, info = [], {}

    def add(model, feats, split, part, pred, resid_var=0.0):
        met = (classification_metrics(part[y_col], pred, classes) if classification
               else regression_metrics(part[y_col], pred, with_qlike, resid_var))
        for k, v in met.items():
            rows.append(dict(target=name, model=model, features=feats, split=split,
                             metric=k, value=float(v), n_obs=len(part)))
        if classification:  # ticker-level rows: n_obs = number of tickers
            tm = ticker_level_metrics(part, y_col, pred, classes)
            n = tm.pop("_n")
            for k, v in tm.items():
                rows.append(dict(target=name, model=model, features=feats, split=split,
                                 metric=k, value=float(v), n_obs=n))

    # baseline
    for split, part in (("val", va), ("test", te)):
        if classification:
            prior = tr[y_col].value_counts(normalize=True).reindex(classes).fillna(0).to_numpy()
            proba = np.tile(prior, (len(part), 1))
            # majority-class prediction for accuracy / F1, class priors for log loss
            met = classification_metrics(part[y_col], proba, classes)
            for k, v in met.items():
                rows.append(dict(target=name, model="baseline", features="none", split=split,
                                 metric=k, value=float(v), n_obs=len(part)))
            tm = ticker_level_metrics(part, y_col, proba, classes)
            n = tm.pop("_n")
            for k, v in tm.items():
                rows.append(dict(target=name, model="baseline", features="none", split=split,
                                 metric=k, value=float(v), n_obs=n))
        else:
            base_v = float(np.var(va[y_col] - tr[y_col].mean()))
            add("baseline", "none", split, part, np.full(len(part), tr[y_col].mean()), base_v)

    fitted = {}
    sets = [("raw", raw_cols), ("fe", fe_cols)] + ([("fe2", fe2_cols)] if fe2_cols else [])
    sets += list(extra_sets or [])
    for feats, cols in sets:
        # linear
        pl = Prep(cols, linear=True).fit(tr)
        Xtr, Xva, Xte = (pl.transform(d) for d in (tr, va, te))
        if classification:
            m, hp = fit_logit(Xtr, tr[y_col], Xva, va[y_col])
            pf = m.predict_proba
        else:
            m, hp = fit_ridge(Xtr, tr[y_col], Xva, va[y_col])
            pf = m.predict
        info[f"linear_{feats}"] = hp
        rv = 0.0 if classification else float(np.var(va[y_col] - pf(Xva)))
        add(f"linear_{feats}", feats, "val", va, pf(Xva), rv)
        add(f"linear_{feats}", feats, "test", te, pf(Xte), rv)
        # lgbm
        pt = Prep(cols, linear=False).fit(tr)
        Xtr, Xva, Xte = (pt.transform(d) for d in (tr, va, te))
        ytr, yva = tr[y_col], va[y_col]
        m, hp = fit_lgbm(Xtr, ytr, Xva, yva, classification)
        info[f"lgbm_{feats}"] = hp
        pf = m.predict_proba if classification else m.predict
        rv = 0.0 if classification else float(np.var(va[y_col] - pf(Xva)))
        add(f"lgbm_{feats}", feats, "val", va, pf(Xva), rv)
        add(f"lgbm_{feats}", feats, "test", te, pf(Xte), rv)
        fitted[f"lgbm_{feats}"] = (m, Xte)
    return rows, fitted, info, classes
