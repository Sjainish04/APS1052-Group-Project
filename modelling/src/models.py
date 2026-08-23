"""
Model roster and pipelines.

The brief requires at least five models with at least one needing scaled
inputs. Nine are run here, spanning four genuinely different inductive biases
-- boosted trees, bagged trees, penalised linear, and distance/kernel/neural --
so that "the trees won" is a finding rather than an artefact of only having
tried trees.

    boosted trees     LightGBM, HistGradientBoosting
    bagged trees      RandomForest, ExtraTrees
    penalised linear  Ridge, ElasticNet
    other             SVR (RBF), kNN, Keras MLP

PIPELINE DISCIPLINE
-------------------
The brief is emphatic: "Do not use StandardScaler and/or SelectKBest OUTSIDE
the grid search pipeline". Fitted outside, both would learn their statistics
from the validation fold as well, and every score would be optimistic. Here
they are pipeline steps, refit inside every fold.

`f_regression` scores the linear models and `mutual_info_regression` the
non-linear ones, which is the split the brief specifies.

TARGET SCALING
--------------
Every estimator is wrapped in a `TransformedTargetRegressor` that standardises
y during fitting and inverts on predict. The target's standard deviation is
0.024, so unwrapped, every regularisation constant would have to be quoted in
units nobody can reason about -- and SVR's default `epsilon=0.1` would be 4.2x
the target's entire spread, silently keeping 59 support vectors out of 8,000
and returning a near-constant while looking converged. Wrapped, every constant
is in standard deviations. The transformer is refit inside each fold.

IMPUTATION
----------
Eight features carry NaN (thin option chains, single-member sectors). LightGBM
and HistGradientBoosting handle that natively and get no imputer. The rest get
a median impute -- and note what that means for an already per-day
cross-sectionally standardised feature: the median is approximately zero, so
the fill reads as "this name was average that day". A deliberate neutral
default, not an accident.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.feature_selection import (SelectKBest, f_regression,
                                       mutual_info_regression)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from lightgbm import LGBMRegressor

RANDOM_STATE = 0

__all__ = ["KerasMLP", "build_roster", "wrap", "mi_score", "RANDOM_STATE"]


def mi_score(X, y):
    """SelectKBest scorer for the non-linear models (named, so it pickles)."""
    return mutual_info_regression(X, y, random_state=RANDOM_STATE)


def wrap(estimator):
    """Standardise y during fit, invert on predict. Refit inside every fold."""
    return TransformedTargetRegressor(regressor=estimator, transformer=StandardScaler())


def _impute():
    return ("impute", SimpleImputer(strategy="median"))


def _scale():
    return ("scale", StandardScaler())


# --------------------------------------------------------------------------
# Neural network
# --------------------------------------------------------------------------

class KerasMLP(BaseEstimator, RegressorMixin):
    """Minimal sklearn-compatible Keras regressor.

    Hand-written rather than pulling in scikeras, so the submission has one
    fewer dependency to bundle.

    The brief asks three things to be documented for a neural network, and all
    three are constructor arguments rather than buried inside `fit`:

        loss            Huber. The raw target's kurtosis is 36.5, so squared
                        error would spend most of the gradient on a handful of
                        earnings gaps nobody can predict.
        penalties       L2 weight decay on every dense layer, plus dropout and
                        early stopping.
        sample weights  proportional to |y|, normalised to mean 1, switchable
                        via `sample_weighting` so the choice can be ablated
                        rather than asserted. The strategy only ever trades the
                        extreme quantiles, so large movers are the observations
                        worth getting right.

    Early stopping validates on the chronological TAIL of the training fold.
    A shuffled split would let it stop on days the model has already trained
    around, which is a quiet way to leak.
    """

    def __init__(self, hidden=(64, 32), dropout=0.3, l2=1e-3, lr=1e-3,
                 epochs=200, batch_size=256, huber_delta=1.0,
                 sample_weighting=True, patience=12, seed=RANDOM_STATE):
        self.hidden = hidden
        self.dropout = dropout
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.huber_delta = huber_delta
        self.sample_weighting = sample_weighting
        self.patience = patience
        self.seed = seed

    def fit(self, X, y):
        import keras
        from keras import layers, regularizers
        keras.utils.set_random_seed(self.seed)

        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32").ravel()

        net = keras.Sequential([keras.Input(shape=(X.shape[1],))])
        for h in self.hidden:
            net.add(layers.Dense(h, activation="relu",
                                 kernel_regularizer=regularizers.l2(self.l2)))
            net.add(layers.Dropout(self.dropout))
        net.add(layers.Dense(1))
        net.compile(optimizer=keras.optimizers.Adam(self.lr),
                    loss=keras.losses.Huber(delta=self.huber_delta))

        w = None
        if self.sample_weighting:
            w = np.abs(y)
            w = w / w.mean() if w.mean() > 0 else np.ones_like(y)

        net.fit(X, y, sample_weight=w, epochs=self.epochs,
                batch_size=self.batch_size, verbose=0, shuffle=False,
                validation_split=0.15,
                callbacks=[keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=self.patience,
                    restore_best_weights=True)])
        self.model_ = net
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype="float32"), verbose=0).ravel()


# --------------------------------------------------------------------------
# Roster
# --------------------------------------------------------------------------

def build_roster(n_features: int, fast: bool = False) -> dict:
    """Model name -> {factory, grid, family, scaled}.

    `fast=True` trims the two slowest grids (SVR, MLP) for smoke runs.
    """
    k_all = n_features

    roster = {
        # ---------------- boosted trees --------------------------------
        "LightGBM": {
            "factory": lambda: wrap(LGBMRegressor(
                random_state=RANDOM_STATE, verbose=-1, n_estimators=400,
                colsample_bytree=0.7, subsample=0.8, subsample_freq=1)),
            "grid": [{"num_leaves": nl, "learning_rate": lr,
                      "min_child_samples": mcs, "reg_lambda": rl}
                     for nl in (7, 15) for lr in (0.02, 0.05)
                     for mcs in (100, 300) for rl in (5.0, 20.0)],
            "family": "boosted trees", "scaled": False,
        },
        "HistGBM": {
            "factory": lambda: wrap(HistGradientBoostingRegressor(
                random_state=RANDOM_STATE, max_iter=400,
                early_stopping=False)),
            "grid": [{"max_leaf_nodes": ml, "learning_rate": lr,
                      "min_samples_leaf": msl, "l2_regularization": l2}
                     for ml in (7, 15) for lr in (0.02, 0.05)
                     for msl in (100, 300) for l2 in (1.0, 10.0)],
            "family": "boosted trees", "scaled": False,
        },
        # ---------------- bagged trees ---------------------------------
        "RandomForest": {
            "factory": lambda: wrap(Pipeline([
                _impute(), ("model", RandomForestRegressor(
                    n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))])),
            "grid": [{"model__max_depth": d, "model__min_samples_leaf": lf,
                      "model__max_features": mf}
                     for d in (4, 8) for lf in (50, 200) for mf in (0.3, 0.6)],
            "family": "bagged trees", "scaled": False,
        },
        "ExtraTrees": {
            "factory": lambda: wrap(Pipeline([
                _impute(), ("model", ExtraTreesRegressor(
                    n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))])),
            "grid": [{"model__max_depth": d, "model__min_samples_leaf": lf,
                      "model__max_features": mf}
                     for d in (4, 8) for lf in (50, 200) for mf in (0.3, 0.6)],
            "family": "bagged trees", "scaled": False,
        },
        # ---------------- penalised linear (need scaling) ---------------
        "Ridge": {
            "factory": lambda: wrap(Pipeline([
                _impute(), _scale(),
                ("select", SelectKBest(score_func=f_regression)),
                ("model", Ridge(random_state=RANDOM_STATE))])),
            "grid": [{"select__k": k, "model__alpha": a}
                     for k in (15, k_all)
                     for a in (1.0, 10.0, 100.0, 1000.0)],
            "family": "penalised linear", "scaled": True,
        },
        "ElasticNet": {
            "factory": lambda: wrap(Pipeline([
                _impute(), _scale(),
                ("select", SelectKBest(score_func=f_regression)),
                ("model", ElasticNet(max_iter=5000, random_state=RANDOM_STATE))])),
            "grid": [{"select__k": k, "model__alpha": a, "model__l1_ratio": l1}
                     for k in (15, k_all)
                     for a in (1e-4, 1e-3, 1e-2)
                     for l1 in (0.15, 0.9)],
            "family": "penalised linear", "scaled": True,
        },
        # ---------------- other (need scaling) --------------------------
        "SVR": {
            "factory": lambda: wrap(Pipeline([
                _impute(), _scale(),
                ("select", SelectKBest(score_func=mi_score)),
                ("model", SVR(kernel="rbf", cache_size=1000))])),
            "grid": [{"select__k": k, "model__C": c, "model__epsilon": e,
                      "model__gamma": "scale"}
                     for k in (15,) for c in (0.1, 0.3, 1.0)
                     for e in (0.02, 0.05, 0.2)],
            "family": "kernel", "scaled": True,
        },
        "kNN": {
            "factory": lambda: wrap(Pipeline([
                _impute(), _scale(),
                ("select", SelectKBest(score_func=mi_score)),
                ("model", KNeighborsRegressor(n_jobs=-1))])),
            "grid": [{"select__k": k, "model__n_neighbors": nn,
                      "model__weights": w}
                     for k in (10, 15) for nn in (100, 300, 800)
                     for w in ("uniform", "distance")],
            "family": "distance", "scaled": True,
        },
        "KerasMLP": {
            "factory": lambda: wrap(Pipeline([
                _impute(), _scale(),
                ("select", SelectKBest(score_func=mi_score)),
                ("model", KerasMLP())])),
            "grid": [{"select__k": k, "model__hidden": h,
                      "model__dropout": dr, "model__l2": l2}
                     for k in (15, k_all) for h in ((32,), (64, 32))
                     for dr in (0.3,) for l2 in (1e-3, 1e-2)],
            "family": "neural", "scaled": True,
        },
    }

    if fast:
        for name in ("SVR", "KerasMLP", "kNN"):
            roster[name]["grid"] = roster[name]["grid"][:2]
        for name in ("LightGBM", "HistGBM", "RandomForest", "ExtraTrees",
                     "Ridge", "ElasticNet"):
            roster[name]["grid"] = roster[name]["grid"][:3]

    return roster
