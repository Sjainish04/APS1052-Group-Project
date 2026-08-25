"""
Baselines.

Without these, "our model scored a rank IC of +0.03" is an uninterpretable
number. The question a reader actually has is *compared to what* -- and there
are three answers worth having:

    ZeroPredictor       predicts a constant. It scores the BEST MAE of anything
                        in the study and is worth exactly nothing to trade. It
                        exists to make visible, in a table, why MAE cannot be
                        the selection criterion.

    RandomPredictor     draws noise. Fixes the zero point: with 665 days and 20
                        names, what rank IC does pure chance produce? Run over
                        many seeds it gives an empirical null, which is a far
                        more honest reference than "0.0".

    BestSingleFeature   the single strongest feature from the Part-1 Profit
                        Factor screen, sign-corrected, used raw as the
                        prediction. This is the one that matters. If nine
                        tuned models cannot beat shorting one z-scored column,
                        the machine learning added nothing, and the deck should
                        say so rather than quietly omit the comparison.

All three implement fit/predict so they drop into the same evaluation loop as
the real models, which keeps the comparison honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

__all__ = ["ZeroPredictor", "RandomPredictor", "BestSingleFeature",
           "random_ic_null"]


class ZeroPredictor(BaseEstimator, RegressorMixin):
    """Predicts the training mean for everything. Optimal MAE, zero ordering."""

    def fit(self, X, y):
        self.const_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.const_)


class RandomPredictor(BaseEstimator, RegressorMixin):
    """Gaussian noise on the scale of the training target."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X, y):
        self.sd_ = float(np.std(y))
        return self

    def predict(self, X):
        rng = np.random.default_rng(self.seed)
        return rng.normal(0.0, self.sd_ if self.sd_ > 0 else 1.0, len(X))


class BestSingleFeature(BaseEstimator, RegressorMixin):
    """Use the strongest single feature, sign-corrected, as the prediction.

    Selection happens inside `fit`, so it only ever sees the training fold --
    picking the feature on the full sample would be exactly the look-ahead the
    rest of the pipeline works to avoid.

    Ranking is by |Profit Factor - 1| rather than by Profit Factor, because a
    feature at 0.92 is not weak: it is a working signal pointing the other way,
    and the sign correction below flips it. Ranking on the raw value would pick
    the feature closest to useless.
    """

    def __init__(self, feature_names=None):
        self.feature_names = feature_names

    def fit(self, X, y):
        Xdf = pd.DataFrame(X, columns=self.feature_names) if self.feature_names is not None \
            else pd.DataFrame(X)
        y = np.asarray(y, float)

        best, best_edge, best_sign = None, -np.inf, 1.0
        for col in Xdf.columns:
            v = Xdf[col].to_numpy(float)
            ok = np.isfinite(v) & np.isfinite(y)
            if ok.sum() < 100 or np.unique(v[ok]).size < 3:
                continue
            r = np.sign(v[ok]) * y[ok]
            gain, loss = r[r > 0].sum(), -r[r < 0].sum()
            if loss <= 0:
                continue
            pf = gain / loss
            edge = abs(pf - 1.0)
            if edge > best_edge:
                best, best_edge, best_sign = col, edge, (1.0 if pf > 1 else -1.0)

        self.feature_ = best
        self.sign_ = best_sign
        self.pf_ = 1.0 + best_edge * best_sign
        self.scale_ = float(np.std(y))
        return self

    def predict(self, X):
        Xdf = pd.DataFrame(X, columns=self.feature_names) if self.feature_names is not None \
            else pd.DataFrame(X)
        v = Xdf[self.feature_].to_numpy(float)
        v = np.nan_to_num(v, nan=0.0)
        sd = np.std(v)
        return self.sign_ * (v / sd if sd > 0 else v) * self.scale_


def random_ic_null(y_true, dates, n_draws: int = 500, seed: int = 0):
    """Empirical null distribution of the rank IC under pure chance.

    Shuffles predictions WITHIN each day, so it preserves the exact panel shape
    -- same number of days, same cross-section size, same target distribution --
    and destroys only the ordering. That makes it the right null for a
    cross-sectional ranking score.

    Returns (draws, mean, sd, two-sided 95% interval).
    """
    from .metrics import rank_ic

    rng = np.random.default_rng(seed)
    y = np.asarray(y_true, float)
    d = np.asarray(dates)
    order = pd.Series(np.arange(len(y))).groupby(pd.Series(d))

    draws = np.empty(n_draws)
    for i in range(n_draws):
        pred = np.empty(len(y))
        for _, idx in order.groups.items():
            idx = np.asarray(idx)
            pred[idx] = rng.normal(0, 1, len(idx))
        draws[i] = rank_ic(y, pred, d)[0]

    lo, hi = np.percentile(draws, [2.5, 97.5])
    return draws, float(draws.mean()), float(draws.std(ddof=1)), (float(lo), float(hi))
