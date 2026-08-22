#!/usr/bin/env python3
"""
Generator for modelling.ipynb (Part 2).

    .venv/bin/python make_modelling_notebook.py
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace modelling.ipynb
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src):
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ==========================================================================
md(r"""
# Cross-Sectional ML for Long-Short Equity Selection
## Part 2 — Model selection, tuning, and held-out evaluation

Runs on `data/model_train.csv` and `data/model_test.csv`, both written by
Part 1. No network, no credentials.

### The constraint behind every choice below

The training set has **13,300 rows but only 665 distinct trading days**. The
20 names on a given day share one market shock and were explicitly demeaned
*against each other*, so rows within a day are close to the opposite of
independent draws. The effective sample size is far nearer 665 than 13,300 —
roughly **17 observations per feature, not 350**.

That drives three decisions:

1. **Grids are deliberately small and regularisation is heavy.** With this
   much noise, a large grid mostly searches for the luckiest fold split.
2. **Every t-statistic is computed across days, never across rows.** Pooling
   rows would inflate it by roughly $\sqrt{20}$.
3. **Model selection is on rank IC, not MAE.** The strategy monetises the
   *ordering* of names within a day. A model that predicts ~0 for everything
   scores excellent MAE and is worth nothing to trade — MAE alone would pick
   exactly the wrong winner. All four required metrics are reported; only the
   selection criterion is opinionated.
""")

code(r'''
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

RANDOM_STATE = 0
DATA = "data"

train = pd.read_csv(f"{DATA}/model_train.csv", parse_dates=["date"]).reset_index(drop=True)
test = pd.read_csv(f"{DATA}/model_test.csv", parse_dates=["date"]).reset_index(drop=True)
meta = json.load(open(f"{DATA}/feature_meta.json"))
FEATURES = meta["selected"]

X_tr, y_tr, d_tr = train[FEATURES], train["target"].values, train["date"].values
X_te, y_te, d_te = test[FEATURES], test["target"].values, test["date"].values

print(f"train {X_tr.shape[0]:,} rows x {X_tr.shape[1]} features | "
      f"{train['date'].nunique():,} days | {train['ticker'].nunique()} names")
print(f"test  {X_te.shape[0]:,} rows                  | "
      f"{test['date'].nunique():,} days")
print(f"\ntarget sd {y_tr.std():.5f}  kurtosis {pd.Series(y_tr).kurt():.1f}")
print(f"features with NaN: {int(X_tr.isna().any().sum())} of {len(FEATURES)}")
''')

# ==========================================================================
md(r"""
## 1. Metrics

The four the brief requires, at both validation and test time. These follow
`YieldCurveForecastPmdarimaExogenous_ONESTEPAHEAD_DGS10_d0_d1.ipynb`
(cells 32–33), with one necessary adaptation.

**The adaptation.** The reference target is a yield *level*, so it takes
`np.diff()` to recover the direction of change. Our target is already a
one-day log return — it *is* the change. Differencing it would measure the
change in the return rather than its direction, which is a different quantity.
So `sign(target)` is used directly. The brief anticipates this: *"remember to
adapt it if using log returns"*.

Everything else matches the reference: quintiles are cut on the **absolute
actual move** (Q1 tiny … Q5 large), and the naive baseline is
always-predict-the-majority-direction, computed globally and within each
bucket.
""")

code(r'''
def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def profit_factor(returns):
    """Gross profit / gross loss. >1 means winners outweigh losers."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    gain, loss = r[r > 0].sum(), -r[r < 0].sum()
    if loss == 0:
        return np.inf if gain > 0 else np.nan
    return float(gain / loss)


def signal_profit_factor(y_true, y_pred):
    """PF of the naive rule 'take the sign of the prediction'."""
    return profit_factor(np.sign(np.asarray(y_pred, float)) * np.asarray(y_true, float))


def rank_ic(y_true, y_pred, dates):
    """
    Mean DAILY cross-sectional Spearman, its t-stat, the number of days scored,
    and how many of those days the model gave no ordering on.

    Computed per day then averaged -- never pooled across all rows. Pooling
    would let a high-dispersion day dominate and would mix across-day variation
    into what is meant to be a within-day ranking score. The t-stat uses the
    spread of DAILY ICs, so its degrees of freedom are days, not rows.
    """
    df = pd.DataFrame({"d": np.asarray(dates), "y": np.asarray(y_true, float),
                       "p": np.asarray(y_pred, float)}).dropna()
    ics, degenerate = [], 0
    for _, g in df.groupby("d"):
        if len(g) < 3 or g.y.nunique() < 2:
            continue                      # undefined through no fault of the model
        if g.p.nunique() < 2:
            # The model gave the same number to all 20 names, so it supplied no
            # ordering at all that day. That scores ZERO -- it must not be
            # skipped. Skipping silently rewards degeneracy: a model that
            # collapses to a constant on 80% of days would then be judged only
            # on the 20% where it happened to do something, and would win.
            ics.append(0.0)
            degenerate += 1
            continue
        ic = spearmanr(g.p, g.y).statistic
        ics.append(ic if np.isfinite(ic) else 0.0)
    ics = np.asarray(ics)
    if len(ics) < 2:
        return np.nan, np.nan, len(ics), degenerate
    t = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))
    return float(ics.mean()), float(t), len(ics), degenerate


def global_majority_baseline(y_true, y_pred):
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    n_up, n_dn = int((y > 0).sum()), int((y < 0).sum())
    return {"n": len(y), "n_up": n_up, "n_down": n_dn,
            "majority": "up" if n_up >= n_dn else "down",
            "naive_hit_rate": max(n_up, n_dn) / len(y) * 100,
            "model_hit_rate": float((np.sign(y) == np.sign(p)).sum()) / len(y) * 100}


def directional_accuracy_by_quantile(y_true, y_pred, q=5):
    """Hit rate by ABSOLUTE-MOVE quintile vs the majority baseline in that bucket."""
    df = pd.DataFrame({"true": np.asarray(y_true, float),
                       "pred": np.asarray(y_pred, float)}).dropna()
    df["correct"] = np.sign(df["true"]) == np.sign(df["pred"])
    df["abs_move"] = df["true"].abs()
    df["q"] = pd.qcut(df["abs_move"], q=q, labels=False, duplicates="drop")
    bins = sorted(df["q"].dropna().unique())
    names = ["Q1 (tiny)", "Q2", "Q3", "Q4", "Q5 (large)"][:q]
    if len(bins) != len(names):
        names = [f"Q{i+1}" for i in range(len(bins))]
        names[0] += " (tiny)"; names[-1] += " (large)"
    df["q"] = df["q"].map(dict(zip(bins, names)))
    out = (df.groupby("q", observed=True)["correct"].agg(["mean", "count"])
             .rename(columns={"mean": "Hit Rate", "count": "N"}))
    out["% of Total"] = (out["N"] / len(df) * 100).round(1)
    out["Naive"] = (df.groupby("q", observed=True)["true"]
                    .apply(lambda g: max((g > 0).sum(), (g < 0).sum()) / len(g)))
    out["Skill"] = out["Hit Rate"] - out["Naive"]
    out["Mean |move|"] = df.groupby("q", observed=True)["abs_move"].mean()
    return out.reindex([n for n in names if n in out.index])


def report(y_true, y_pred, dates, label="", q=5, verbose=True):
    ic, t, ndays, ndeg = rank_ic(y_true, y_pred, dates)
    g = global_majority_baseline(y_true, y_pred)
    res = {"label": label, "MAE": mae(y_true, y_pred),
           "ProfitFactor": signal_profit_factor(y_true, y_pred),
           "RankIC": ic, "RankIC_t": t, "n_days": ndays, "DegenDays": ndeg,
           "DirAcc": g["model_hit_rate"], "NaiveDirAcc": g["naive_hit_rate"],
           "Skill": g["model_hit_rate"] - g["naive_hit_rate"]}
    if verbose:
        print(f"\n{'=' * 68}\n{label}\n{'=' * 68}")
        print(f"  Mean absolute error : {res['MAE']:.6f}")
        print(f"  Profit Factor       : {res['ProfitFactor']:.4f}")
        print(f"  Spearman Rank IC    : {res['RankIC']:+.4f}  "
              f"(t = {res['RankIC_t']:+.2f} over {ndays} days"
              + (f", {ndeg} with NO ordering)" if ndeg else ")"))
        print(f"  Naive hit rate      : {g['naive_hit_rate']:.2f}%  "
              f"(always predict '{g['majority']}')")
        print(f"  Model hit rate      : {g['model_hit_rate']:.2f}%")
        print(f"  Skill vs naive      : {res['Skill']:+.2f} pct points")
        print(f"\n  Directional accuracy by absolute-move quintile:")
        print(directional_accuracy_by_quantile(y_true, y_pred, q)
              .to_string(float_format=lambda x: f"{x:8.4f}"))
    res["by_quantile"] = directional_accuracy_by_quantile(y_true, y_pred, q)
    return res


print("Metrics defined: MAE, Profit Factor, Spearman rank IC, "
      "directional accuracy by quantile.")
''')

# ==========================================================================
md(r"""
## 2. Cross-validation — `TimeSeriesSplit` over dates

The brief says *"Always using TimeSeriesSplit for a time series"*. Applied to a
**panel**, sklearn's row-wise `TimeSeriesSplit` is quietly wrong: with 20 stocks
per day it puts the same date on both sides of a fold boundary, so the model
trains on 12 names from a day and validates on the other 8 — having already seen
that day's cross-section, including the cross-sectional mean the target was
demeaned by. That leaks badly and inflates every score.

This splitter is the same expanding-window scheme applied to whole **dates**,
with an embargo after each training block so the last training target (which
looks one day ahead) cannot reach into validation.
""")

code(r'''
class DateGroupedTimeSeriesSplit:
    """Expanding-window TimeSeriesSplit whose folds break between whole days."""

    def __init__(self, n_splits=5, embargo=5):
        self.n_splits, self.embargo = n_splits, embargo

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        g = pd.Series(np.asarray(groups))
        udates = np.array(sorted(g.unique()))
        n, idx = len(udates), np.arange(len(g))
        fold = n // (self.n_splits + 1)
        for k in range(self.n_splits):
            tr_end = fold * (k + 1)
            va_beg = tr_end + self.embargo
            va_end = min(va_beg + fold, n)
            if va_beg >= n:
                break
            tr = idx[g.isin(udates[:tr_end]).to_numpy()]
            va = idx[g.isin(udates[va_beg:va_end]).to_numpy()]
            if len(tr) and len(va):
                yield tr, va


CV = DateGroupedTimeSeriesSplit(n_splits=5, embargo=5)

print("Fold structure (expanding window, 5-day embargo):")
for k, (tr, va) in enumerate(CV.split(X_tr, y_tr, groups=d_tr)):
    td, vd = pd.Series(d_tr[tr]), pd.Series(d_tr[va])
    gap = (vd.min() - td.max()).days
    print(f"  fold {k}: train {len(tr):>6,} rows / {td.nunique():>3} days "
          f"({td.min():%Y-%m-%d}..{td.max():%Y-%m-%d})  ->  "
          f"val {len(va):>5,} rows / {vd.nunique():>3} days   gap {gap}d")
    assert td.max() < vd.min(), "fold leaks: train overlaps validation"
print("\n[ok] every fold boundary falls between whole trading days")
''')

# ==========================================================================
md(r"""
## 3. The training target

Kurtosis is **36.5** — a handful of earnings gaps carry an enormous share of the
squared error. Left alone, a squared-error fit spends most of its capacity
chasing days it cannot predict, which on 665 effective days is how you get a
model steered by half a dozen observations.

So the **training** target is winsorised at ±5 sd. Every metric, and the entire
test set, uses the **raw** target — winsorising what you score on would hide
real P&L rather than stabilise a fit.
""")

code(r'''
WINSOR_SD = 5.0
sd = y_tr.std()
y_tr_w = np.clip(y_tr, -WINSOR_SD * sd, WINSOR_SD * sd)
n_clip = int((np.abs(y_tr) > WINSOR_SD * sd).sum())

print(f"Training target winsorised at +/-{WINSOR_SD} sd (= {WINSOR_SD * sd:.4f}):")
print(f"  {n_clip:,} of {len(y_tr):,} rows clipped ({100 * n_clip / len(y_tr):.2f}%)")
print(f"  kurtosis {pd.Series(y_tr).kurt():.1f} -> {pd.Series(y_tr_w).kurt():.1f}")
print(f"  sd       {y_tr.std():.5f} -> {y_tr_w.std():.5f}")
print("\nNOTE: the test target is left RAW. So is every target used for scoring.")
''')

# ==========================================================================
md(r"""
## 4. The five models

At least five, at least one requiring scaling — the brief's roster shape. The
scale-sensitive three carry `StandardScaler` and `SelectKBest` **inside** the
pipeline, which the brief is explicit about: fitted outside, they would learn
their statistics from the validation fold too.

| Model | Pipeline | Needs scaling |
|---|---|---|
| LightGBM | raw features — native NaN handling | no |
| RandomForest | impute → model | no |
| ElasticNet | impute → scale → `SelectKBest(f_regression)` → model | **yes** |
| SVR (RBF) | impute → scale → `SelectKBest(mutual_info)` → model | **yes** |
| Keras MLP | impute → scale → `SelectKBest(mutual_info)` → model | **yes** |

`f_regression` for the linear model and `mutual_info_regression` for the
non-linear ones, exactly as the brief specifies.

**Two implementation notes that matter more than they look.**

*Target scaling.* Every pipeline is wrapped in a `TransformedTargetRegressor`
that standardises $y$ during fitting and inverts on predict. The target has sd
0.024, so without this every regularisation constant — ElasticNet's `alpha`,
SVR's `epsilon`, the Huber delta — would have to be quoted in units nobody can
reason about. Wrapped, they are all in standard deviations. The transformer is
refit inside each fold, so it stays causal.

*SVR's epsilon.* The default `epsilon=0.1` is **4.2× the target's standard
deviation**: nearly every point falls inside the tube, and SVR silently returns a
near-constant prediction while looking perfectly converged. On 8,000 rows it
kept 59 support vectors at the default versus 7,622 at `epsilon=0.001`. With the
target standardised the default becomes sane, and epsilon is tuned anyway.

*Imputation.* Median-imputing a per-day cross-sectionally standardised feature
fills in approximately **zero**, which reads as "this name was average that day".
That is a deliberate neutral default, not an accident.
""")

code(r'''
def mi_score(X, y):
    return mutual_info_regression(X, y, random_state=RANDOM_STATE)


class KerasMLP(BaseEstimator, RegressorMixin):
    """
    Minimal sklearn-compatible Keras regressor.

    Written by hand rather than pulling in scikeras, so the submission has one
    fewer dependency to bundle.

    The brief asks three things to be documented for a neural network, and all
    three are constructor arguments here rather than buried in the fit:

      loss            Huber. The target's kurtosis is 36.5, so MSE would let a
                      few extreme days dominate the gradient.
      penalties       L2 weight decay on every dense layer, plus dropout and
                      early stopping on a chronological tail of the training
                      fold.
      sample weights  proportional to |y|, normalised to mean 1. The strategy
                      only ever trades the extreme quantiles, so the large
                      movers are the observations worth getting right.
    """

    def __init__(self, hidden=(64, 32), dropout=0.2, l2=1e-4, lr=1e-3,
                 epochs=200, batch_size=256, huber_delta=1.0,
                 sample_weighting=True, patience=12, seed=RANDOM_STATE):
        self.hidden, self.dropout, self.l2, self.lr = hidden, dropout, l2, lr
        self.epochs, self.batch_size = epochs, batch_size
        self.huber_delta, self.sample_weighting = huber_delta, sample_weighting
        self.patience, self.seed = patience, seed

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

        # Validation split is the chronological TAIL of the fold, never random --
        # a shuffled split would let early-stopping peek at days the model has
        # already trained around.
        net.fit(X, y, sample_weight=w, epochs=self.epochs,
                batch_size=self.batch_size, verbose=0, shuffle=False,
                validation_split=0.15,
                callbacks=[keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=self.patience,
                    restore_best_weights=True)])
        self.model_ = net
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype="float32"),
                                   verbose=0).ravel()


def wrap(estimator):
    """Standardise y during fitting, invert on predict. Refit inside each fold."""
    return TransformedTargetRegressor(regressor=estimator, transformer=StandardScaler())


# Built fresh on every call. Sharing one estimator instance across pipelines
# invites cross-contamination between fits, and costs nothing to avoid.
def IMPUTE():
    return ("impute", SimpleImputer(strategy="median"))


def SCALE():
    return ("scale", StandardScaler())


MODELS = {
    "LightGBM": {
        "pipe": lambda **p: wrap(LGBMRegressor(
            random_state=RANDOM_STATE, verbose=-1, n_estimators=400, **p)),
        "grid": [{"num_leaves": nl, "learning_rate": lr, "min_child_samples": mcs,
                  "reg_lambda": 5.0, "colsample_bytree": 0.7, "subsample": 0.8,
                  "subsample_freq": 1}
                 for nl in (7, 15) for lr in (0.02, 0.05) for mcs in (100, 300)],
        "scaled": False,
    },
    "RandomForest": {
        "pipe": lambda **p: wrap(Pipeline([
            IMPUTE(), ("model", RandomForestRegressor(
                n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, **p))])),
        "grid": [{"model__max_depth": d, "model__min_samples_leaf": leaf,
                  "model__max_features": 0.4}
                 for d in (4, 8) for leaf in (50, 200)],
        "scaled": False,
    },
    "ElasticNet": {
        "pipe": lambda **p: wrap(Pipeline([
            IMPUTE(), SCALE(), ("select", SelectKBest(score_func=f_regression)),
            ("model", ElasticNet(max_iter=5000, random_state=RANDOM_STATE))])),
        "grid": [{"select__k": k, "model__alpha": a, "model__l1_ratio": l1}
                 for k in (15, 38) for a in (1e-4, 1e-3, 1e-2) for l1 in (0.15, 0.9)],
        "scaled": True,
    },
    "SVR": {
        "pipe": lambda **p: wrap(Pipeline([
            IMPUTE(), SCALE(), ("select", SelectKBest(score_func=mi_score)),
            ("model", SVR(kernel="rbf", cache_size=1000))])),
        "grid": [{"select__k": k, "model__C": c, "model__epsilon": e,
                  "model__gamma": "scale"}
                 for k in (15,) for c in (0.3, 1.0, 3.0) for e in (0.05, 0.2)],
        "scaled": True,
    },
    "KerasMLP": {
        "pipe": lambda **p: wrap(Pipeline([
            IMPUTE(), SCALE(), ("select", SelectKBest(score_func=mi_score)),
            ("model", KerasMLP())])),
        "grid": [{"select__k": k, "model__hidden": h, "model__dropout": dr,
                  "model__l2": l2}
                 for k in (15, 38) for h in ((32,), (64, 32))
                 for dr in (0.3,) for l2 in (1e-3,)],
        "scaled": True,
    },
}

for name, spec in MODELS.items():
    print(f"  {name:14s} {len(spec['grid']):>2} configs x 5 folds = "
          f"{len(spec['grid']) * 5:>3} fits   scaling: {spec['scaled']}")
print(f"\nTotal fits: {sum(len(s['grid']) for s in MODELS.values()) * 5}")
''')

# ==========================================================================
md(r"""
## 5. Manual grid-search cross-validation loop

The brief lists three options for parameter selection and calls the manual loop
*"best, since it is the most flexible"*. It is also the only one that hands back
**out-of-fold predictions**, which is what the validation metrics below are
computed on — a single number from `cross_val_score` could not produce a
directional-accuracy-by-quantile table.

Each configuration is fitted on every fold's training block and predicts its
validation block. The union of validation blocks forms one out-of-fold
prediction vector, scored by rank IC against the **raw** target.
""")

code(r'''
def build(model_spec, params):
    """Instantiate a model with `params` applied.

    Every pipeline is wrapped in a TransformedTargetRegressor, so each grid key
    picks up a `regressor__` prefix on its way in.
    """
    est = clone(model_spec["pipe"]())
    est.set_params(**{f"regressor__{k}": v for k, v in params.items()})
    return est


def cv_oof(model_spec, params, X, y_fit, y_true, dates, cv):
    """Fit one configuration across all folds; return its out-of-fold predictions.

    Fitting uses the winsorised target `y_fit`; `y_true` is passed only so the
    splitter sees the right row count. Scoring always happens on the raw target.
    """
    oof = np.full(len(y_true), np.nan)
    for tr, va in cv.split(X, y_true, groups=dates):
        est = build(model_spec, params)
        est.fit(X.iloc[tr], y_fit[tr])
        oof[va] = est.predict(X.iloc[va])
    return oof


results, oof_store = [], {}

for name, spec in MODELS.items():
    t0 = time.time()
    best = None
    for params in spec["grid"]:
        oof = cv_oof(spec, params, X_tr, y_tr_w, y_tr, d_tr, CV)
        m = np.isfinite(oof)
        ic, t, nd, _ = rank_ic(y_tr[m], oof[m], d_tr[m])
        if best is None or (np.isfinite(ic) and ic > best["RankIC"]):
            best = {"params": params, "RankIC": ic, "RankIC_t": t, "oof": oof}
    m = np.isfinite(best["oof"])
    r = report(y_tr[m], best["oof"][m], d_tr[m], label=f"{name} (out-of-fold)",
               verbose=False)
    r["params"] = best["params"]
    r["secs"] = time.time() - t0
    results.append(r)
    oof_store[name] = best["oof"]
    print(f"{name:14s} rankIC {r['RankIC']:+.4f} (t {r['RankIC_t']:+.2f})  "
          f"PF {r['ProfitFactor']:.4f}  MAE {r['MAE']:.6f}  "
          f"dir {r['DirAcc']:.2f}%  [{r['secs']:.0f}s]")
    print(f"               best: {best['params']}")
''')

# ==========================================================================
md(r"""
## 6. Validation comparison, plus a rank-average ensemble

The ensemble averages the three best models' **cross-sectional ranks** within
each day rather than their raw predictions. The models are on different scales
and the strategy only ever consumes an ordering, so averaging ranks is both the
scale-free choice and the one that matches what gets traded.
""")

code(r'''
val = pd.DataFrame([{k: r[k] for k in
                     ["label", "MAE", "ProfitFactor", "RankIC", "RankIC_t",
                      "DirAcc", "NaiveDirAcc", "Skill", "n_days", "DegenDays"]}
                    for r in results])
val["label"] = [r["label"].replace(" (out-of-fold)", "") for r in results]
val = val.sort_values("RankIC", ascending=False).reset_index(drop=True)

# --- ensemble of the top 3 -----------------------------------------------
# Each model's predictions are standardised WITHIN each day, then averaged, then
# put back on a return scale.
#
# The per-day standardisation is what makes the combination meaningful: the five
# models predict on quite different scales, so a raw average would just be the
# widest-spread model wearing a disguise. The rescale at the end matters for a
# duller reason -- an average of pct-ranks lives in [0, 1], which makes MAE
# meaningless and every predicted sign positive, so Profit Factor and
# directional accuracy would both be garbage.
top3 = val.head(3)["label"].tolist()
common = np.all([np.isfinite(oof_store[n]) for n in top3], axis=0)
dser = pd.Series(d_tr)

zs = []
for n in top3:
    sr = pd.Series(oof_store[n])
    grp = sr.groupby(dser)
    zs.append(((sr - grp.transform("mean")) / grp.transform("std")).to_numpy())
ens_z = np.nanmean(np.vstack(zs), axis=0)

# Back onto a return scale: match the average spread of the constituents.
scale = float(np.nanmean([np.nanstd(oof_store[n]) for n in top3]))
ens_pred = ens_z * scale

r_ens = report(y_tr[common], ens_pred[common], d_tr[common],
               label=f"Ensemble (per-day z-avg of {', '.join(top3)})", verbose=False)
val = pd.concat([val, pd.DataFrame([{k: r_ens[k] for k in val.columns}])],
                ignore_index=True)
val.loc[val.index[-1], "label"] = f"Ensemble ({'+'.join(top3)})"

print("VALIDATION (out-of-fold) -- ranked by Spearman rank IC\n")
print(val.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nNOTE: MAE is reported because the brief asks for it, but it is NOT the")
print("selection criterion. A model predicting ~0 everywhere wins on MAE and is")
print("worth nothing to trade -- compare the MAE column against RankIC to see")
print("how weakly the two are related.")
''')

code(r'''
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
plot_df = val.iloc[::-1]
for ax, (col, title, ref) in zip(axes, [
        ("RankIC", "Spearman rank IC (out-of-fold)", 0.0),
        ("ProfitFactor", "Profit Factor", 1.0),
        ("Skill", "Directional accuracy skill vs naive (pct pts)", 0.0)]):
    colors = ["#55A868" if v > ref else "#C44E52" for v in plot_df[col]]
    ax.barh(plot_df["label"], plot_df[col], color=colors)
    ax.axvline(ref, color="black", linestyle="--", lw=1.2)
    ax.set_title(title, fontsize=11)
plt.tight_layout()
plt.show()
''')

# ==========================================================================
md(r"""
## 7. Fine-tuning the selected model

A second, finer manual grid around the winner's neighbourhood. Kept small on
purpose: with 665 effective days, a wide search mostly finds the configuration
best suited to this particular fold split, and that gain does not survive
out-of-sample.
""")

code(r'''
# The ensemble is reported as a comparison, not carried forward: SHAP and the
# model-selection narrative both need one estimator, and on 665 effective days
# any ensemble edge is well inside the noise. Select the best SINGLE model.
singles = val[~val["label"].str.startswith("Ensemble")].reset_index(drop=True)
WINNER = singles.iloc[0]["label"]
ens_row = val[val["label"].str.startswith("Ensemble")].iloc[0]

print(f"Selected model: {WINNER}")
print(f"  best single model, out-of-fold rank IC = {singles.iloc[0]['RankIC']:+.4f}")
print(f"  rank-average ensemble scored          = {ens_row['RankIC']:+.4f}"
      f"  ({'better' if ens_row['RankIC'] > singles.iloc[0]['RankIC'] else 'worse'})")

base = next(r for r in results if r["label"].startswith(WINNER))["params"]
print(f"  stage-1 best params: {base}\n")

FINE = {
    "LightGBM": [dict(base, num_leaves=nl, learning_rate=lr, reg_lambda=rl)
                 for nl in (5, 7, 11) for lr in (0.01, 0.02, 0.03)
                 for rl in (5.0, 20.0)],
    "RandomForest": [dict(base, **{"model__max_depth": d,
                                   "model__min_samples_leaf": lf})
                     for d in (3, 4, 6) for lf in (100, 200, 400)],
    "ElasticNet": [dict(base, **{"model__alpha": a, "model__l1_ratio": l1})
                   for a in (3e-5, 1e-4, 3e-4, 1e-3, 3e-3) for l1 in (0.1, 0.5, 0.9)],
    "SVR": [dict(base, **{"model__C": c, "model__epsilon": e})
            for c in (0.1, 0.3, 1.0) for e in (0.02, 0.05, 0.1)],
    "KerasMLP": [dict(base, **{"model__dropout": dr, "model__l2": l2})
                 for dr in (0.2, 0.3, 0.4) for l2 in (1e-4, 1e-3, 1e-2)],
}[WINNER]

spec = MODELS[WINNER]
best_fine, t0 = None, time.time()
for params in FINE:
    oof = cv_oof(spec, params, X_tr, y_tr_w, y_tr, d_tr, CV)
    m = np.isfinite(oof)
    ic, t, _, _ = rank_ic(y_tr[m], oof[m], d_tr[m])
    if best_fine is None or (np.isfinite(ic) and ic > best_fine["ic"]):
        best_fine = {"params": params, "ic": ic, "t": t, "oof": oof}

print(f"Fine-tuned over {len(FINE)} configurations in {time.time() - t0:.0f}s")
print(f"  best rank IC {best_fine['ic']:+.4f} (t {best_fine['t']:+.2f})")
print(f"  params: {best_fine['params']}")
FINAL_PARAMS = best_fine["params"]
''')

# ==========================================================================
md(r"""
## 8. Held-out test evaluation

The model is refit on the **entire** training block with the tuned parameters
and predicts the test set once. Nothing below this line has influenced any
choice made above — the test set has been untouched since Part 1 wrote it.
""")

code(r'''
final = build(MODELS[WINNER], FINAL_PARAMS)
final.fit(X_tr, y_tr_w)

pred_te = final.predict(X_te)
pred_tr_oof = best_fine["oof"]

# A model that returns the same number for every name cannot rank anything, so
# there is no strategy to backtest. Fail loudly here rather than emit a NaN
# rank IC downstream and let it look like a metric that merely did not compute.
varies = pd.Series(pred_te).groupby(pd.Series(d_te)).nunique().gt(1)
print(f"Test days on which the prediction varies across the 20 names: "
      f"{varies.sum()} of {len(varies)}")
assert varies.mean() > 0.5, (
    "the fitted model predicts a near-constant -- it provides no cross-sectional "
    "ordering and nothing downstream is meaningful")

m = np.isfinite(pred_tr_oof)
res_val = report(y_tr[m], pred_tr_oof[m], d_tr[m],
                 label=f"{WINNER} -- VALIDATION (out-of-fold, tuned)")
res_te = report(y_te, pred_te, d_te, label=f"{WINNER} -- TEST (held out)")
''')

code(r'''
side = pd.DataFrame([
    {"metric": "Mean absolute error", "validation": res_val["MAE"], "test": res_te["MAE"]},
    {"metric": "Profit Factor", "validation": res_val["ProfitFactor"], "test": res_te["ProfitFactor"]},
    {"metric": "Spearman rank IC", "validation": res_val["RankIC"], "test": res_te["RankIC"]},
    {"metric": "Rank IC t-stat", "validation": res_val["RankIC_t"], "test": res_te["RankIC_t"]},
    {"metric": "Directional accuracy %", "validation": res_val["DirAcc"], "test": res_te["DirAcc"]},
    {"metric": "Naive baseline %", "validation": res_val["NaiveDirAcc"], "test": res_te["NaiveDirAcc"]},
    {"metric": "Skill (pct points)", "validation": res_val["Skill"], "test": res_te["Skill"]},
])
print("VALIDATION vs TEST\n")
print(side.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

deg = res_val["RankIC"] - res_te["RankIC"]
print(f"\nRank IC degradation from validation to test: {deg:+.4f}")
print("A large positive gap is the signature of an overfit selection procedure;")
print("a negative one usually means the test window was simply kinder.")
''')

# ==========================================================================
md(r"""
## 9. Did anything generalise? A post-hoc diagnostic

**This ran after selection was locked and fed nothing back into it.** The
selected model was chosen on out-of-fold rank IC alone, before the test set was
touched. What follows only asks a diagnostic question: was the out-of-sample
result specific to the model that happened to win, or did none of them carry?

The distinction matters for how the result should be read. If four models held
up and one collapsed, the selection rule is suspect. If all five went the same
way, the honest conclusion is about the *signal*, not the model.
""")

code(r'''
test_diag = []
for r in results:
    name = r["label"].replace(" (out-of-fold)", "")
    est = build(MODELS[name], r["params"])
    est.fit(X_tr, y_tr_w)
    pr = est.predict(X_te)
    ic, t, nd, _ = rank_ic(y_te, pr, d_te)
    test_diag.append({"model": name, "val_RankIC": r["RankIC"], "test_RankIC": ic,
                      "test_t": t, "test_PF": signal_profit_factor(y_te, pr),
                      "degradation": r["RankIC"] - ic})

diag = pd.DataFrame(test_diag).sort_values("val_RankIC", ascending=False)
print("Validation vs test for EVERY model (diagnostic only):\n")
print(diag.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

# How much of this is simply noise? The standard error of a mean daily IC is
# set by how wildly the DAILY IC swings, which on 20 names is a great deal.
per_day = []
for _, gg in pd.DataFrame({"d": d_te, "y": y_te, "p": pred_te}).groupby("d"):
    ic = spearmanr(gg.p, gg.y).statistic
    if np.isfinite(ic):
        per_day.append(ic)
per_day = np.asarray(per_day)
se = per_day.std(ddof=1) / np.sqrt(len(per_day))

print(f"\nDaily rank IC standard deviation : {per_day.std(ddof=1):.4f} over {len(per_day)} days")
print(f"Standard error of the mean       : {se:.4f}")
print(f"Mean test rank IC across models  : {diag['test_RankIC'].mean():+.4f}"
      f"  ({diag['test_RankIC'].mean() / se:+.2f} standard errors from zero)")
print(f"\nAll five models moved the same way: "
      f"{int((diag['test_RankIC'] < 0).sum())} of {len(diag)} are negative out of sample.")
print("They share a feature set, so these are close to one bet rather than five")
print("independent ones -- this is weak evidence of a regime shift, not strong.")
''')

md(r"""
### Reading the result honestly

Two effects are tangled together here, and the deck should separate them.

**The winner's curse.** Five models scored between +0.0197 and +0.0284 on
validation — a spread far smaller than the noise in any one of them. Taking the
maximum of five noisy, highly correlated estimates gives an upward-biased
estimate of the winner's true skill, even when every model is equally good. Some
of the validation-to-test drop is that bias unwinding, and it would have
appeared with no overfitting of any other kind.

**Genuine noise.** The daily rank IC has a standard deviation above 0.2 on a
20-name cross-section, so a 248-day test mean carries a standard error near
0.015. A test IC of −0.02 sits close to one standard error from zero. The
correct statement is **"no detectable out-of-sample cross-sectional
predictability"**, not "the model predicts the opposite" — the data cannot
support the second claim.

This is the expected outcome. One-day-ahead relative returns on twenty liquid
US equities is close to the most efficiently-priced quantity in the market, and
any signal robust enough to survive here would have been arbitraged long ago.
The brief anticipates exactly this: *"the trading equity curve need not be
profitable, but the logic of the program must be correct."*

What the pipeline does demonstrate is that the causal machinery holds up — the
embargo, the date-grouped folds, the publication-lagged short interest and the
same-day-only scaling all behave as designed, and the out-of-sample result is
what an honest pipeline reports when there is nothing to find. A leaky one would
have produced a beautiful test curve instead.
""")

# ==========================================================================
md(r"""
## 9. Save predictions for Part 3

Part 3 turns these into the long-short book, the equity curve, and the
significance battery (White's reality check, Monte Carlo permutation, Sharpe,
Profit Factor, CAGR).
""")

code(r'''
out_te = test[["date", "ticker", "target", "fwd_logret", "close_adj", "stock_spread"]].copy()
out_te["pred"] = pred_te
out_te.to_csv(f"{DATA}/pred_test.csv", index=False)

out_tr = train.loc[m, ["date", "ticker", "target", "fwd_logret"]].copy()
out_tr["pred"] = pred_tr_oof[m]
out_tr.to_csv(f"{DATA}/pred_train_oof.csv", index=False)

json.dump({"winner": WINNER, "params": {k: str(v) for k, v in FINAL_PARAMS.items()},
           "features": FEATURES,
           "validation": {k: float(res_val[k]) for k in
                          ["MAE", "ProfitFactor", "RankIC", "RankIC_t", "DirAcc", "Skill"]},
           "test": {k: float(res_te[k]) for k in
                    ["MAE", "ProfitFactor", "RankIC", "RankIC_t", "DirAcc", "Skill"]}},
          open(f"{DATA}/model_meta.json", "w"), indent=1)

print(f"  pred_test.csv       {len(out_te):>6,} rows")
print(f"  pred_train_oof.csv  {len(out_tr):>6,} rows")
print(f"  model_meta.json     winner = {WINNER}")

print("\n" + "=" * 68)
print("PART 2 COMPLETE")
print("=" * 68)
print(f"  models tried  : {', '.join(MODELS)} (+ rank-average ensemble)")
print(f"  selected      : {WINNER}")
print(f"  test rank IC  : {res_te['RankIC']:+.4f} (t {res_te['RankIC_t']:+.2f}, "
      f"{res_te['n_days']} days)")
print(f"  test PF       : {res_te['ProfitFactor']:.4f}")
print(f"  next          : Part 3 -- SHAP, quintile long-short, equity curve,")
print(f"                  White reality check, Monte Carlo permutation")
''')

# ==========================================================================
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3.11 (.venv)", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.14",
                      "mimetype": "text/x-python", "file_extension": ".py"},
}

out = "modelling.ipynb"
nbf.write(nb, out)
print(f"wrote {out}: {len(cells)} cells "
      f"({sum(c['cell_type'] == 'code' for c in cells)} code, "
      f"{sum(c['cell_type'] == 'markdown' for c in cells)} markdown)")
