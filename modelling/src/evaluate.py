"""
The manual grid-search cross-validation loop, and model selection.

The brief lists three options for parameter selection and calls the manual
loop "best, since it is the most flexible". It is also the only one that hands
back OUT-OF-FOLD PREDICTIONS -- a single number from `cross_val_score` cannot
produce a directional-accuracy-by-quantile table, and cannot be reassembled
into a backtest path.

SELECTION RULE
--------------
Selection is on rank IC, not MAE. The strategy monetises the ordering of names
within a day; a model predicting ~0 everywhere has excellent MAE and no trading
value.

Two guards sit on top of it:

  Per-fold scoring, not pooled. A model that is excellent in one fold and
  useless in four is not a good model, and pooling hides that entirely.

  The one-standard-error rule. Nine models scoring between +0.020 and +0.028
  are not nine distinguishable models -- that spread is far smaller than the
  standard error of any one of them. Taking the maximum of nine correlated
  noisy estimates is upward-biased even when every model is equally good, and
  that bias is spent the moment the test set is touched. So: find the best mean
  fold IC, then among every model within one standard error of it, prefer the
  SIMPLEST. This is standard practice and it is the direct fix for the
  winner's-curse gap observed on the first pass.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone

from .metrics import rank_ic, report

__all__ = ["build", "cv_oof", "fold_scores", "run_grid", "run_roster",
           "select_one_se", "FAMILY_COMPLEXITY"]


# Simpler is better when scores tie. Ordering reflects how much freedom each
# family has to fit noise on 665 effective observations.
FAMILY_COMPLEXITY = {
    "baseline": 0,
    "penalised linear": 1,
    "distance": 2,
    "bagged trees": 3,
    "kernel": 4,
    "boosted trees": 5,
    "neural": 6,
}


def build(spec, params):
    """Instantiate a model with `params` applied.

    Everything is wrapped in a TransformedTargetRegressor, so each grid key
    picks up a `regressor__` prefix on the way in.
    """
    est = clone(spec["factory"]())
    if params:
        est.set_params(**{f"regressor__{k}": v for k, v in params.items()})
    return est


def cv_oof(spec, params, X, y_fit, y_true, dates, cv):
    """Fit one configuration across every fold; return out-of-fold predictions.

    Fitting uses `y_fit` (winsorised); `y_true` is only ever used for scoring.
    Folds may overlap in CPCV, so predictions are averaged where a row is
    predicted more than once.
    """
    total = np.zeros(len(y_true))
    count = np.zeros(len(y_true))
    for tr, va in cv.split(X, y_true, groups=dates):
        est = build(spec, params)
        est.fit(X.iloc[tr], y_fit[tr])
        total[va] += est.predict(X.iloc[va])
        count[va] += 1
    oof = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return oof


def fold_scores(spec, params, X, y_fit, y_true, dates, cv):
    """Rank IC computed separately in each fold. Exposes stability."""
    out = []
    for tr, va in cv.split(X, y_true, groups=dates):
        est = build(spec, params)
        est.fit(X.iloc[tr], y_fit[tr])
        p = est.predict(X.iloc[va])
        ic, _, _, _ = rank_ic(y_true[va], p, dates[va])
        out.append(ic if np.isfinite(ic) else 0.0)
    return np.asarray(out)


def run_grid(name, spec, X, y_fit, y_true, dates, cv, verbose=True):
    """Search one model's grid. Returns the best config by MEAN FOLD rank IC."""
    t0 = time.time()
    best = None
    for params in spec["grid"]:
        ics = fold_scores(spec, params, X, y_fit, y_true, dates, cv)
        mean_ic, sd = float(np.mean(ics)), float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0
        se = sd / np.sqrt(len(ics)) if len(ics) > 1 else np.nan
        if best is None or mean_ic > best["mean_fold_ic"]:
            best = {"params": params, "mean_fold_ic": mean_ic,
                    "fold_ic_sd": sd, "fold_ic_se": se, "fold_ics": ics}
    best["oof"] = cv_oof(spec, best["params"], X, y_fit, y_true, dates, cv)
    best["secs"] = time.time() - t0
    if verbose:
        pos = int((best["fold_ics"] > 0).sum())
        print(f"  {name:14s} meanFoldIC {best['mean_fold_ic']:+.4f} "
              f"(se {best['fold_ic_se']:.4f})  positive in {pos}/{len(best['fold_ics'])} folds"
              f"  [{best['secs']:.0f}s]")
    return best


def run_roster(roster, X, y_fit, y_true, dates, cv, verbose=True):
    """Search every model in the roster. Returns a results dict keyed by name."""
    results = {}
    for name, spec in roster.items():
        best = run_grid(name, spec, X, y_fit, y_true, dates, cv, verbose=verbose)
        m = np.isfinite(best["oof"])
        rep = report(y_true[m], best["oof"][m], dates[m], label=name, verbose=False)
        results[name] = {**best, **rep, "family": spec.get("family", "?"),
                         "scaled": spec.get("scaled", False)}
    return results


def select_one_se(results: dict) -> tuple[str, pd.DataFrame]:
    """Best mean fold IC, then the simplest model within one standard error.

    Returns (winner_name, the comparison table used to choose).
    """
    rows = []
    for name, r in results.items():
        rows.append({
            "model": name,
            "family": r["family"],
            "mean_fold_ic": r["mean_fold_ic"],
            "fold_ic_se": r["fold_ic_se"],
            "folds_positive": int((r["fold_ics"] > 0).sum()),
            "n_folds": len(r["fold_ics"]),
            "oof_RankIC": r["RankIC"],
            "oof_RankIC_t": r["RankIC_t"],
            "MAE": r["MAE"],
            "ProfitFactor": r["ProfitFactor"],
            "DirAcc": r["DirAcc"],
            "Skill": r["Skill"],
            "DegenDays": r["DegenDays"],
            "complexity": FAMILY_COMPLEXITY.get(r["family"], 9),
        })
    tab = pd.DataFrame(rows).sort_values("mean_fold_ic", ascending=False).reset_index(drop=True)

    top = tab.iloc[0]
    threshold = top["mean_fold_ic"] - top["fold_ic_se"]
    within = tab[tab["mean_fold_ic"] >= threshold].copy()
    within = within.sort_values(["complexity", "mean_fold_ic"],
                                ascending=[True, False])
    winner = within.iloc[0]["model"]

    tab["within_1se"] = tab["mean_fold_ic"] >= threshold
    tab["selected"] = tab["model"] == winner
    return winner, tab
