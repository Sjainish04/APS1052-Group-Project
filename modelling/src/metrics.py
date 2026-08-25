"""
The four metrics the brief requires, at validation and at test time.

    Mean absolute error
    Profit Factor
    Spearman Rank Correlation RHO
    Directional accuracy by quantile vs the majority-direction baseline

The directional-accuracy format follows the course reference notebook,
`YieldCurveForecastPmdarimaExogenous_ONESTEPAHEAD_DGS10_d0_d1.ipynb`
(cells 32-33), with one necessary adaptation.

    THE ADAPTATION. The reference target is a yield LEVEL, so it takes
    np.diff() to recover the direction of change. Our target is already a
    one-day log return -- it IS the change. Differencing would measure the
    change in the return rather than its direction, which is a different
    quantity. So sign(target) is used directly. The brief anticipates this:
    "remember to adapt it if using log returns".

Everything else matches the reference: quintiles are cut on the ABSOLUTE
ACTUAL move (Q1 tiny ... Q5 large), and the naive baseline is
always-predict-the-majority-direction, computed globally and within each
bucket.

Run `python -m modelling.src.metrics` for the self-test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

__all__ = [
    "mae", "profit_factor", "signal_profit_factor", "rank_ic",
    "global_majority_baseline", "directional_accuracy_by_quantile", "report",
]


# --------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------

def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def profit_factor(returns) -> float:
    """Gross profit / gross loss. >1 means winners outweigh losers.

    Returns inf where there are no losses at all, rather than silently
    becoming a large finite number.
    """
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    gain, loss = r[r > 0].sum(), -r[r < 0].sum()
    if loss == 0:
        return np.inf if gain > 0 else np.nan
    return float(gain / loss)


def signal_profit_factor(y_true, y_pred) -> float:
    """PF of the naive rule 'take the sign of the prediction'.

    This is Timothy Masters' VarScreen screening statistic, so it is also how
    single features are ranked during selection.
    """
    return profit_factor(np.sign(np.asarray(y_pred, float)) * np.asarray(y_true, float))


def rank_ic(y_true, y_pred, dates):
    """Mean DAILY cross-sectional Spearman.

    Returns (mean_ic, t_stat, n_days_scored, n_days_with_no_ordering).

    Computed per day then averaged -- never pooled across rows. Pooling would
    let a wide-dispersion day dominate, and would mix across-day variation into
    what is meant to be a within-day ranking score. The t-statistic uses the
    spread of DAILY ICs, so its degrees of freedom are days, not rows: with 20
    names per day, pooling would inflate it by roughly sqrt(20).

    A day on which the model returns the same number for every name scores
    ZERO rather than being skipped. Skipping silently rewards degeneracy -- a
    model that collapses to a constant on 80% of days would then be judged
    only on the 20% where it did something, and would win a comparison it
    should lose. This is not hypothetical: it happened, and an over-regularised
    ElasticNet won on it.
    """
    df = pd.DataFrame({"d": np.asarray(dates), "y": np.asarray(y_true, float),
                       "p": np.asarray(y_pred, float)}).dropna()
    ics, degenerate = [], 0
    for _, g in df.groupby("d"):
        if len(g) < 3 or g.y.nunique() < 2:
            continue                       # undefined through no fault of the model
        if g.p.nunique() < 2:
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


def daily_ic_series(y_true, y_pred, dates) -> pd.Series:
    """Per-day rank IC, indexed by date. Feeds the stability plots."""
    df = pd.DataFrame({"d": np.asarray(dates), "y": np.asarray(y_true, float),
                       "p": np.asarray(y_pred, float)}).dropna()
    out = {}
    for d, g in df.groupby("d"):
        if len(g) < 3 or g.y.nunique() < 2:
            continue
        if g.p.nunique() < 2:
            out[d] = 0.0
            continue
        ic = spearmanr(g.p, g.y).statistic
        out[d] = ic if np.isfinite(ic) else 0.0
    return pd.Series(out).sort_index()


# --------------------------------------------------------------------------
# Directional accuracy -- reference-notebook format
# --------------------------------------------------------------------------

def global_majority_baseline(y_true, y_pred) -> dict:
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    n_up, n_dn = int((y > 0).sum()), int((y < 0).sum())
    return {"n": len(y), "n_up": n_up, "n_down": n_dn,
            "majority": "up" if n_up >= n_dn else "down",
            "naive_hit_rate": max(n_up, n_dn) / len(y) * 100,
            "model_hit_rate": float((np.sign(y) == np.sign(p)).sum()) / len(y) * 100}


def directional_accuracy_by_quantile(y_true, y_pred, q: int = 5) -> pd.DataFrame:
    """Hit rate by ABSOLUTE-MOVE quantile vs the majority baseline in that bucket.

    duplicates='drop' mirrors the reference notebook: identical absolute moves
    produce duplicate bin edges, and qcut raises rather than merging them.
    """
    df = pd.DataFrame({"true": np.asarray(y_true, float),
                       "pred": np.asarray(y_pred, float)}).dropna()
    df["correct"] = np.sign(df["true"]) == np.sign(df["pred"])
    df["abs_move"] = df["true"].abs()
    df["q"] = pd.qcut(df["abs_move"], q=q, labels=False, duplicates="drop")

    bins = sorted(df["q"].dropna().unique())
    names = ["Q1 (tiny)", "Q2", "Q3", "Q4", "Q5 (large)"][:q]
    if len(bins) != len(names):
        names = [f"Q{i + 1}" for i in range(len(bins))]
        names[0] += " (tiny)"
        names[-1] += " (large)"
    df["q"] = df["q"].map(dict(zip(bins, names)))

    out = (df.groupby("q", observed=True)["correct"].agg(["mean", "count"])
             .rename(columns={"mean": "Hit Rate", "count": "N"}))
    out["% of Total"] = (out["N"] / len(df) * 100).round(1)
    out["Naive"] = (df.groupby("q", observed=True)["true"]
                    .apply(lambda g: max((g > 0).sum(), (g < 0).sum()) / len(g)))
    out["Skill"] = out["Hit Rate"] - out["Naive"]
    out["Mean |move|"] = df.groupby("q", observed=True)["abs_move"].mean()
    return out.reindex([n for n in names if n in out.index])


def report(y_true, y_pred, dates, label: str = "", q: int = 5, verbose: bool = True) -> dict:
    """All four required metrics, in the reference notebook's presentation order."""
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
              + (f", {ndeg} with no ordering)" if ndeg else ")"))
        print(f"  Naive hit rate      : {g['naive_hit_rate']:.2f}%  "
              f"(always predict '{g['majority']}')")
        print(f"  Model hit rate      : {g['model_hit_rate']:.2f}%")
        print(f"  Skill vs naive      : {res['Skill']:+.2f} pct points")
        print("\n  Directional accuracy by absolute-move quintile:")
        print(directional_accuracy_by_quantile(y_true, y_pred, q)
              .to_string(float_format=lambda x: f"{x:8.4f}"))
    res["by_quantile"] = directional_accuracy_by_quantile(y_true, y_pred, q)
    return res


# --------------------------------------------------------------------------

def _selftest():
    print("metrics self-test")
    rng = np.random.default_rng(0)

    # -- profit factor against a hand-computed case ------------------------
    pf = profit_factor([0.03, 0.03, -0.05])
    assert abs(pf - 1.2) < 1e-12, pf
    print(f"  [ok] profit factor = {pf:.4f} (0.06 gain / 0.05 loss)")

    # -- signed PF: a perfect predictor never loses ------------------------
    y = rng.normal(0, 0.02, 500)
    assert signal_profit_factor(y, y) == np.inf
    assert abs(signal_profit_factor(y, -y)) < 1e-12
    print("  [ok] signed PF: inf for a perfect signal, 0 for its exact inverse")

    # -- rank IC on a noisy copy of the target -----------------------------
    days = np.repeat(pd.date_range("2024-01-01", periods=60), 20)
    yy = rng.normal(0, 0.02, len(days))
    pp = yy + rng.normal(0, 0.01, len(days))
    ic, t, nd, ndeg = rank_ic(yy, pp, days)
    assert ic > 0.5 and t > 5 and nd == 60 and ndeg == 0
    print(f"  [ok] rank IC on a noisy copy: {ic:+.3f} (t {t:+.1f}, {nd} days)")

    # -- THE degeneracy guard ---------------------------------------------
    # A model constant on most days must be penalised, not excused. Build one
    # that orders perfectly on 10 days of 60 and is flat on the rest.
    const = np.zeros(len(days))
    mask = np.isin(days, pd.date_range("2024-01-01", periods=10))
    const[mask] = yy[mask]
    ic_d, t_d, nd_d, ndeg_d = rank_ic(yy, const, days)
    assert ndeg_d == 50, ndeg_d
    assert nd_d == 60, nd_d
    assert ic_d < 0.2, ic_d
    print(f"  [ok] degeneracy: {ndeg_d}/60 flat days scored 0, "
          f"IC diluted to {ic_d:+.3f} (not excused to ~1.0)")

    # -- market-wide degeneracy: PF is identically 1 on a demeaned target ---
    d2 = np.repeat(pd.date_range("2024-01-01", periods=80), 20)
    raw = rng.normal(0, 0.02, len(d2))
    demeaned = raw - pd.Series(raw).groupby(pd.Series(d2)).transform("mean").to_numpy()
    market_wide = pd.Series(d2).map(
        dict(zip(pd.unique(d2), rng.normal(0, 1, 80)))).to_numpy()
    pf_mw = signal_profit_factor(demeaned, market_wide)
    assert abs(pf_mw - 1.0) < 1e-9, pf_mw
    print(f"  [ok] market-wide feature scores PF = {pf_mw:.10f} exactly "
          "(sign is constant within a day; demeaned target sums to zero)")

    # -- directional accuracy table ----------------------------------------
    tab = directional_accuracy_by_quantile(yy, pp, q=5)
    assert len(tab) == 5 and tab["N"].sum() == len(yy)
    assert (tab["Hit Rate"] > 0.5).all()
    print(f"  [ok] quantile table: 5 buckets, N sums to {int(tab['N'].sum())}, "
          "hit rate above 0.5 in every bucket")

    # -- MAE ---------------------------------------------------------------
    assert abs(mae([1.0, 2.0], [1.5, 2.5]) - 0.5) < 1e-12
    print("  [ok] MAE = 0.5 on a hand-computed case")

    print("\nall metrics checks passed.")


if __name__ == "__main__":
    _selftest()
