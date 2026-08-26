#!/usr/bin/env python3
"""
Does moving to a WEEKLY horizon help?

Two experiments, because "weekly rebalancing" is ambiguous and the two readings
answer different questions.

  A) SAME daily-horizon model, traded weekly.
     The prediction still forecasts tomorrow, but positions are held 5 days.
     Turnover falls ~5x, so costs fall ~5x. This isolates the COST effect --
     and it is a deliberately mismatched setup: a 1-day forecast held 5 days.

  B) WEEKLY-horizon target, traded weekly.
     The target is the 5-day-forward cross-sectionally demeaned return, so the
     forecast horizon and the holding period agree. This is what the brief means
     by "weekly data is less noisy than daily data", and it is the real test.

The features are IDENTICAL in both, and identical to the daily run -- they are
per-day cross-sectional z-scores and do not depend on the target at all. Only
the target and the trading cadence change, which makes this a controlled
comparison rather than a new study.

CAVEAT, stated up front: overlapping 5-day targets are autocorrelated by
construction, so a rank-IC t-statistic computed over all days is inflated. Every
t-stat below is therefore ALSO reported on a non-overlapping subsample (every
5th day), which is the honest one.

    .venv/bin/python weekly_experiment.py
"""

import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, ".")
from modelling.src.models import build_roster
from modelling.src.evaluate import build, cv_oof, fold_scores
from modelling.src.metrics import rank_ic, signal_profit_factor, mae
from modelling.src.splitters import DateGroupedTimeSeriesSplit

DATA = "data"
RANDOM_STATE = 0
HORIZON = 5


def banner(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


# --------------------------------------------------------------------------
# Panel: reuse the already-built features, swap in a 5-day target
# --------------------------------------------------------------------------
banner("PANEL")

train = pd.read_csv(f"{DATA}/model_train.csv", parse_dates=["date"])
test = pd.read_csv(f"{DATA}/model_test.csv", parse_dates=["date"])
FEATURES = json.load(open(f"{DATA}/feature_meta.json"))["selected"]
panel = pd.concat([train, test], ignore_index=True).sort_values(["ticker", "date"])

prices = pd.read_csv(f"{DATA}/prices.csv", parse_dates=["date"])
prices = prices[prices["ticker"].isin(panel["ticker"].unique())].copy()
prices["logret"] = np.log1p(prices["dlyret"])
prices = prices.sort_values(["ticker", "date"])

# 5-day forward total log return: cumulative, then difference forward.
cum = prices.groupby("ticker")["logret"].cumsum()
prices["fwd5"] = cum.groupby(prices["ticker"]).shift(-HORIZON) - cum
panel = panel.merge(prices[["ticker", "date", "fwd5"]], on=["ticker", "date"], how="left")

# Cross-sectionally demean, same as the daily target.
panel["target5"] = panel["fwd5"] - panel.groupby("date")["fwd5"].transform("mean")
# Drop ONLY rows with no target. Sparse features (thin option chains,
# single-member sectors) legitimately carry NaN and are handled by the pipeline
# imputers -- dropping them here would remove individual names from a day and
# break the per-day zero-sum the demeaning just established.
panel = panel.dropna(subset=["target5"])

chk = panel.groupby("date")["target5"].mean().abs().max()
assert chk < 1e-10, f"demeaning failed: {chk}"
print(f"rows {len(panel):,} | days {panel['date'].nunique():,} | names {panel['ticker'].nunique()}")
print(f"1-day target sd {panel['target'].std():.5f}  kurtosis {panel['target'].kurt():.1f}")
print(f"5-day target sd {panel['target5'].std():.5f}  kurtosis {panel['target5'].kurt():.1f}")
print(f"  -> the 5-day target is {panel['target5'].std()/panel['target'].std():.2f}x the daily sd "
      f"(sqrt(5) = {np.sqrt(5):.2f} if returns were independent)")

TEST_START = pd.Timestamp("2024-09-03")
# Embargo must exceed the label horizon: a 5-day forward target means the last
# training label reaches 5 days into whatever follows. 10 = horizon + buffer.
EMBARGO = HORIZON + 5
embargo_end = TEST_START - pd.tseries.offsets.BDay(EMBARGO)
tr = panel[panel["date"] < embargo_end].copy()
te = panel[panel["date"] >= TEST_START].copy()
print(f"\ntrain {len(tr):,} rows / {tr['date'].nunique()} days   "
      f"test {len(te):,} rows / {te['date'].nunique()} days   embargo {EMBARGO} bdays")

X_tr, d_tr = tr[FEATURES], tr["date"].values
X_te, d_te = te[FEATURES], te["date"].values
y1_tr, y1_te = tr["target"].values, te["target"].values      # daily target
y5_tr, y5_te = tr["target5"].values, te["target5"].values    # weekly target

sd5 = y5_tr.std()
y5_tr_w = np.clip(y5_tr, -5 * sd5, 5 * sd5)
sd1 = y1_tr.std()
y1_tr_w = np.clip(y1_tr, -5 * sd1, 5 * sd1)


def nonoverlap_ic(y, p, d, step=HORIZON):
    """Rank IC on every 5th day only -- no overlapping labels, so the t-stat is honest."""
    udates = np.array(sorted(pd.unique(d)))
    keep = set(udates[::step])
    m = np.isin(d, list(keep))
    return rank_ic(y[m], p[m], d[m])


# --------------------------------------------------------------------------
# Experiment B: weekly target
# --------------------------------------------------------------------------
banner("EXPERIMENT B -- weekly target, full model roster")

CV = DateGroupedTimeSeriesSplit(n_splits=5, embargo=EMBARGO)
roster = build_roster(len(FEATURES))
rows, oof_store, params_store = [], {}, {}

for name, spec in roster.items():
    t0, best = time.time(), None
    for params in spec["grid"]:
        ics = fold_scores(spec, params, X_tr, y5_tr_w, y5_tr, d_tr, CV)
        if best is None or ics.mean() > best["ic"]:
            best = {"params": params, "ic": float(ics.mean()),
                    "se": float(ics.std(ddof=1) / np.sqrt(len(ics))), "ics": ics}
    oof = cv_oof(spec, best["params"], X_tr, y5_tr_w, y5_tr, d_tr, CV)
    m = np.isfinite(oof)
    ic_all, t_all, nd, _ = rank_ic(y5_tr[m], oof[m], d_tr[m])
    ic_no, t_no, nd_no, _ = nonoverlap_ic(y5_tr[m], oof[m], d_tr[m])
    oof_store[name], params_store[name] = oof, best["params"]
    rows.append({"model": name, "family": spec["family"],
                 "mean_fold_ic": best["ic"], "fold_ic_se": best["se"],
                 "folds_positive": int((best["ics"] > 0).sum()),
                 "oof_ic": ic_all, "oof_t_overlap": t_all,
                 "oof_t_nonoverlap": t_no, "n_days_nonoverlap": nd_no})
    print(f"  {name:14s} meanFoldIC {best['ic']:+.4f} (se {best['se']:.4f})  "
          f"pos {int((best['ics']>0).sum())}/5   oofIC {ic_all:+.4f} "
          f"(t {t_all:+.2f} overlapping / {t_no:+.2f} non-overlapping)  [{time.time()-t0:.0f}s]")

wk = pd.DataFrame(rows).sort_values("mean_fold_ic", ascending=False).reset_index(drop=True)
thr = wk.iloc[0]["mean_fold_ic"] - wk.iloc[0]["fold_ic_se"]
COMPLEX = {"penalised linear": 1, "distance": 2, "bagged trees": 3,
           "kernel": 4, "boosted trees": 5, "neural": 6}
wk["within_1se"] = wk["mean_fold_ic"] >= thr
WINNER5 = (wk[wk["within_1se"]].assign(c=lambda x: x["family"].map(COMPLEX))
           .sort_values(["c", "mean_fold_ic"], ascending=[True, False]).iloc[0]["model"])
print(f"\nSELECTED (weekly): {WINNER5}   within 1 s.e.: "
      f"{', '.join(wk[wk['within_1se']]['model'])}")

spec5 = roster[WINNER5]
final5 = build(spec5, params_store[WINNER5]); final5.fit(X_tr, y5_tr_w)
p5_te = final5.predict(X_te)
ic5_all, t5_all, _, _ = rank_ic(y5_te, p5_te, d_te)
ic5_no, t5_no, nd5_no, _ = nonoverlap_ic(y5_te, p5_te, d_te)
print(f"\n{WINNER5} on TEST (weekly target):")
print(f"  rank IC {ic5_all:+.4f}  t {t5_all:+.2f} (overlapping) / {t5_no:+.2f} "
      f"(non-overlapping, {nd5_no} days)")
print(f"  MAE {mae(y5_te, p5_te):.6f}   PF {signal_profit_factor(y5_te, p5_te):.4f}")

# every model on the weekly test set -- diagnostic
diag5 = []
for name, sp in roster.items():
    est = build(sp, params_store[name]); est.fit(X_tr, y5_tr_w)
    p = est.predict(X_te)
    ic, t_, _, _ = rank_ic(y5_te, p, d_te)
    ic_n, t_n, _, _ = nonoverlap_ic(y5_te, p, d_te)
    diag5.append({"model": name, "val_ic": wk.set_index("model").loc[name, "mean_fold_ic"],
                  "test_ic": ic, "test_t_nonoverlap": t_n})
diag5 = pd.DataFrame(diag5).sort_values("val_ic", ascending=False)
print("\nEvery model on the weekly TEST set:")
print(diag5.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
print(f"\nmean test IC {diag5['test_ic'].mean():+.4f} | "
      f"positive: {int((diag5['test_ic']>0).sum())} of {len(diag5)}")


# --------------------------------------------------------------------------
# Backtests
# --------------------------------------------------------------------------
banner("BACKTESTS")

N_POS = 8


def perf(r, freq=252):
    r = pd.Series(r).dropna()
    mu, sd = r.mean(), r.std(ddof=1)
    cum = r.cumsum(); dd = (cum - cum.cummax()).min()
    g_, l_ = r[r > 0].sum(), -r[r < 0].sum()
    yrs = len(r) / freq
    return {"AnnRet": np.exp(mu * freq) - 1, "AnnVol": sd * np.sqrt(freq),
            "Sharpe": mu / sd * np.sqrt(freq) if sd > 0 else np.nan,
            "PF": g_ / l_ if l_ > 0 else np.inf,
            "CAGR": np.exp(cum.iloc[-1] / yrs) - 1, "MaxDD": np.expm1(dd),
            "HitRate": (r > 0).mean()}


def book(g, pred_col):
    """+1/8 on the top quintile, -1/8 on the bottom. Returns the weight vector."""
    g = g.sort_values(pred_col, kind="mergesort")
    k = len(g) // 5
    w = pd.Series(0.0, index=g.index)
    w.iloc[:k] = -1.0 / N_POS
    w.iloc[len(g) - k:] = 1.0 / N_POS
    return g, w


def backtest(df, pred_col, ret_col, trade_every=1, freq=252):
    """Rebalance every `trade_every` trading days; hold until the next rebalance.

    Costs are charged only on days the book actually turns over, which is the
    whole point of trading less often.
    """
    dates = np.array(sorted(df["date"].unique()))
    rebal = set(dates[::trade_every])
    rows, prev_w, prev_idx = [], None, None
    for d_, g in df.groupby("date"):
        if d_ not in rebal:
            continue
        g, w = book(g, pred_col)
        gross = float((w * g[ret_col]).sum())
        spread = g["stock_spread"].fillna(g["stock_spread"].median()).fillna(0.0).to_numpy()
        if prev_w is None:
            dw = w.abs().to_numpy()
        else:
            prev_al = pd.Series(prev_w.values, index=prev_idx).reindex(g.index).fillna(0.0)
            dw = (w - prev_al).abs().to_numpy()
        cost = float(np.sum(dw * spread / 2.0))
        rows.append({"date": d_, "gross": gross, "net": gross - cost,
                     "turnover": dw.sum() / 2.0, "cost": cost})
        prev_w, prev_idx = w, g.index
    out = pd.DataFrame(rows).set_index("date")
    # annualisation factor: how many rebalances per year
    out.attrs["freq"] = freq / trade_every
    return out


# --- reference: the daily result already reported -------------------------
daily_model = json.load(open("results_summary.json"))
p1_te = pd.read_csv("results_strategy_daily_returns.csv", parse_dates=["date"])
print("Reference (from the main notebook): daily target, daily rebalance")
print(f"  test net Sharpe {daily_model['test_net_spread']['Sharpe']:+.3f}  "
      f"PF {daily_model['test_net_spread']['PF']:.3f}  "
      f"CAGR {daily_model['test_net_spread']['CAGR']:+.1%}  "
      f"turnover ~50%/day")

# --- Experiment A: daily target/model, traded weekly ----------------------
# Refit the daily winner. Rather than parse the stringified params out of JSON,
# re-run its grid against the DAILY target -- cheap, and guaranteed consistent.
WINNER1 = daily_model["winner"]
spec1 = roster[WINNER1]
best1 = None
for params in spec1["grid"]:
    ics = fold_scores(spec1, params, X_tr, y1_tr_w, y1_tr, d_tr,
                      DateGroupedTimeSeriesSplit(5, 5))
    if best1 is None or ics.mean() > best1["ic"]:
        best1 = {"params": params, "ic": float(ics.mean())}
print(f"  refit daily winner {WINNER1}: meanFoldIC {best1['ic']:+.4f}  {best1['params']}")
p1 = build(spec1, best1["params"]); p1.fit(X_tr, y1_tr_w)
te_A = te[["date", "ticker", "stock_spread"]].copy()
te_A["pred"] = p1.predict(X_te)
te_A["target"] = y1_te
te_A["target5"] = y5_te

res = {}
bt = backtest(te_A, "pred", "target", trade_every=1)
res["A0 daily model, DAILY rebalance"] = (perf(bt["net"], 252), bt)
bt = backtest(te_A, "pred", "target5", trade_every=HORIZON)
res["A1 daily model, WEEKLY rebalance"] = (perf(bt["net"], 252 / HORIZON), bt)

# --- Experiment B: weekly target, traded weekly ---------------------------
te_B = te[["date", "ticker", "stock_spread"]].copy()
te_B["pred"] = p5_te
te_B["target5"] = y5_te
bt = backtest(te_B, "pred", "target5", trade_every=HORIZON)
res["B  weekly model, WEEKLY rebalance"] = (perf(bt["net"], 252 / HORIZON), bt)
bt_g = backtest(te_B, "pred", "target5", trade_every=HORIZON)
res["B  weekly model, WEEKLY (gross)"] = (perf(bt_g["gross"], 252 / HORIZON), bt_g)

tabl = pd.DataFrame({k: v[0] for k, v in res.items()}).T
print("\nSTRATEGY COMPARISON (test window, net of quoted spread unless noted)\n")
print(tabl.round(4).to_string())
print("\nturnover and cost per rebalance:")
for k, (_, b) in res.items():
    print(f"  {k:38s} rebalances {len(b):>3}  mean turnover {b['turnover'].mean():5.1%}  "
          f"mean cost {b['cost'].mean():.6f}  mean gross {b['gross'].mean():+.6f}")

out = {"daily_reference": daily_model["test_net_spread"],
       "weekly_winner": WINNER5,
       "weekly_test_ic": float(ic5_all),
       "weekly_test_t_nonoverlap": float(t5_no),
       "weekly_models_positive": int((diag5["test_ic"] > 0).sum()),
       "strategies": {k: {kk: float(vv) for kk, vv in v[0].items()} for k, v in res.items()}}
json.dump(out, open("results_weekly_experiment.json", "w"), indent=1)
wk.to_csv("results_weekly_model_comparison.csv", index=False)
print("\nwritten: results_weekly_experiment.json, results_weekly_model_comparison.csv")
