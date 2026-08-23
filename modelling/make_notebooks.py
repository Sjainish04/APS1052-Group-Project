#!/usr/bin/env python3
"""
Generators for the two modelling notebooks.

    cd <repo root>
    .venv/bin/python modelling/make_notebooks.py
"""

import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "notebooks")

BOOT = r'''
import os, sys, json, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Notebooks live in modelling/notebooks/, so the repo root is two levels up.
ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
if not os.path.isdir(os.path.join(ROOT, "data")):
    ROOT = os.path.abspath(os.getcwd())          # already at the root
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "modelling", "results")
FIGS = os.path.join(RESULTS, "figures")
os.makedirs(FIGS, exist_ok=True)

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

from modelling.src.metrics import (mae, profit_factor, signal_profit_factor,
                                   rank_ic, daily_ic_series, report,
                                   directional_accuracy_by_quantile)
from modelling.src.splitters import DateGroupedTimeSeriesSplit, CombinatorialPurgedCV
from modelling.src.models import build_roster, wrap, RANDOM_STATE
from modelling.src.baselines import (ZeroPredictor, RandomPredictor,
                                     BestSingleFeature, random_ic_null)
from modelling.src.evaluate import build, cv_oof, fold_scores, run_roster, select_one_se

train = pd.read_csv(f"{DATA}/model_train.csv", parse_dates=["date"]).reset_index(drop=True)
test = pd.read_csv(f"{DATA}/model_test.csv", parse_dates=["date"]).reset_index(drop=True)
FEATURES = json.load(open(f"{DATA}/feature_meta.json"))["selected"]

X_tr, y_tr, d_tr = train[FEATURES], train["target"].values, train["date"].values
X_te, y_te, d_te = test[FEATURES], test["target"].values, test["date"].values

# Training target winsorised at +/-5 sd; every score uses the RAW target.
WINSOR_SD = 5.0
_sd = y_tr.std()
y_tr_w = np.clip(y_tr, -WINSOR_SD * _sd, WINSOR_SD * _sd)
'''


# ==========================================================================
# Notebook 1 -- model selection
# ==========================================================================

def notebook_one():
    cells = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
    code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

    md(r"""
# Model Selection
## Nine models, three baselines, one honest comparison

Runs on `data/model_train.csv` and `data/model_test.csv` from Part 1. No
network, no credentials.

### What this notebook is careful about

**Effective sample size.** 13,300 training rows, but only **665 distinct
trading days**. The 20 names on a day share one market shock and were demeaned
against each other, so rows within a day are close to the opposite of
independent. Roughly **17 effective observations per feature, not 350**. Every
grid here is deliberately small, every t-statistic is computed across days, and
no impressive number is taken at face value.

**Selection on rank IC, not MAE.** The strategy monetises the *ordering* of
names within a day. A model predicting ~0 everywhere scores the best MAE in the
study and is worth nothing. The `ZeroPredictor` baseline below exists to make
that visible rather than argued.

**The one-standard-error rule.** Nine models scoring within a band narrower
than their own standard errors are not nine distinguishable models. Taking the
maximum of nine correlated noisy estimates is upward-biased even if all are
equally good — and that bias gets paid back the moment the test set is opened.
So the rule is: best mean fold IC, then among everything within one standard
error of it, take the **simplest**.
""")

    code(BOOT + r'''
print(f"train {X_tr.shape[0]:,} rows x {X_tr.shape[1]} features | "
      f"{train['date'].nunique():,} days | {train['ticker'].nunique()} names")
print(f"test  {X_te.shape[0]:,} rows | {test['date'].nunique():,} days")
print(f"\ntarget sd {y_tr.std():.5f}  kurtosis {pd.Series(y_tr).kurt():.1f}")
print(f"winsorised at +/-{WINSOR_SD} sd -> "
      f"{int((np.abs(y_tr) > WINSOR_SD*_sd).sum())} rows clipped "
      f"({100*(np.abs(y_tr) > WINSOR_SD*_sd).mean():.2f}%), "
      f"kurtosis -> {pd.Series(y_tr_w).kurt():.1f}")
print(f"features carrying NaN: {int(X_tr.isna().any().sum())} of {len(FEATURES)}")
''')

    md(r"""
## 1. What does chance look like?

Before any model, fix the zero point. Predictions are shuffled **within each
day**, which preserves the exact panel shape — same days, same cross-section
size, same target distribution — and destroys only the ordering. That makes it
the right null for a cross-sectional ranking score.
""")

    code(r'''
draws, null_mean, null_sd, (null_lo, null_hi) = random_ic_null(
    y_tr, d_tr, n_draws=500, seed=RANDOM_STATE)

print(f"Empirical null over 500 draws ({train['date'].nunique()} days x 20 names):")
print(f"  mean rank IC      {null_mean:+.5f}")
print(f"  standard deviation {null_sd:.5f}")
print(f"  95% interval      [{null_lo:+.4f}, {null_hi:+.4f}]")
print(f"\nAnything inside that interval is indistinguishable from chance.")

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(draws, bins=40, color="#7A8C8B", edgecolor="white")
ax.axvline(0, color="black", lw=1)
ax.axvspan(null_lo, null_hi, color="#0D5B63", alpha=.12, label="95% of chance outcomes")
ax.set_xlabel("rank IC under random predictions"); ax.set_ylabel("draws")
ax.set_title("The zero point: what a rank IC of pure chance looks like on this panel")
ax.legend(); plt.tight_layout()
plt.savefig(f"{FIGS}/null_distribution.png", dpi=140, bbox_inches="tight"); plt.show()
''')

    md(r"""
## 2. Baselines

Three, each answering a different "compared to what".

`BestSingleFeature` is the one that matters. It picks the strongest single
feature **inside each training fold** (never on the full sample) and uses it
raw, sign-corrected, as the prediction. If nine tuned models cannot beat one
z-scored column, the machine learning added nothing — and the deck should
report that rather than omit the comparison.
""")

    code(r'''
CV = DateGroupedTimeSeriesSplit(n_splits=5, embargo=5)

print("Fold structure (expanding window, 5-day embargo):")
for k, (tr, va) in enumerate(CV.split(X_tr, y_tr, groups=d_tr)):
    td, vd = pd.Series(d_tr[tr]), pd.Series(d_tr[va])
    print(f"  fold {k}: train {len(tr):>6,} / {td.nunique():>3}d "
          f"({td.min():%Y-%m-%d}..{td.max():%Y-%m-%d})  ->  "
          f"val {len(va):>5,} / {vd.nunique():>3}d   gap {(vd.min()-td.max()).days}d")

BASELINES = {
    "ZeroPredictor":     {"factory": lambda: wrap(ZeroPredictor()),
                          "grid": [{}], "family": "baseline", "scaled": False},
    "RandomPredictor":   {"factory": lambda: wrap(RandomPredictor(seed=RANDOM_STATE)),
                          "grid": [{}], "family": "baseline", "scaled": False},
    "BestSingleFeature": {"factory": lambda: wrap(BestSingleFeature(feature_names=FEATURES)),
                          "grid": [{}], "family": "baseline", "scaled": False},
}

print("\nBaselines:")
base_results = run_roster(BASELINES, X_tr, y_tr_w, y_tr, d_tr, CV)

bsf = build(BASELINES["BestSingleFeature"], {})
bsf.fit(X_tr, y_tr_w)
inner = bsf.regressor_
print(f"\nBestSingleFeature picked: {inner.feature_}  "
      f"(train PF {inner.pf_:.4f}, sign {'+' if inner.sign_ > 0 else '-'})")
print(f"  MAE comparison -- ZeroPredictor {base_results['ZeroPredictor']['MAE']:.6f} "
      f"vs BestSingleFeature {base_results['BestSingleFeature']['MAE']:.6f}")
print("  The constant wins on MAE and cannot trade. That is the argument against MAE.")
''')

    md(r"""
## 3. The nine models

Four genuinely different inductive biases, so that whichever family wins is a
finding rather than an artefact of only having tried one kind of model.

| Family | Models | Scaling |
|---|---|---|
| Boosted trees | LightGBM, HistGBM | no — native NaN handling |
| Bagged trees | RandomForest, ExtraTrees | no |
| Penalised linear | Ridge, ElasticNet | **yes** |
| Kernel / distance / neural | SVR, kNN, Keras MLP | **yes** |

`StandardScaler` and `SelectKBest` are pipeline steps, refit inside every fold
— the brief is explicit that fitting them outside would leak validation-fold
statistics into training. `f_regression` scores the linear models,
`mutual_info_regression` the non-linear ones.
""")

    code(r'''
roster = build_roster(len(FEATURES))
total_fits = sum(len(s["grid"]) for s in roster.values()) * CV.get_n_splits()
print(f"{'model':14s} {'family':18s} {'configs':>8s} {'scaled':>7s}")
for n, s in roster.items():
    print(f"{n:14s} {s['family']:18s} {len(s['grid']):8d} {str(s['scaled']):>7s}")
print(f"\n{total_fits} fits for the grid search, plus {len(roster)} for out-of-fold prediction.")

t0 = time.time()
results = run_roster(roster, X_tr, y_tr_w, y_tr, d_tr, CV)
print(f"\ncompleted in {(time.time()-t0)/60:.1f} min")
''')

    md(r"""
## 4. Comparison and selection
""")

    code(r'''
all_results = {**base_results, **results}
winner, table = select_one_se(all_results)

show = table[["model", "family", "mean_fold_ic", "fold_ic_se", "folds_positive",
              "oof_RankIC", "oof_RankIC_t", "MAE", "ProfitFactor", "DirAcc",
              "Skill", "DegenDays", "within_1se", "selected"]]
print("Ranked by MEAN FOLD rank IC (not pooled):\n")
print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

top = table.iloc[0]
print(f"\nBest mean fold IC : {top['model']} at {top['mean_fold_ic']:+.4f} "
      f"(se {top['fold_ic_se']:.4f})")
print(f"Within 1 s.e.     : {', '.join(table[table['within_1se']]['model'])}")
print(f"SELECTED          : {winner}  (simplest family within one standard error)")

bsf_ic = table.loc[table['model'] == 'BestSingleFeature', 'mean_fold_ic'].iloc[0]
win_ic = table.loc[table['model'] == winner, 'mean_fold_ic'].iloc[0]
print(f"\nvs the single-feature baseline: {win_ic:+.4f} against {bsf_ic:+.4f} "
      f"({win_ic - bsf_ic:+.4f})")
print(f"vs chance (95% null upper bound {null_hi:+.4f}): "
      f"{'clears it' if win_ic > null_hi else 'INSIDE the null -- not distinguishable from chance'}")

table.to_csv(f"{RESULTS}/model_comparison.csv", index=False)
''')

    code(r'''
fig, axes = plt.subplots(1, 3, figsize=(17, 6))
t = table.iloc[::-1]
cols = ["#0D5B63" if s else ("#9BB0AF" if w else "#C9D4D3")
        for s, w in zip(t["selected"], t["within_1se"])]

axes[0].barh(t["model"], t["mean_fold_ic"], color=cols,
             xerr=t["fold_ic_se"], error_kw=dict(ecolor="#55605F", lw=1))
axes[0].axvline(0, color="black", lw=1)
axes[0].axvspan(null_lo, null_hi, color="#8B3F33", alpha=.10)
axes[0].set_title("Mean fold rank IC (+/- 1 s.e.)\nred band = indistinguishable from chance", fontsize=10)

axes[1].barh(t["model"], t["folds_positive"], color=cols)
axes[1].axvline(2.5, color="black", ls="--", lw=1)
axes[1].set_title("Folds with a positive IC (out of 5)\nconsistency, not magnitude", fontsize=10)

axes[2].barh(t["model"], t["MAE"], color=cols)
axes[2].set_title("MAE -- note the ranking is almost the REVERSE\nof rank IC", fontsize=10)
axes[2].set_xlim(t["MAE"].min() * .98, t["MAE"].max() * 1.02)

plt.tight_layout()
plt.savefig(f"{FIGS}/model_comparison.png", dpi=140, bbox_inches="tight"); plt.show()
''')

    md(r"""
## 5. Fine-tuning the selected model

A second, finer manual grid around the winner. Kept deliberately small: with
665 effective days, a wide search mostly locates the configuration best suited
to this particular fold split, and that advantage does not survive contact with
the test set.
""")

    code(r'''
spec = roster[winner] if winner in roster else BASELINES[winner]
base_params = all_results[winner]["params"]
print(f"Fine-tuning {winner}; stage-1 best: {base_params}")

FINE = {
    "LightGBM":     [dict(base_params, num_leaves=nl, learning_rate=lr, reg_lambda=rl)
                     for nl in (5, 7, 11) for lr in (0.01, 0.02) for rl in (10.0, 30.0)],
    "HistGBM":      [dict(base_params, max_leaf_nodes=ml, learning_rate=lr, l2_regularization=l2)
                     for ml in (5, 7, 11) for lr in (0.01, 0.02) for l2 in (5.0, 20.0)],
    "RandomForest": [dict(base_params, **{"model__max_depth": d, "model__min_samples_leaf": lf})
                     for d in (3, 4, 6) for lf in (100, 200, 400)],
    "ExtraTrees":   [dict(base_params, **{"model__max_depth": d, "model__min_samples_leaf": lf})
                     for d in (3, 4, 6) for lf in (100, 200, 400)],
    "Ridge":        [dict(base_params, **{"model__alpha": a})
                     for a in (10., 30., 100., 300., 1000., 3000.)],
    "ElasticNet":   [dict(base_params, **{"model__alpha": a, "model__l1_ratio": l1})
                     for a in (3e-4, 1e-3, 3e-3, 1e-2) for l1 in (0.1, 0.5, 0.9)],
    "SVR":          [dict(base_params, **{"model__C": c, "model__epsilon": e})
                     for c in (0.05, 0.1, 0.3) for e in (0.02, 0.05, 0.1)],
    "kNN":          [dict(base_params, **{"model__n_neighbors": nn})
                     for nn in (200, 400, 600, 800, 1200)],
    "KerasMLP":     [dict(base_params, **{"model__dropout": dr, "model__l2": l2})
                     for dr in (0.2, 0.3, 0.4) for l2 in (1e-3, 1e-2)],
}.get(winner, [base_params])

# The stage-1 winner MUST be in the fine grid. Without it, a finer search over a
# neighbourhood that happens to exclude the incumbent can only move sideways or
# down -- and it did: HistGBM's stage-1 best (max_leaf_nodes=15, lr=0.05) fell
# outside the refined ranges, so "tuning" cost 0.0073 of mean fold IC and the
# worse configuration was kept. Tuning must never be able to lose.
FINE = [base_params] + [f for f in FINE if f != base_params]

best_fine, t0 = None, time.time()
for params in FINE:
    ics = fold_scores(spec, params, X_tr, y_tr_w, y_tr, d_tr, CV)
    if best_fine is None or ics.mean() > best_fine["ic"]:
        best_fine = {"params": params, "ic": float(ics.mean()),
                     "se": float(ics.std(ddof=1) / np.sqrt(len(ics))), "ics": ics}
print(f"searched {len(FINE)} configurations (incumbent included) in {time.time()-t0:.0f}s")
stage1_ic = all_results[winner]["mean_fold_ic"]
print(f"  stage-1 incumbent scored {stage1_ic:+.4f}; tuning "
      f"{'improved on it' if best_fine['ic'] > stage1_ic else 'could not beat it (incumbent kept)'}")
print(f"  best mean fold IC {best_fine['ic']:+.4f} (se {best_fine['se']:.4f}), "
      f"positive in {int((best_fine['ics'] > 0).sum())}/{len(best_fine['ics'])} folds")
print(f"  params: {best_fine['params']}")
FINAL_PARAMS = best_fine["params"]
oof_final = cv_oof(spec, FINAL_PARAMS, X_tr, y_tr_w, y_tr, d_tr, CV)
''')

    md(r"""
## 6. The held-out test set

Refit on the whole training block with the tuned parameters, predict once.
Nothing below has influenced any choice above.
""")

    code(r'''
final = build(spec, FINAL_PARAMS)
final.fit(X_tr, y_tr_w)
pred_te = final.predict(X_te)

varies = pd.Series(pred_te).groupby(pd.Series(d_te)).nunique().gt(1)
print(f"Test days on which the prediction varies across the 20 names: "
      f"{varies.sum()} of {len(varies)}")
assert varies.mean() > 0.5, "the fitted model gives no cross-sectional ordering"

m = np.isfinite(oof_final)
res_val = report(y_tr[m], oof_final[m], d_tr[m], label=f"{winner} -- VALIDATION (out-of-fold)")
res_te = report(y_te, pred_te, d_te, label=f"{winner} -- TEST (held out)")
''')

    code(r'''
side = pd.DataFrame([
    {"metric": "Mean absolute error",    "validation": res_val["MAE"],          "test": res_te["MAE"]},
    {"metric": "Profit Factor",          "validation": res_val["ProfitFactor"], "test": res_te["ProfitFactor"]},
    {"metric": "Spearman rank IC",       "validation": res_val["RankIC"],       "test": res_te["RankIC"]},
    {"metric": "Rank IC t-stat",         "validation": res_val["RankIC_t"],     "test": res_te["RankIC_t"]},
    {"metric": "Directional accuracy %", "validation": res_val["DirAcc"],       "test": res_te["DirAcc"]},
    {"metric": "Naive baseline %",       "validation": res_val["NaiveDirAcc"],  "test": res_te["NaiveDirAcc"]},
    {"metric": "Skill (pct points)",     "validation": res_val["Skill"],        "test": res_te["Skill"]},
])
print("VALIDATION vs TEST\n")
print(side.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
side.to_csv(f"{RESULTS}/validation_vs_test.csv", index=False)

print(f"\nRank IC degradation: {res_val['RankIC'] - res_te['RankIC']:+.4f}")

# Every model on the test set -- a DIAGNOSTIC, run after selection was locked
# and fed back into nothing. It answers one question: was the outcome specific
# to the selected model, or did none of them carry?
diag = []
for name, spec_i in roster.items():
    est = build(spec_i, all_results[name]["params"]); est.fit(X_tr, y_tr_w)
    p = est.predict(X_te)
    ic, t, nd, _ = rank_ic(y_te, p, d_te)
    diag.append({"model": name, "val_mean_fold_ic": all_results[name]["mean_fold_ic"],
                 "test_RankIC": ic, "test_t": t,
                 "test_PF": signal_profit_factor(y_te, p)})
diag = pd.DataFrame(diag).sort_values("val_mean_fold_ic", ascending=False)
print("\n\nPOST-HOC DIAGNOSTIC -- every model on the test set (not used for selection):\n")
print(diag.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
print(f"\nmean test rank IC across all {len(diag)} models: {diag['test_RankIC'].mean():+.4f}")
print(f"models positive out of sample: {int((diag['test_RankIC'] > 0).sum())} of {len(diag)}")
diag.to_csv(f"{RESULTS}/test_diagnostic_all_models.csv", index=False)
''')

    code(r'''
out_te = test[["date", "ticker", "target", "fwd_logret", "close_adj", "stock_spread"]].copy()
out_te["pred"] = pred_te
out_te.to_csv(f"{RESULTS}/pred_test.csv", index=False)

out_tr = train.loc[m, ["date", "ticker", "target", "fwd_logret"]].copy()
out_tr["pred"] = oof_final[m]
out_tr.to_csv(f"{RESULTS}/pred_train_oof.csv", index=False)

# Pickle rather than JSON: several grids carry tuples (e.g. hidden=(64,32))
# and a JSON round-trip would hand notebook 2 a list, which Keras rejects.
import pickle
with open(f"{RESULTS}/params.pkl", "wb") as fh:
    pickle.dump({"winner": winner,
                 "final_params": FINAL_PARAMS,
                 "best_params_by_model": {n: r["params"] for n, r in all_results.items()}}, fh)

json.dump({"winner": winner, "family": all_results[winner]["family"],
           "params": {k: str(v) for k, v in FINAL_PARAMS.items()},
           "features": FEATURES, "n_models": len(roster), "n_baselines": len(BASELINES),
           "null_95_upper": null_hi,
           "validation": {k: float(res_val[k]) for k in
                          ["MAE", "ProfitFactor", "RankIC", "RankIC_t", "DirAcc", "Skill"]},
           "test": {k: float(res_te[k]) for k in
                    ["MAE", "ProfitFactor", "RankIC", "RankIC_t", "DirAcc", "Skill"]}},
          open(f"{RESULTS}/model_meta.json", "w"), indent=1)

print("=" * 70)
print("MODEL SELECTION COMPLETE")
print("=" * 70)
print(f"  tried      : {len(roster)} models + {len(BASELINES)} baselines")
print(f"  selected   : {winner} ({all_results[winner]['family']})")
print(f"  validation : rank IC {res_val['RankIC']:+.4f} (t {res_val['RankIC_t']:+.2f})")
print(f"  test       : rank IC {res_te['RankIC']:+.4f} (t {res_te['RankIC_t']:+.2f})")
print(f"  chance     : 95% of random predictions fall below {null_hi:+.4f}")
print(f"\n  written to modelling/results/")
''')

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3.11 (.venv)",
                                     "language": "python", "name": "python3"},
                      "language_info": {"name": "python", "version": "3.11.14",
                                        "mimetype": "text/x-python",
                                        "file_extension": ".py"}}
    return nb


# ==========================================================================
# Notebook 2 -- diagnostics
# ==========================================================================

def notebook_two():
    cells = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
    code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

    md(r"""
# Diagnostics
## How much of the result is signal, and how much is one lucky path?

Notebook 1 produced a single walk-forward number. This one asks how much to
trust it, along four lines:

1. **Combinatorial purged CV** — 15 splits reassembled into **5 complete
   backtest paths**, giving a *distribution* where walk-forward gives a point.
2. **Stability through time** — rolling rank IC across the whole sample, which
   separates "the edge decayed" from "we overfit".
3. **SHAP** — feature importance on the held-out set, as the brief requires.
4. **Per-fold detail** — where the pooled number came from.
""")

    code(BOOT + r'''
import pickle
meta = json.load(open(f"{RESULTS}/model_meta.json"))
with open(f"{RESULTS}/params.pkl", "rb") as fh:
    saved = pickle.load(fh)
WINNER = saved["winner"]
FINAL_PARAMS = saved["final_params"]
BEST_BY_MODEL = saved["best_params_by_model"]
roster = build_roster(len(FEATURES))
spec = roster[WINNER]
print(f"Selected model from notebook 1: {WINNER}")
print(f"  params: {FINAL_PARAMS}")
''')

    md(r"""
## 1. Combinatorial purged cross-validation

Walk-forward gives one path: the single history that happened. CPCV splits the
timeline into 6 contiguous blocks and holds out every pair — 15 splits — with
purging and an embargo on **both sides** of each held-out block, since an
interior block leaks in both directions in a way a walk-forward split never has
to handle.

Because each block is held out in 5 of the 15 splits, the predictions reassemble
into **5 complete backtest paths**. The spread across those paths is the honest
uncertainty on the single walk-forward number.
""")

    code(r'''
CPCV = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo=5)
print(f"CPCV(6,2): {CPCV.get_n_splits()} splits, {CPCV.n_paths} reassemblable paths")

TOP = list(pd.read_csv(f"{RESULTS}/model_comparison.csv")
           .query("family != 'baseline'").head(4)["model"])
if WINNER not in TOP:
    TOP.append(WINNER)
print(f"Running CPCV on: {', '.join(TOP)}\n")

path_maps = CPCV.path_assignments(d_tr)
cpcv_rows, path_store = [], {}

for name in TOP:
    sp = roster[name]
    params = FINAL_PARAMS if name == WINNER else BEST_BY_MODEL[name]
    t0 = time.time()
    # path_id -> accumulated predictions over the whole timeline
    paths = {p: np.full(len(y_tr), np.nan) for p in range(CPCV.n_paths)}
    udates = np.array(sorted(pd.unique(d_tr)))
    blocks = CPCV._blocks(udates)          # same blocks the splitter used
    pos = pd.Series(np.arange(len(udates)), index=udates)
    row_pos = pd.Series(d_tr).map(pos).to_numpy()

    for split_i, (tr, te) in enumerate(CPCV.split(X_tr, y_tr, groups=d_tr)):
        est = build(sp, params); est.fit(X_tr.iloc[tr], y_tr_w[tr])
        p = est.predict(X_tr.iloc[te])
        for blk, path_id in path_maps[split_i].items():
            sel = np.isin(row_pos[te], blocks[blk])
            paths[path_id][te[sel]] = p[sel]

    ics = []
    for pid, pv in paths.items():
        msk = np.isfinite(pv)
        ic, t, nd, _ = rank_ic(y_tr[msk], pv[msk], d_tr[msk])
        ics.append(ic)
    path_store[name] = paths
    cpcv_rows.append({"model": name, "mean": np.mean(ics), "sd": np.std(ics, ddof=1),
                      "min": np.min(ics), "max": np.max(ics),
                      "paths_positive": int(np.sum(np.asarray(ics) > 0)),
                      "secs": time.time() - t0,
                      **{f"path_{i}": v for i, v in enumerate(ics)}})
    print(f"  {name:14s} paths {' '.join(f'{v:+.4f}' for v in ics)}   "
          f"mean {np.mean(ics):+.4f} sd {np.std(ics, ddof=1):.4f}  [{time.time()-t0:.0f}s]")

cpcv = pd.DataFrame(cpcv_rows)
cpcv.to_csv(f"{RESULTS}/cpcv_paths.csv", index=False)
print("\nThe SD across paths is the uncertainty the single walk-forward number hides.")
''')

    code(r'''
fig, ax = plt.subplots(figsize=(11, 5))
pcols = [c for c in cpcv.columns if c.startswith("path_")]
for i, row in cpcv.iterrows():
    ax.scatter(row[pcols].values, [i] * len(pcols), s=70, color="#0D5B63",
               alpha=.75, zorder=3, label="individual path" if i == 0 else None)
    ax.plot([row["min"], row["max"]], [i, i], color="#9BB0AF", lw=2, zorder=2)
    ax.scatter([row["mean"]], [i], marker="|", s=400, color="#8B3F33",
               zorder=4, label="mean" if i == 0 else None)
ax.set_yticks(range(len(cpcv))); ax.set_yticklabels(cpcv["model"])
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("rank IC per reassembled backtest path")
ax.set_title("CPCV: five backtest paths per model — the spread is the real uncertainty")
ax.legend(loc="lower right"); plt.tight_layout()
plt.savefig(f"{FIGS}/cpcv_paths.png", dpi=140, bbox_inches="tight"); plt.show()
''')

    md(r"""
## 2. Did the edge decay, or was it never there?

Every model went negative on the test set. Two explanations fit that, and they
have very different implications for the write-up:

- **Overfitting** — the validation edge was never real, and the test set simply
  revealed it.
- **Regime shift** — the relationships held in the training window and stopped
  holding afterwards.

A rolling rank IC across the whole sample distinguishes them. Under
overfitting, out-of-fold IC hovers near zero throughout. Under a regime shift,
it is positive through the training window and turns over.
""")

    code(r'''
oof = pd.read_csv(f"{RESULTS}/pred_train_oof.csv", parse_dates=["date"])
pte = pd.read_csv(f"{RESULTS}/pred_test.csv", parse_dates=["date"])

ic_tr = daily_ic_series(oof["target"], oof["pred"], oof["date"])
ic_te = daily_ic_series(pte["target"], pte["pred"], pte["date"])
ic_all = pd.concat([ic_tr, ic_te]).sort_index()

W = 63   # one quarter
roll = ic_all.rolling(W).mean()
split_date = pte["date"].min()

fig, ax = plt.subplots(figsize=(13, 5))
ax.axhline(0, color="black", lw=1)
ax.plot(ic_all.index, ic_all.values, color="#C9D4D3", lw=.7, label="daily rank IC")
ax.plot(roll.index, roll.values, color="#0D5B63", lw=2, label=f"{W}-day rolling mean")
ax.axvline(split_date, color="#8B3F33", ls="--", lw=1.5)
ax.annotate("test set opens", xy=(split_date, ax.get_ylim()[1] * .82),
            xytext=(8, 0), textcoords="offset points", color="#8B3F33", fontsize=10)
ax.set_ylabel("rank IC"); ax.legend(loc="upper left")
ax.set_title("Rolling rank IC — out-of-fold through the training window, held-out after")
plt.tight_layout()
plt.savefig(f"{FIGS}/rolling_ic.png", dpi=140, bbox_inches="tight"); plt.show()

print(f"mean daily IC, out-of-fold (train window): {ic_tr.mean():+.4f}  "
      f"over {len(ic_tr)} days")
print(f"mean daily IC, held-out test             : {ic_te.mean():+.4f}  "
      f"over {len(ic_te)} days")
print(f"\nby calendar year:")
print(ic_all.groupby(ic_all.index.year).agg(["mean", "std", "count"])
      .to_string(float_format=lambda x: f"{x:.4f}"))
''')

    md(r"""
## 3. SHAP — feature importance on the held-out set

The brief asks for SHAP values on the testing set. A model-agnostic
`PermutationExplainer` is used rather than `TreeExplainer`, because the
estimator is a `TransformedTargetRegressor` wrapping a pipeline — explaining
`predict` end-to-end attributes importance in the units the strategy actually
consumes, and keeps this cell correct whichever family won.
""")

    code(r'''
import shap

final = build(spec, FINAL_PARAMS)
final.fit(X_tr, y_tr_w)

rng = np.random.default_rng(RANDOM_STATE)
bg_idx = rng.choice(len(X_tr), size=min(100, len(X_tr)), replace=False)
ex_idx = rng.choice(len(X_te), size=min(300, len(X_te)), replace=False)
background = X_tr.iloc[bg_idx].fillna(X_tr.median())
explain_on = X_te.iloc[ex_idx].fillna(X_tr.median())

t0 = time.time()
explainer = shap.PermutationExplainer(final.predict, background)
sv = explainer(explain_on, max_evals=2 * len(FEATURES) + 1, silent=True)
print(f"SHAP over {len(explain_on)} test rows in {time.time()-t0:.0f}s")

imp = (pd.DataFrame({"feature": FEATURES,
                     "mean_abs_shap": np.abs(sv.values).mean(axis=0)})
       .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
imp.to_csv(f"{RESULTS}/shap_importance.csv", index=False)
print("\nTop 15 by mean |SHAP| on the test set:\n")
print(imp.head(15).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

BAR = set(json.load(open(f"{DATA}/feature_meta.json"))["bar"])
imp["bucket"] = ["bar" if f.replace("z_", "") in BAR else "exogenous/calendar"
                 for f in imp["feature"]]
share = imp.groupby("bucket")["mean_abs_shap"].sum()
print(f"\nShare of total importance:  "
      + "  ".join(f"{k} {v/share.sum():.0%}" for k, v in share.items()))
''')

    code(r'''
fig, ax = plt.subplots(figsize=(10, 8))
top = imp.head(20).iloc[::-1]
cols = ["#8B3F33" if b == "bar" else "#0D5B63" for b in top["bucket"]]
ax.barh(top["feature"], top["mean_abs_shap"], color=cols)
ax.set_xlabel("mean |SHAP| on the held-out test set")
ax.set_title("Feature importance — teal = exogenous/calendar, red = target's own bar")
plt.tight_layout()
plt.savefig(f"{FIGS}/shap_importance.png", dpi=140, bbox_inches="tight"); plt.show()

plt.figure()
shap.summary_plot(sv.values, explain_on, feature_names=FEATURES, show=False, max_display=20)
plt.tight_layout()
plt.savefig(f"{FIGS}/shap_beeswarm.png", dpi=140, bbox_inches="tight"); plt.show()
''')

    md(r"""
## 4. Where the pooled number came from
""")

    code(r'''
CV = DateGroupedTimeSeriesSplit(n_splits=5, embargo=5)
ics = fold_scores(spec, FINAL_PARAMS, X_tr, y_tr_w, y_tr, d_tr, CV)
folds = []
for k, (tr, va) in enumerate(CV.split(X_tr, y_tr, groups=d_tr)):
    vd = pd.Series(d_tr[va])
    folds.append({"fold": k, "val_from": vd.min().date(), "val_to": vd.max().date(),
                  "val_days": vd.nunique(), "rank_IC": ics[k]})
fd = pd.DataFrame(folds)
print(f"{WINNER} — rank IC computed separately in each fold:\n")
print(fd.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
print(f"\nmean {ics.mean():+.4f}   sd {ics.std(ddof=1):.4f}   "
      f"positive in {int((ics>0).sum())}/{len(ics)}")
fd.to_csv(f"{RESULTS}/fold_stability.csv", index=False)

print("\n" + "=" * 70)
print("DIAGNOSTICS COMPLETE")
print("=" * 70)
print(f"  CPCV        : {CPCV.n_paths} paths per model -> modelling/results/cpcv_paths.csv")
print(f"  stability   : rolling IC -> figures/rolling_ic.png")
print(f"  SHAP        : {len(imp)} features -> shap_importance.csv, 2 figures")
print(f"  fold detail : fold_stability.csv")
''')

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3.11 (.venv)",
                                     "language": "python", "name": "python3"},
                      "language_info": {"name": "python", "version": "3.11.14",
                                        "mimetype": "text/x-python",
                                        "file_extension": ".py"}}
    return nb


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for fn, name in [(notebook_one, "01_model_selection.ipynb"),
                     (notebook_two, "02_diagnostics.ipynb")]:
        nb = fn()
        p = os.path.join(OUT, name)
        nbf.write(nb, p)
        n_code = sum(c["cell_type"] == "code" for c in nb["cells"])
        print(f"wrote {name}: {len(nb['cells'])} cells ({n_code} code)")
