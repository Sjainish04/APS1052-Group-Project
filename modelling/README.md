# Modelling

Model selection and diagnostics for the cross-sectional long-short strategy.
Reads the panel Part 1 produced (`data/model_train.csv`, `data/model_test.csv`)
and writes everything it learns to `results/`.

No network, no WRDS account, no credentials. Everything runs from the bundled
CSVs.

## Layout

```
modelling/
  src/
    metrics.py      the four metrics the brief requires, + self-test
    splitters.py    date-grouped TimeSeriesSplit and combinatorial purged CV, + self-test
    models.py       nine models, their pipelines and grids; the Keras MLP wrapper
    baselines.py    zero / random / best-single-feature, and the empirical null
    evaluate.py     the manual grid-search CV loop and the one-standard-error rule
  notebooks/
    01_model_selection.ipynb   baselines -> roster -> selection -> tuning -> test
    02_diagnostics.ipynb       CPCV paths, stability through time, SHAP, per-fold detail
  results/          CSVs, figures/, and the pickled parameters notebook 2 reads
  make_notebooks.py generator for both notebooks
```

`src/` exists so the two notebooks share one implementation of every metric and
splitter instead of duplicating them. The brief requires the final deliverable
to be a **single self-contained Jupyter notebook**, so these get inlined in one
consolidation pass at the end — developing against modules first is only to make
the intervening iterations cheap and testable.

## Running it

From the repository root:

```bash
.venv/bin/python -m modelling.src.metrics        # self-test
.venv/bin/python -m modelling.src.splitters      # self-test
.venv/bin/python modelling/make_notebooks.py     # regenerate both notebooks

cd modelling/notebooks
../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 01_model_selection.ipynb
../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 02_diagnostics.ipynb
```

Notebook 1 takes roughly 15 minutes, notebook 2 rather longer — CPCV fits every
model 15 times over. Notebook 2 depends on `results/params.pkl`, so run them in
order.

## The four decisions worth knowing about

**Selection is on rank IC, never MAE.** The strategy monetises the ordering of
names within a day. A model predicting ~0 for everything scores the best MAE in
the entire study and is worth nothing to trade. `ZeroPredictor` is in the
baseline set specifically so that appears as a row in a table rather than as a
claim in a paragraph.

**Scoring is per fold, then averaged — not pooled.** A model that is excellent
in one fold and useless in four is not a good model, and pooling conceals that
completely. The reported standard error is across folds.

**The one-standard-error rule breaks ties.** Nine models landing inside a band
narrower than their own standard errors are not nine distinguishable models.
Taking the maximum of nine correlated noisy estimates is upward-biased even when
every model is equally good, and the bias is repaid the moment the test set is
opened. So: best mean fold IC, then the **simplest** model within one standard
error of it.

**A day with no ordering scores zero, and is not skipped.** If a model returns
the same number for all 20 names, it supplied no ranking information that day.
Dropping such days from the average silently rewards degeneracy — a model that
collapses to a constant on 80% of days would be judged only on the 20% where it
did something. This is not hypothetical: an over-regularised ElasticNet won an
earlier comparison exactly this way, and `metrics.py` now has a unit test for it.

## Leakage guards, and where they live

| Guard | Where |
|---|---|
| Folds break between whole dates, never inside one | `splitters.py` |
| 5-day embargo after each training block | `splitters.py` |
| CPCV purges on **both** sides of interior test blocks | `splitters.py` |
| `StandardScaler` / `SelectKBest` refit inside every fold | `models.py` |
| Target standardiser refit inside every fold | `models.py` (`wrap`) |
| Early stopping validates on a chronological tail, not a shuffle | `models.py` (`KerasMLP`) |
| Best single feature chosen inside the training fold only | `baselines.py` |
| Training target winsorised; scoring target always raw | notebook 1 |

## Outputs

| File | Contents |
|---|---|
| `model_comparison.csv` | every model and baseline, ranked, with the 1-s.e. band |
| `validation_vs_test.csv` | the seven headline metrics side by side |
| `test_diagnostic_all_models.csv` | post-hoc: every model on the test set |
| `cpcv_paths.csv` | rank IC per reassembled backtest path |
| `fold_stability.csv` | per-fold rank IC for the selected model |
| `shap_importance.csv` | mean \|SHAP\| per feature on the held-out set |
| `pred_test.csv`, `pred_train_oof.csv` | predictions, for the Part 3 backtest |
| `params.pkl` | selected model and tuned parameters, read by notebook 2 |
| `figures/` | null distribution, model comparison, CPCV paths, rolling IC, SHAP |
