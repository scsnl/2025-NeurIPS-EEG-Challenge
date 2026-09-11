# Experiments

One file per hyperparameter search that was actually run. They are deliberately
not merged: several are near-duplicates differing only in the grid, and that
difference *is* the record of what was tried.

Most scripts submit themselves — running one from a login node sbatches a Slurm
array sized to its parameter grid. See `src/eegchallenge/slurm.py`.

## Anatomy of a search

Every grid-search script has the same four parts:

```python
TO = config.grid_search_dir(challenge=1)   # per-job output directory
PIPE = Pipeline([...])                     # scale -> frequency transform -> reshape -> model
PARAM_GRID = [{...}]                       # what to search
grid_search(PIPE, PARAM_GRID, TO)          # this member's slice of the grid
```

The `frequency` step is the interesting one: it selects a representation from
`eegchallenge.features` (`passthrough` = raw 2 s waveform, or `power_phase_stft`,
`power_phase_circular`, `enveloppe`, `bandpower_canonical`,
`functional_connectivity`, and so on). Most of the work was deciding which
representation and which gradient-boosting regularisation to use.

## Challenge 1

| Script | Searches |
|---|---|
| `grid_search_1101_part{1,2,3}.py` | HistGradientBoosting on raw windows; the main sweep over learning rate, L2, leaf count, feature fraction, depth |
| `grid_search_stft_1101_part{1,2,3}.py` | Same, on STFT power+phase features |
| `grid_search_mlp_1101.py` | TruncatedSVD then a small MLP |
| `grid_search_biasrt.py` | `BiasRTRegressor` post-hoc bias correction on top of a fitted ensemble |
| `grid_search_ensembling.py` | `SimpleStackingRegressor` over the top-10 estimators |
| `refit_features.py` | Refits two fixed pipelines (`power_phase_circular`, `enveloppe`) without searching |
| `pickle_top_k.py` | Not a search: ranks all estimators across jobs and writes the top *k* |
| `boosting.py` | `SimpleBoostingRegressor` over saved estimators |
| `ensembling_test.py` | Evaluates ensembling strategies: mean, weighted mean, ridge, HGB stacking |

## Challenge 2

| Script | Searches |
|---|---|
| `grid_search_1102.py` | HistGradientBoosting on raw windows |
| `lightgbm_1102.py` | LightGBM, trained directly rather than through sklearn's grid search |
| `lightgbm_score_1102.py` | Scores a saved LightGBM booster |
| `data_to_csv.py` | Exports arrays to CSV for LightGBM's file-based loader |
| `find_mean.py` | Standalone algebra to recover the p-factor mean and std from three leaderboard probes |

## Results

`results/grid_search/<job_id>.csv` is the `cv_results_` table for one Slurm array
job. Best score per job, on the model-selection split:

| Job | Points | Best R² | Best NRMSE | Model | Representation |
|---|---|---|---|---|---|
| 8865913 | 912 | 0.0972 | 0.9502 | HistGradientBoosting | raw |
| 8898187 | 191 | 0.0960 | 0.9508 | HistGradientBoosting | raw |
| 8897851 | 674 | 0.0928 | 0.9525 | HistGradientBoosting | raw |
| 8867010 | 474 | 0.0915 | 0.9532 | HistGradientBoosting | STFT power+phase |
| 8904260 | 279 | 0.0897 | 0.9541 | HistGradientBoosting | STFT power+phase |
| 8899909 | 405 | 0.0892 | 0.9544 | HistGradientBoosting | raw |
| 8904275 | 150 | 0.0891 | 0.9544 | HistGradientBoosting | STFT power+phase |
| 8924180 | 100 | −0.0028 | 1.0014 | MLP (after SVD) | raw |
| 8783173 | 214 | 0.1013 | — | HistGradientBoosting, Ridge | STFT power+phase |
| 8783179 | 216 | 0.0941 | — | HistGradientBoosting, Ridge | power+phase circular |
| 8783182 | 142 | 0.0962 | — | HistGradientBoosting, Ridge | raw, power+phase |
| 8791474 | 720 | 0.0962 | — | HistGradientBoosting | cepstrum, envelope, raw |
| 8441740 | 516 | — | — | HistGradientBoosting, ExtraTrees | canonical band power |
| 8520196 | 505 | — | — | HistGradientBoosting, ExtraTrees | functional connectivity |
| 8520950 | 150 | — | — | HistGradientBoosting, ExtraTrees | power+phase temporal |

Reading this: nothing beat R² ≈ 0.10 on reaction time, and raw windows were
roughly as good as any engineered representation. The MLP was at chance. Gains
came from ensembling many similar gradient-boosted models rather than from a
better representation — which is why `pickle_top_k.py` and the ensembling
scripts exist. Jobs 8865913 (raw) and 8867010 (STFT) are the two that fed the
top-10 ensemble used for the submissions.

Blank columns are earlier runs whose scoring columns differ; `nrmse_test` was
added to the output partway through.

Caveat on the last four rows: those searches were run by earlier variants of
these scripts that were superseded before release and are not included here.
Their results are kept because they document which representations were ruled
out, but the exact code that produced them is not in this repository.
