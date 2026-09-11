<!-- prettier-ignore -->
<div align="center">

# SCSNL Code for NeurIPS 2025 EEG Foundation Challenge

[![Stanford SCSNL](https://img.shields.io/badge/Stanford-SCSNL-8C1515?style=flat-square)](https://scsnl.stanford.edu)
[![EEG Foundation Challenge](https://img.shields.io/badge/EEG%20Foundation%20Challenge-NeurIPS%202025-8A2BE2?style=flat-square)](https://eeg2025.github.io/)
<br>
[![Python](https://img.shields.io/badge/Python->=3.12-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch Lightning](https://img.shields.io/badge/Lightning-792ee5?style=flat-square&logo=lightning&logoColor=white)](https://lightning.ai/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-f7931e?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-green?style=flat-square)](LICENSE)

> Cross-task and cross-subject EEG decoding from the Stanford Cognitive and Systems Neuroscience Laboratory (SCSNL)

[The challenge](#the-challenge) • [Approaches](#approaches) • [Getting started](#getting-started) • [Reproducing](#reproducing) • [Project structure](#project-structure)

</div>

This repository holds the SCSNL code for [EEG Foundation
Challenge](https://eeg2025.github.io/), a NeurIPS 2025 competition on learning
transferable representations of EEG. We built and compared a wide range of
models, from simple linear regression through gradient boosted trees to deep
networks, along with engineered features and self-supervised pretraining, to
predict reaction time and psychopathology from EEG in the HBN-EEG dataset.

> [!NOTE]
> No data or trained weights are included here. The HBN-EEG releases are
> distributed by the Child Mind Institute Healthy Brain Network under their own
> data use agreement, so request access through them.

## The challenge

The challenge uses [HBN-EEG](https://neuromechanist.github.io/data/hbn/),
128-channel EEG recorded from over 3,000 children and young adults by the Child
Mind Institute Healthy Brain Network across six paradigms: resting state,
surround suppression, movie watching, contrast change detection, sequence
learning, and symbol search.

The data is published in numbered releases. We trained our model on R1 to R4 and R6
to R11 and used R5 for validation. The competition
used R12, a hidden held-out release for test and scoring. Models are evaluated by normalised
RMSE, measured on participants in R12.

We preprocessed the raw EEG recordings with
[braindecode](https://braindecode.org/), which reads the BIDS releases through
MNE. The EEG is resampled to 100 Hz and cut into 2 second windows, so one input
is a `129 x 200` array of channels by time points.

| | Challenge 1 | Challenge 2 |
| --- | --- | --- |
| **Input** | A 2-s window per trial, starting 0.5 s after stimulus onset, on the contrast change detection task, keeping only trials with both a stimulus and a response | windows cropped from resting state and task recordings |
| **Output** | reaction time in seconds, one value per trial | the externalizing score, one value per participant |
| **Predicted per** | trial | participant |

Both tasks are challenging, as they require cross-subject transfer as well as cross-task transfer learning. Challenge 1 requires generalization from the passive tasks, resting state, surround suppression and movie watching, to the active contrast change detection task that the model was never trained on. Challenge 2 requires prediction of a person's psychopathology trait, the externalizing score, from resting-state and task data.


## Approaches

We started with the simplest model that could work and added complexity to test which model and approach has a better performance.

| | Approach | Where |
| --- | --- | --- |
| 1 | Linear and ridge regression on raw windows | `experiments/challenge{1,2}/` |
| 2 | Engineered spectral and temporal features | `src/eegchallenge/features.py` |
| 3 | Gradient boosted trees on engineered features | `experiments/challenge1/grid_search_*.py` |
| 4 | Model ensembling and bias correction | `src/eegchallenge/sklearn_models.py` |
| 5 | Deep learning networks trained end to end, 22 architectures | `pipelines/train_*.py` |
| 6 | Self supervised pretraining, then fine tuning | `pipelines/pretrain_*.py` |

**What we found.** Nothing did better than an R² of about 0.10 on reaction time.
Raw windows worked about as well as any engineered feature set, and an MLP on
SVD-reduced features was no better than guessing. The best single model was
histogram-based gradient boosting on raw windows, at a normalised RMSE of 0.950
(R² 0.097) on the validation split for Challenge 1. And ensembling works:
averaging the predictions of the top ten gradient boosted models explains more
variance than using an engineered feature set or a deeper network. The numbers
for each search are in [`experiments/README.md`](experiments/README.md).

> [!NOTE]
> That 0.950 is measured on the validation split, not on held out data. See
> [Reproducibility notes](#reproducibility-notes) for why it flatters the model.

### Classical machine learning

Classical machine learning pipelines are written as scikit-learn pipelines with
four steps, and ran as a Slurm job array:

1. **scale**: normalise the window and clip outliers, or pass it through unchanged
2. **features**: turn the window into one of the representations below, or leave it as the raw waveform
3. **reshape**: flatten the result into the single feature vector a regressor expects
4. **regressor**: the model being fitted, such as ridge or gradient boosting

Every grid search varies the choices at these four steps.

We began with ridge and ordinary linear regression on the raw 2 second window,
to see how far the simplest thing would get. The next question was whether a
better feature would help, so `src/eegchallenge/features.py`
implements eleven features: the raw waveform, canonical
band power, power spectral density, STFT power and phase, circular power and
phase, the signal envelope, real and complex cepstrum, functional connectivity,
polynomial expansion, and temporal derivatives. Scaling and clipping based on
the median absolute deviation keeps outliers from dominating.

We then swapped the linear model for gradient boosted trees, and most of the
compute went into tuning how strongly they are regularised.

We implemented a grid search of different models, feature engineering approaches, hyper-parameter tunings and submitted them as parallel jobs using SLURM. Below are the searches we actually ran:

| Search | Variant |
| --- | --- |
| `challenge1/grid_search_1101_part{1,2,3}.py` | the main sweep, gradient boosting on raw windows, over learning rate, L2, leaf count, feature fraction, depth |
| `challenge1/grid_search_stft_1101_part{1,2,3}.py` | the same sweep on STFT power and phase features |
| `challenge1/grid_search_mlp_1101.py` | TruncatedSVD followed by a small MLP |
| `challenge1/refit_features.py` | refits two fixed pipelines, circular power/phase and envelope, without searching |
| `challenge2/grid_search_1102.py` | gradient boosting on raw windows |
| `challenge2/lightgbm_1102.py` | LightGBM, trained directly rather than through scikit-learn |

`src/eegchallenge/sklearn_models.py` adds custom scikit-learn regressors that
average the members' predictions (plain, weighted, or summed), stack them by
fitting a second model on their outputs, or boost them by fitting each member on
what the previous ones got wrong. `pickle_top_k.py` ranks every model across all
jobs and keeps the best *k*, and `grid_search_ensembling.py` and
`ensembling_test.py` search for the best way to combine them. Separately,
`BiasRTRegressor` (`grid_search_biasrt.py`) corrects a consistent offset in the
predicted reaction times.

### Deep learning

We trained 22 deep neural networks on the preprocessed EEG windows with PyTorch Lightning
(`pipelines/train_*.py`), with the settings for each architecture in
`config/model_config.json`:

- **braindecode backbones**: ATCNet, AttnSleep, BIOT, CTNet, Deep4Net,
  EEGConformer, EEGNet, EEGNeX, EEGSimpleConv, FBCNet, Labram, MSVTNet,
  SignalJEPA, SPARCNet, SyncNet, TSception, plus `BIOT_PRE`, a BIOT variant with
  its own preprocessing front end
- **custom CNN architectures** (`src/eegchallenge/models/cnn.py`): AsdNet, ConvNet,
  eConvNet, stConvNet
- **EEGPT**, an EEG foundation model
  ([NeurIPS 2024](https://github.com/BINE022/EEGPT)) pretrained with a dual
  self-supervised objective: masked reconstruction of the signal, plus alignment
  of spatio-temporal representations, which avoids learning from the raw
  low-SNR signal alone

`src/eegchallenge/models/ensembling.py` picks the best checkpoints across runs
and combines them, doing for the deep models what `pickle_top_k.py` does for the
trees.

> [!IMPORTANT]
> EEGPT is not a braindecode model. It is a separate project that we connected
> to the same interface, so you can select it like any other model, but its code
> is not copied into this repository. `src/eegchallenge/models/eegpt.py` imports
> it only when you ask for it, from the upstream
> [EEGPT](https://github.com/BINE022/EEGPT) repository on your `PYTHONPATH`.
> Every other model works without it. Set `EEGPT_CHECKPOINT` and `PYTHONPATH` as
> shown in `config/paths.example.sh`.

### Self supervised pretraining

We also experimented with self-supervised pretraining of our models on the
HBN-EEG data without using the labels, then fine tuned them on the challenge
target.

- **Challenge 1**: masked autoencoder pretraining, in `pipelines/pretrain_mae_challenge1.py`
- **Challenge 2**: multi task pretraining across HBN tasks, in `pipelines/pretrain_challenge2.py`

## Getting started

Python 3.12 or newer is required, because the code uses nested same quote
f-strings that are a syntax error on earlier versions.

```bash
conda env create -f environment.yml
conda activate 2025_eeg_challenge
pip install -e .

cp config/paths.example.sh config/paths.sh
$EDITOR config/paths.sh        # set your data, working, and results directories
source config/paths.sh
```

No file paths are written into the code. Every module, pipeline, experiment, and
launcher reads its locations from `EEGCHALLENGE_DATA`, `EEGCHALLENGE_WORK`, and
`EEGCHALLENGE_RESULTS`, which `config/paths.sh` sets. That file is gitignored,
so your own paths never end up in the repository. See
`src/eegchallenge/config.py`.

`requirements-lock.txt` is a full `pip freeze` of the environment used for the
runs described here.

> [!NOTE]
> The notebooks are kept as they were run, so the directories they read no
> longer exist and they will not run end to end until you regenerate the
> intermediate files. Each notebook says so in its first cell.

## Reproducing

We developed this on a Slurm cluster, and the launchers in `slurm/` are written
for Slurm. The resource requests below reflect that, so adjust them for your own
scheduler.

**1. Preprocess.** Turn the raw BIDS recordings into arrays.

```bash
sbatch slurm/supervised_linear_preprocess.sh     # requests 256G, retries at 1500G
```

This writes `X.npy`, `y.npy`, `test_fold.npy`, and the per split arrays into
`$EEGCHALLENGE_WORK/challenge{1,2}/`.

> [!WARNING]
> This step takes about an hour and well over 40 GB of memory, because it builds
> every window at once instead of streaming them.

**2. Search.** Each experiment script submits itself as a Slurm array job, sized
to its own parameter grid.

```bash
python experiments/challenge1/grid_search_1101_part1.py
```

Each job in the array writes one pickle into
`$EEGCHALLENGE_WORK/challenge1/supervised_linear_grid_search/$JOBID/`.

> [!WARNING]
> These scripts submit jobs to the cluster rather than running the search on the
> spot, and the command above submits roughly 2,000 of them. Each script prints
> how many it is about to submit, so check that number first.

**3. Select and combine.** Take the best *k* models across jobs and combine
them.

```bash
python experiments/challenge1/pickle_top_k.py
python experiments/challenge1/ensembling_test.py --regressor mean <pickles...>
```

**4. Bundle.** [`submission/README.md`](submission/README.md) describes the
layout Codabench expects, which is strict and easy to get wrong.

To run the deep learning track instead:

```bash
bash slurm/supervised_chal1.sh "EEGNeX,BIOT"    # or: all
bash slurm/test_chal1_allreleases.sh EEGNeX
```

The settings for each model come from `config/model_config.json`.

## Reproducibility notes

> [!IMPORTANT]
> This repository splits the data differently from the way our competition
> submissions did. Any score measured on release R5 with the original code looks
> better than it should, and must not be reported as performance on held out
> data.

The preprocessed arrays include a `test_fold.npy` that marks each row as
training (`-1`) or model selection (`0`). This code trains on `X_train` and
selects on `X_val`. Our November 2025 submissions trained on `X_train + X_val`
and selected on `X_test`, which is release R5. This did not affect the
leaderboard, which scored against R12, a set we never touched.

The code as submitted is preserved under `submission/as_submitted/`, unchanged
except that the hardcoded cluster paths have their username redacted to
`<user>`. Please do not reformat or relocate those files: fitted pipelines are
pickled by module path, so renaming or moving a module breaks every existing
`.pkl`.

## Project structure

```
.
├── src/eegchallenge/           importable library
│   ├── config.py               filesystem locations, from environment variables
│   ├── data.py                 HBN-EEG loading, windowing, subject level splits
│   ├── features.py             the eleven spectral and temporal representations
│   ├── pipeline.py             grid search driver (PredefinedSplit + Slurm arrays)
│   ├── sklearn_models.py       ensembling, boosting, bias correction regressors
│   ├── slurm.py                self submitting job helpers
│   ├── parser.py               command line and config argument handling
│   ├── io.py                   run directory creation and logging
│   ├── plot.py                 loss curves from metrics.csv
│   ├── memory_report.py        parses peak memory out of Slurm logs
│   └── models/
│       ├── load.py             backbone registry, builds a model from config
│       ├── generic.py          the Lightning module every deep model shares
│       ├── ensembling.py       checkpoint selection and ensembling
│       ├── cnn.py              custom CNN architectures
│       ├── eegpt.py            adapter for the third party EEGPT backbone
│       ├── biot_pre.py         BIOT preprocessing front end
│       ├── augmentation.py     training time augmentations
│       ├── loss.py             loss functions, including normalised RMSE
│       └── stats.py            output statistics tracked during training
├── pipelines/                  runnable end to end stages
│   ├── preprocess*.py          raw BIDS to .npy arrays
│   ├── train_*.py              supervised training, per challenge
│   ├── evaluate_*.py           test set evaluation, per challenge
│   ├── pretrain_*.py           masked autoencoder and multi task pretraining
│   └── make_submission*.py     build a Codabench bundle
├── experiments/                every hyperparameter search, one file per search
│   ├── challenge1/
│   └── challenge2/
├── results/grid_search/        the tables those searches produced
├── submission/                 entry points, and the code as submitted
│   └── as_submitted/           the code as submitted, do not edit
├── slurm/                      cluster launchers
├── notebooks/                  exploratory analysis
└── config/                     model hyperparameters, path template
```

Three directories carry their own README:
[`experiments/`](experiments/README.md) indexes each search and the results file
it produced, [`submission/`](submission/README.md) documents the Codabench
bundle layout, and [`slurm/`](slurm/README.md) covers the launchers.

`experiments/` is deliberately not deduplicated. Each file is one search that
was actually run, and several are near duplicates differing only in the grid.
That difference is the record of what was tried, so collapsing them into a
single parameterised script would have erased it.

## Contributors

- Anthony Strock ([@a-strock](https://github.com/a-strock))
- Nicholas Branigan ([@nkbranigan](https://github.com/nkbranigan))
- Saksham Pruthi ([@Consilium5128](https://github.com/Consilium5128))
- Tong Shan ([@TongShan4869](https://github.com/TongShan4869))
- Linjing Jiang ([@linjjiang](https://github.com/linjjiang))

[Stanford Cognitive and Systems Neuroscience Laboratory](https://scsnl.stanford.edu)

## Citation

If you use this code, please cite the challenge paper
([arXiv:2506.19141](https://arxiv.org/abs/2506.19141)) and acknowledge the
Healthy Brain Network dataset.
