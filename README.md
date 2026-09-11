# NeurIPS 2025 EEG Foundation Challenge — SCSNL

This repository contains the Stanford Cognitive and Systems Neuroscience
Laboratory entry to the [EEG Foundation Challenge](https://eeg2025.github.io/)
(NeurIPS 2025 Competition Track). It covers both tracks:

- **Challenge 1** — predicting reaction time on the contrast change detection
  task (cross-task transfer).
- **Challenge 2** — predicting the p-factor from resting-state and task EEG
  (cross-subject generalisation).

Both are scored by normalised RMSE on held-out subjects. We pursued two
independent approaches, and both are published here in full:

| Track | Approach | Where |
|---|---|---|
| Deep learning | braindecode backbones (EEGNeX, ATCNet, Deep4Net, BIOT, LaBraM, EEGPT, …) trained with PyTorch Lightning | `pipelines/train_*.py` |
| Feature-based | engineered spectral features fed to gradient-boosted trees, grid-searched at scale | `experiments/` |

Our final submissions used the feature-based track.

---

## Layout

```
src/eegchallenge/     library code
  config.py           filesystem locations, from environment variables
  data.py             HBN-EEG loading, windowing, subject-level splits
  features.py         spectral/temporal transforms for the sklearn track
  pipeline.py         grid-search driver (PredefinedSplit + Slurm job arrays)
  sklearn_models.py   custom estimators (stacking, boosting, bias correction)
  slurm.py            self-submitting job helpers
  models/             braindecode wrappers, Lightning modules, custom CNNs, MAE

pipelines/            runnable end-to-end stages
  preprocess.py       raw BIDS -> .npy arrays  (memory-hungry; see below)
  train_*.py          supervised training, per challenge
  evaluate_*.py       test-set evaluation, per challenge
  pretrain_*.py       MAE and multi-task pretraining
  make_submission.py  build a Codabench bundle

experiments/          every hyperparameter search we ran, one file per search
results/grid_search/  the search results those files produced (see manifest)
submission/           submission entry points, plus the code as actually submitted
slurm/                cluster launchers
notebooks/            exploratory analysis
config/               model hyperparameters, path template
```

`experiments/` is deliberately not deduplicated: each file is one search that was
actually run, and `experiments/README.md` maps each file to the results it
produced. Collapsing them into a single parameterised script would have erased
the record of what was tried.

## Setup

```bash
conda env create -f environment.yml
conda activate 2025_eeg_challenge
pip install -e .

cp config/paths.example.sh config/paths.sh
$EDITOR config/paths.sh        # set your data, working, and results directories
source config/paths.sh
```

Python 3.12 or newer is required: the code uses nested same-quote f-strings,
which are a syntax error on earlier versions. `requirements-lock.txt` is a full
`pip freeze` of the environment used for the runs described here.

Paths are never hardcoded. Every library module, pipeline, experiment, and
launcher resolves locations through `EEGCHALLENGE_DATA`, `EEGCHALLENGE_WORK`,
and `EEGCHALLENGE_RESULTS`; see `src/eegchallenge/config.py`.

The exception is `notebooks/`, which is preserved as originally run and still
contains absolute paths from the cluster it was run on. Each notebook notes this
in its first cell.

## Reproducing

This pipeline was developed on a Slurm cluster, and the launchers in `slurm/`
target Slurm directly. The resource requests below reflect that environment;
adapt them to your own scheduler as needed.

**1. Preprocess** — convert raw BIDS recordings into dense arrays. This stage is
expensive: roughly an hour, and well over 40 GB of RAM, because it materialises
every window at once.

```bash
sbatch slurm/supervised_linear_preprocess.sh     # requests 256G, retries at 1500G
```

It writes `X.npy`, `y.npy`, `test_fold.npy`, and the per-split arrays into
`$EEGCHALLENGE_WORK/challenge{1,2}/`.

**2. Search** — each experiment script submits itself as a Slurm array sized to
its own parameter grid:

```bash
python experiments/challenge1/grid_search_1101_part1.py
```

Note that this submits jobs rather than running the search locally — for the
script above, roughly 2,000 of them. Each script prints its array size before
submitting, so check that number first. Every array member writes one pickle
into `$EEGCHALLENGE_WORK/challenge1/supervised_linear_grid_search/$JOBID/`.

**3. Select and ensemble** — combine the best *k* estimators across jobs:

```bash
python experiments/challenge1/pickle_top_k.py
python experiments/challenge1/ensembling_test.py --regressor mean <pickles...>
```

**4. Bundle** — `submission/README.md` documents the expected Codabench layout,
which is strict and easy to get wrong.

To run the deep-learning track instead:

```bash
bash slurm/supervised_chal1.sh "EEGNeX,BIOT"    # or: all
bash slurm/test_chal1_allreleases.sh EEGNeX
```

Per-model hyperparameters come from `config/model_config.json`.

## Reproducibility

**The split definition in this repository differs from the one used for the
competition submissions.** The preprocessed arrays carry a `test_fold.npy` that
marks each row as training (`-1`) or model selection (`0`). This code uses:

```
train on   = X_train
select on  = X_val
```

The November 2025 competition submissions were produced with:

```
train on   = X_train + X_val
select on  = X_test      (release R5)
```

That is, hyperparameters were originally selected against R5. This had no effect
on the leaderboard, which scored against a hidden R12 set that was never
touched. It does mean that any number computed on R5 with the original code is
optimistic and should not be reported as held-out performance.

The code exactly as submitted is preserved byte-for-byte under
`submission/as_submitted/`. Please do not reformat or relocate those files:
fitted pipelines are pickled by module path, so renaming or moving a module
breaks every existing `.pkl`.

## Data

The data is not included here. The HBN-EEG releases are distributed by the Child
Mind Institute Healthy Brain Network under their own data use agreement; request
access through them rather than through this repository. No raw data,
participant-level output, or `participants.tsv` is committed to this repository.

Challenge details and preprocessing conventions are described in the
[challenge paper](https://arxiv.org/abs/2506.19141).

## Model weights

Trained weights are not distributed; reproduce them with the steps above.

The `EEGPT` backbone additionally requires the upstream
[EEGPT](https://github.com/BINE022/EEGPT) repository on `PYTHONPATH` along with
its pretrained checkpoint, neither of which is vendored here. Set
`EEGPT_CHECKPOINT` and `PYTHONPATH` as shown in `config/paths.example.sh`. Every
other model runs without it: the import is deferred, so a missing EEGPT does not
break the package.

## Contributors

- Anthony Strock ([@a-strock](https://github.com/a-strock))
- Nicholas Branigan ([@nkbranigan](https://github.com/nkbranigan))
- Saksham Pruthi ([@Consilium5128](https://github.com/Consilium5128))
- Tong Shan ([@TongShan4869](https://github.com/TongShan4869))
- Linjing Jiang ([@linjjiang](https://github.com/linjjiang))

Stanford Cognitive and Systems Neuroscience Laboratory.

## Citation

If you use this code, please cite the challenge paper
([arXiv:2506.19141](https://arxiv.org/abs/2506.19141)) and acknowledge the
Healthy Brain Network dataset.

## License

BSD 3-Clause — see [LICENSE](LICENSE).
