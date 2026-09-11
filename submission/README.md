# Submissions

## Bundle layout

Codabench requires a **flat** zip — no top-level folder:

```
my_submission.zip
|-- submission.py
|-- challenge_1.pkl
|-- challenge_2.pkl
`-- python_packages/
    |-- challenges/challenge2/supervised_linear.py
    `-- utils/submit.py
```

`python_packages/` is not optional. joblib pickles store *references* to the
module paths where a pipeline's functions were defined, so unpickling on the
evaluation machine re-imports them by name. A bundle without those modules
loads fine locally — where the real package happens to be importable — and
fails on Codabench.

This is also why `as_submitted/` must stay byte-for-byte as shipped. Those files
define the module paths (`challenges.challenge2.supervised_linear`,
`utils.sklearn_model`) that every existing `.pkl` refers to. Rename, move, or
reformat them and the pickles stop loading.

Build a bundle:

```bash
python pipelines/make_submission.py       # constant-predictor sanity submission
```

Then zip it (single level, from inside the directory) and upload.

## Files

| File | Purpose |
|---|---|
| `submission_sklearn.py` | Entry point for the sklearn track. Wraps a fitted pipeline in `SklearnToTorch`, which exposes `.forward()` / `.eval()` so the harness can treat it like a torch model. |
| `submission_sklearn_ensembling.py` | Same, for ensemble pipelines. |
| `submission_torch.py` | Entry point for the deep-learning track: loads `weights_challenge_{1,2}.pt` into braindecode models. |
| `as_submitted/` | The code exactly as shipped in the November 2025 submissions. Verbatim — see below. |

## `as_submitted/`

Preserved so the competition results stay reproducible. It differs from the
current code in one way that matters:

```
                  train on           select on
as submitted      X_train + X_val    X_test  (release R5)
current code      X_train            X_val
```

Hyperparameters were originally selected against R5. Harmless for the
leaderboard, which scored on a hidden R12 set — but it means R5 numbers from the
original code are optimistic. See the README "Reproducibility" section.

The remaining differences are incidental: `return_train_score`, a `challenge=`
parameter, a `breakpoint()` left in the shipped `submission.py`, and a `psutil`
import.

## Practical notes

The submission runs under a memory and time budget on the evaluation machine.
`submission_sklearn.py` prints CPU count, total RAM, and elapsed time per
forward pass for exactly this reason — large ensembles were the binding
constraint, and the top-10 bundle is roughly 365 MB of pickled estimators.
`resolve_path()` searches `/app/input/res/`, `/app/input/`, then the working
directory, because the harness has moved where it stages files.
