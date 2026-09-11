# Retrospective

Notes written by [Nicholas Branigan](https://github.com/nkbranigan) after the
competition closed, lightly edited. 

Where a claim can be checked against `results/grid_search/`, the measured
numbers are added in brackets. All of them are normalised RMSE on the
validation split, so they are optimistic. See the README section
[Reproducibility notes](README.md#reproducibility-notes).

## What went well

We explained about 10% of the variance in reaction time. The best team explained
about 20%.

We did it with simple models: ensembles of decision trees, and ridge regression.

We tried a lot of things. Transformers, CNNs, pretraining, ensembling, gradient
boosting, linear models, multi-layer perceptrons, and feature engineering in
both the frequency and time domains.

## What we could have done better (maybe)

Overall I think we tried to do too much too fast. Learn to walk before you run.

Starting with the braindecode models was the right first step. But when none of
them predicted better than chance, in retrospect we should have exhaustively
debugged that, focusing on a single model, probably the simplest one. Anthony
Strock had a good idea for sanity checking the deep learning pipeline:
reproduce the successful ridge regression result in PyTorch. If the same model
that works in scikit-learn does not work through our training loop, the bug is
in the loop rather than in the architecture.

The most likely reason the braindecode models failed is inadequate
regularisation. We did not explore that simple explanation, and instead went
after more complex solutions like pretraining.

## What I have learned

**Linear models do not always match deep networks under high noise.** This is in
tension with [Schulz et al.](https://doi.org/10.1016/j.celrep.2023.113597),
which found linear models performing on par with nonlinear ones across most
neuroimaging prediction targets. The leaderboard's top models, presumably deep
networks, explained roughly twice the variance we did, 20% against 10%. That
said, our own deep networks never beat chance, so this conclusion rests on other
teams' results and not on ours.

**Ensembling helps, and it helps more when the members differ.** An ensemble of
one model trained on temporal data and one trained on short-time Fourier
transform data beats an ensemble of two temporal models or two STFT models.
Adding more models to the ensemble also helps.

**Ridge versus gradient boosting.** The best models of each kind land close
together: gradient boosting at 0.948 and ridge at 0.962. For a scientific
setting, ridge may still be the better choice, since interpretability outweighs
slightly worse prediction.

The difference is in how much tuning each one needs. Tuning the ridge
regularisation strength is crucial; tuning the gradient boosting
hyperparameters is not. Many ridge fits are worse than predicting the mean,
while almost no gradient boosting fits are:

| Model | Best NRMSE | Fits worse than chance (NRMSE > 1) | Fits |
| --- | --- | --- | --- |
| Histogram gradient boosting | 0.948 | 0% to 10% | 4,649 |
| Ridge | 0.962 | 84% | 518 |
| PLS regression | 0.967 | 57% | 109 |
| Extra trees | 0.970 | 16% | 81 |
| MLP (after SVD) | 1.001 | 100% | 100 |

If you have the compute to search either one properly, the choice matters
little. If you do not, gradient boosting is far more forgiving.
