# Cluster launchers

Written for Stanford's Sherlock (Slurm). Partitions (`normal`, `owners`,
`bigmem`, `gpu`, `menon`) and the `ruse` module are Sherlock-specific; adjust
for another cluster.

`source config/paths.sh` first — every script reads `EEGCHALLENGE_*` from it,
and batch jobs re-source it via `EEGCHALLENGE_ENV` because they start from a
clean shell.

**Submit from the repository root.** `#SBATCH --output=` cannot expand
environment variables, so the array scripts write to `jobs/` relative to the
submission directory. Either `cd` to the repo root first, or override:

```bash
sbatch --output=$EEGCHALLENGE_JOBS/%x_%A_%a.out slurm/supervised_linear_grid_search_stft_part1.sh
```

## Two launcher styles

**Direct `#SBATCH`** — `supervised_linear_*`, one array job over a parameter
grid. The array size is hardcoded in the script and must match the grid, which
is why they carry a `# Check these!!!` banner. The Python script prints the
correct size when run without Slurm.

**Generator** — `supervised_chal*.sh`, `test_chal*.sh`, `ensemble_test_chal1.sh`.
These read hyperparameters from `config/model_config.json` with `jq`, write a
temporary `.slurm` file per model, submit it, and delete it. They need `jq`.

```bash
bash slurm/supervised_chal1.sh EEGNeX
bash slurm/supervised_chal1.sh "EEGNeX,BIOT,Deep4Net"
bash slurm/supervised_chal1.sh all
```

## Scripts

| Script | Runs |
|---|---|
| `supervised_linear_preprocess.sh` | `pipelines/preprocess_challenge1.py` on `bigmem`; requests 256 GB and resubmits at 1500 GB if that fails |
| `supervised_linear_grid_search_stft_part{1,2,3}.sh` | STFT grid searches |
| `supervised_linear_grid_search_ensemble.sh` | Ensemble search over top-*k* estimators |
| `supervised_chal{1,2}*.sh` | Deep-learning training, one job per model |
| `test_chal1_allreleases.sh` | Test-set evaluation across all releases |
| `pretraining_chal1_mae.sh`, `pretraining_chal2*.sh` | MAE and multi-task pretraining |
| `ensemble_test_chal1.sh` | Ensembling strategies over trained checkpoints |
| `parse_err.sh` | Filters known-harmless warnings out of `.err` files |

Most experiment scripts also submit themselves — see `experiments/README.md`.
These launchers exist for the cases where the array parameters were tuned by
hand.
