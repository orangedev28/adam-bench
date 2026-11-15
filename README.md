# adam-bench

Code and training logs for a benchmark of 17 Adam-family optimizers and SGD with momentum on convolutional networks, and for the follow-up study

> **Update Rule or Weight Decay? Disentangling the Gains of Adam Variants on Batch-Normalized Convolutional Networks**

Everything needed to check the numbers of the paper is here: the training code, the result file of every run, and the analysis scripts. The analysis needs no GPU and no PyTorch.

## Contents

| Path | What it is |
|---|---|
| `bench/` | Library: configuration (`config.py`), data on GPU (`data.py`), models (`models.py`), optimizers (`optim.py`), one training run (`train.py`), statistics (`stats.py`), follow-up experiments (`experiments.py`) |
| `run_cell.py` | Runner of the benchmark: LR sweep, LR selection, main runs, for one cell |
| `run_exp.py` | Runner of the follow-up experiments (X1, E1, E2, E2b, E3, E4, E1c, E5, E5b, E6, E6c) |
| `analyze.py` | Summaries of the benchmark (tables, rankings, plots) |
| `analysis/` | Analysis scripts of the paper (factorial tables, ANOVA, paired tests, figure) |
| `results/benchmark/` | Benchmark results: 893 runs, cells A–E |
| `results/followup/` | Follow-up results: 291 runs, experiments E1, E2, E2b, E3, E4, E5, E5b, E1c, E6, E6c |
| `colab.ipynb`, `colab_exp.ipynb`, `kaggle_exp.ipynb` | Launchers used on Google Colab and Kaggle (the two `*_exp` notebooks are the versions used for the last experiments, E1c–E6c; earlier versions are in the git history) |
| `ref/` | Reference code used only by `--check-ref` |

## Benchmark

Five cells, each a dataset × model pair run entirely on one GPU type (Tesla T4):

| Cell | Dataset | Model | Epochs (sweep / main) | Platform |
|---|---|---|---|---|
| A | Fashion-MNIST | 5-layer CNN with BN | 30 / 30 | Colab |
| B | CIFAR-10 | 5-layer CNN with BN | 50 / 50 | Colab |
| C | CIFAR-10 | ResNet-18 | 30 / 100 | Kaggle |
| D | CIFAR-100 | 5-layer CNN with BN | 50 / 50 | Kaggle |
| E | CIFAR-10 | 5-layer CNN, BN replaced by conv biases | 50 / 50 | Kaggle |

Optimizers: Adam, AdaMax, NAdam, AMSGrad, AdamW, RAdam, AdaBelief, AdamP, Adan, Lion, MARS, AdEMAMix, C-AdamW, Schedule-Free AdamW, SPAM, AdamS, VRAdam, and SGD with momentum as a reference. Adam, AdaMax, NAdam, AMSGrad, AdamW, RAdam and SGD come from `torch.optim`; the others are re-implemented in `bench/optim.py` from the papers and the authors' code (sources are listed at the top of that file).

Protocol, per cell and optimizer:

1. **LR sweep**: a four-point logarithmic grid, one seed, full schedule. If the best LR lies at the edge of the grid, one more point is added beyond it (at most twice per side).
2. **LR selection**: by validation accuracy at the last epoch. The validation set is a fixed split of 5,000 training images; the test set is never used for selection.
3. **Main runs**: the selected LR with seeds 0–4. The same seed gives every optimizer the same initialization, data order and augmentation.

Weight decay is fixed per form (coupled L2 `5e-4`, decoupled `5e-2`, Lion `0.5`) and not tuned in the benchmark. Training uses batch size 128, mixed precision and a per-step cosine schedule without warm-up (Schedule-Free AdamW uses a constant LR with its own warm-up). Runs are deterministic: in the same environment, re-running a run reproduces its training losses bit for bit.

## Follow-up experiments

Optimizer variants are named `base-suffix`: `-l2` (coupled L2), `-dw` (decoupled weight decay), `-xK` (default coefficient of that form multiplied by K), e.g. `sfadamw-l2`, `adabelief-dw`, `adam-x0.01`.

| Experiment | Content | Cells | Runs |
|---|---|---|---|
| E1 | Missing corners of the 2×2 design: `adabelief-dw`, `sfadamw-l2` (LR search + 5 seeds) | B, D, E | 61 |
| E2 | Weight-decay sweep: Adam ×0.1, ×10, ×100 and AdamW ×0.01, ×0.1, ×10 (LR search + 3 seeds) | B, E | 89 |
| E2b | Sweep extension: Adam ×0.01, ×0.001; one more LR point for AdamW ×0.1 in B | B, E | 33 |
| E3 | Re-runs of Adam, AdamW, AdaBelief, SF-AdamW with weight-norm recording (seeds 0–1) | B, E | 16 |
| E4 | More seeds: Adam ×0.01 (seeds 3–4); Adam and AdamW with norm recording (seeds 2–4) | B, E | 16 |
| E5, E5b | Seeds 3–4 for every remaining value of the weight-decay sweep, so that all values have 5 seeds | B, E | 28 |
| E1c | Missing corners of the 2×2 design on ResNet-18: `adabelief-dw`, `sfadamw-l2` (LR search with 30-epoch runs + 5 seeds of 100 epochs) | C | 20 |
| E6, E6c | Re-runs (seeds 0–1) with per-step recording of the measured update: Adam (×1, ×0.01), AdamW, SF-AdamW (L2 and decoupled) in B and E; Adam, AdamW, AdaBelief, SF-AdamW in C | B, C, E | 28 |

`bench/experiments.py` also defines X1 (LR search with shortened cosine schedules); its results are not part of this release.

## Reproducing the numbers of the paper

Run from the repository root; only Python 3 and matplotlib are needed.

```bash
python analysis/facts.py           # Table IV (sweep, selected LRs, imbalance I) -> analysis/out/facts.json
python analysis/stats_blocked.py   # Tables II and III (ANOVA with seeds as blocks, cells B, C, D, E), per-seed contrasts, Holm, paired tests, TOST -> analysis/out/facts_stats.json
python analysis/fig_wd.py          # Fig. 1 -> analysis/out/fig_wd.pdf
python analysis/measured.py        # Table V (imbalance from the proxy and from measured updates) -> analysis/out/facts_measured.json
python analysis/fig_norm.py analysis/out  # Fig. 2 (weight norms per epoch)
python analysis/fig_rho.py analysis/out   # Fig. 3 (measured relative update per layer)
python analysis/verify_rho.py      # summary of proxy vs measured updates over all runs that recorded them
python analyze.py --out results/benchmark   # benchmark summaries (already included under results/benchmark)
```

## Result files

Each run has a directory `results/.../<cell>/runs/<run id>/` with a `result.json`:

- `optimizer`, `lr`, `weight_decay`, `seed`, `phase` (`sweep` or `final`), `diverged`;
- `history`: per epoch `train_loss`, `train_acc`, `val_loss`, `val_acc`, `lr_end`, and in the follow-up runs `w_norms` (Frobenius norm of every conv weight); in the runs of E1c, E5, E5b, E6 and E6c also, per conv layer and averaged over 10 evenly spaced steps of the epoch, `rel_upd` (measured ‖W<sub>t+1</sub> − W<sub>t</sub>‖/‖W<sub>t</sub>‖), `ang_upd` (angle between W<sub>t</sub> and W<sub>t+1</sub>, radians) and `rho_step` (the proxy η<sub>t</sub>√C/‖W<sub>t</sub>‖), and `eta_eff_mean` (mean of η<sub>t</sub>/‖W‖² over the layers);
- `final`: train/validation/test loss and accuracy at the last epoch;
- `env`: GPU, torch, CUDA and cuDNN versions.

Per cell, `best_lr.json` holds the selected LRs, `sweep_ext.json` the grid extensions and `env_lock.json` the environment. The raw `log.txt` files are kept as produced; runs made before the code comments and messages were translated contain log messages in Vietnamese.

## Running the training code

```bash
pip install -r requirements.txt
python run_cell.py --selftest                 # fake data, no download; run first on every machine
python run_cell.py --cell B --phase all --out results --data data
python run_exp.py --plan                      # list the follow-up work
python run_exp.py --exp E1 --cells B --out results_exp --data data
```

Each cell is locked to a Tesla T4 and to the environment of its first run; `--any-gpu` and `--force-env` lift these checks for testing (results then differ slightly from the published ones). Without CUDA the code falls back to MPS or CPU, without mixed precision. `python run_cell.py --check-ref` compares the re-implemented optimizers with the original packages (see `ref/README.md`).

## Notes

- The comments and messages in the code were translated to English after the runs were made; the program logic is unchanged.
- Cell C was first attempted on a laptop GPU that produced computation errors; those results were discarded and the cell was run on Kaggle.
