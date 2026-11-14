# Adam-family benchmark: summary

## Pooled ranking

Cells: A, B, C, D, E: 25 blocks (cell × seed), 18 optimizers. Friedman χ² = 200.4, p = <0.001. Nemenyi CD (α = 0.05) = 5.27: two optimizers whose average ranks differ by less than CD are *not* significantly different.

| Rank | Optimizer | Group | Avg. rank |
|---:|---|---|---:|
| 1 | SF-AdamW | modern | 2.18 |
| 2 | SGD-M | reference | 5.26 |
| 3 | SPAM | modern | 6.00 |
| 4 | C-AdamW | modern | 6.06 |
| 5 | AdamP | classic | 7.10 |
| 6 | AdaBelief | classic | 7.14 |
| 7 | VRAdam | modern | 7.70 |
| 8 | AdamW | classic | 8.44 |
| 9 | Lion | modern | 8.44 |
| 10 | MARS | modern | 8.46 |
| 11 | Adan | modern | 10.64 |
| 12 | AdamS | modern | 12.32 |
| 13 | NAdam | classic | 12.40 |
| 14 | AdEMAMix | modern | 12.66 |
| 15 | RAdam | classic | 12.74 |
| 16 | Adam | classic | 13.28 |
| 17 | AdaMax | classic | 15.06 |
| 18 | AMSGrad | classic | 15.12 |

Average rank of the classic group: 11.41; modern group: 8.27.

## Ablation BatchNorm

Cell B (with BN) and cell E (without BN): same dataset and architecture; BN is replaced by conv biases. Question: does the advantage (or disadvantage) of each optimizer depend on BN? See the change in ranks.

| Optimizer | Group | With BN | Without BN | Δ (BN − no BN) | Rank with BN | Rank without BN |
|---|---|---:|---:|---:|---:|---:|
| SF-AdamW | modern | 90.90 | 89.72 | 1.18 | 1 | 1 |
| SPAM | modern | 90.46 | 88.52 | 1.94 | 2 | 3 |
| C-AdamW | modern | 90.33 | 87.94 | 2.38 | 3 | 6 |
| SGD-M | reference | 90.29 | 87.54 | 2.75 | 4 | 9 |
| AdamW | classic | 90.22 | 87.49 | 2.73 | 5 | 11 |
| MARS | modern | 90.18 | 86.77 | 3.42 | 6 | 16 |
| AdaBelief | classic | 90.14 | 87.69 | 2.45 | 7 | 7 |
| AdEMAMix | modern | 90.13 | 87.20 | 2.93 | 8 | 14 |
| AdamS | modern | 90.10 | 85.87 | 4.23 | 9 | 17 |
| VRAdam | modern | 90.07 | 87.59 | 2.48 | 10 | 8 |
| AdamP | classic | 89.98 | 87.45 | 2.52 | 11 | 12 |
| Lion | modern | 89.93 | 88.40 | 1.53 | 12 | 4 |
| Adan | modern | 89.73 | 81.19 | 8.53 | 13 | 18 |
| RAdam | classic | 88.78 | 88.71 | 0.07 | 14 | 2 |
| Adam | classic | 88.67 | 87.53 | 1.14 | 15 | 10 |
| NAdam | classic | 88.53 | 88.12 | 0.41 | 16 | 5 |
| AMSGrad | classic | 88.52 | 87.30 | 1.22 | 17 | 13 |
| AdaMax | classic | 88.31 | 86.80 | 1.51 | 18 | 15 |

## Cell A: fmnist × bncnn

Environment: Tesla T4, torch 2.11.0+cu130, CUDA 13.0. Sweep/main epochs: 30/30, batch 128.

### LR sweep and LR sensitivity

Final-epoch val acc (%), seed 0. *Mean drop*: average decrease relative to the best LR on the 4-point base grid, diverged counted as 0 (smaller = less sensitive). *Within 1 point*: share of base-grid LRs whose result is within 1 percentage point of the best LR.

| Optimizer | Group | LRs tried -> val acc | Selected LR | Mean drop | Within 1 point | Diverged |
|---|---|---|---:|---:|---:|---:|
| Adam | classic | 0.0001->92.5  0.0003->93.1  0.001->93.8  0.003->93.5 | 0.001 | 0.57 | 75% | 0 |
| AdaMax | classic | 0.0001->91.1  0.0003->92.6  0.001->93.2  0.003->93.3  0.01->93.2 | 0.003 | 0.79 | 75% | 0 |
| NAdam | classic | 0.0001->92.6  0.0003->93.0  0.001->93.7  0.003->93.2 | 0.001 | 0.57 | 75% | 0 |
| AMSGrad | classic | 0.0001->92.6  0.0003->93.2  0.001->93.8  0.003->93.1 | 0.001 | 0.62 | 75% | 0 |
| AdamW | classic | 0.0001->92.6  0.0003->93.1  0.001->93.8  0.003->94.8  0.01->94.3 | 0.003 | 1.19 | 50% | 0 |
| RAdam | classic | 0.0001->92.6  0.0003->93.1  0.001->93.7  0.003->93.4 | 0.001 | 0.51 | 75% | 0 |
| AdaBelief | classic | 0.0001->91.8  0.0003->93.1  0.001->93.7  0.003->94.2  0.01->94.0 | 0.003 | 1.04 | 50% | 0 |
| AdamP | classic | 0.0001->92.5  0.0003->93.3  0.001->93.9  0.003->94.6  0.01->94.4 | 0.003 | 1.02 | 50% | 0 |
| Adan | modern | 0.0003->92.4  0.001->93.3  0.003->93.9  0.01->94.2  0.03->93.9 | 0.01 | 0.80 | 75% | 0 |
| Lion | modern | 1e-05->92.0  3e-05->92.4  0.0001->93.6  0.0003->94.2  0.001->94.1 | 0.0003 | 1.18 | 50% | 0 |
| MARS | modern | 0.0001->92.3  0.0003->92.8  0.001->94.0  0.003->94.3  0.01->94.3  0.03->93.9 | 0.01 | 1.01 | 50% | 0 |
| AdEMAMix | modern | 0.0001->92.7  0.0003->93.0  0.001->93.4  0.003->93.7  0.01->94.1  0.03->94.0 | 0.01 | 0.89 | 50% | 0 |
| C-AdamW | modern | 0.0001->92.5  0.0003->93.2  0.001->94.1  0.003->94.4  0.01->94.7  0.03->94.1 | 0.01 | 1.11 | 50% | 0 |
| SF-AdamW | modern | 0.0003->92.7  0.001->93.2  0.003->93.5  0.01->93.8  0.03->94.1  0.1->94.4 | 0.1 | 1.08 | 50% | 0 |
| SPAM | modern | 0.0001->92.7  0.0003->93.7  0.001->94.1  0.003->94.3  0.01->94.2 | 0.003 | 0.57 | 75% | 0 |
| AdamS | modern | 0.0001->92.8  0.0003->94.0  0.001->94.3  0.003->94.5  0.01->94.0 | 0.003 | 0.56 | 75% | 0 |
| VRAdam | modern | 0.0003->93.1  0.001->93.9  0.003->94.1  0.01->94.6  0.03->94.1 | 0.01 | 0.69 | 75% | 0 |
| SGD-M | reference | 0.01->93.2  0.03->94.3  0.1->94.4  0.3->94.2 | 0.1 | 0.36 | 75% | 0 |

### Main runs: final-epoch test acc (%)

Mean ± std over seeds, [95% CI]. Δ and p versus **Adam**: paired t-test by seed (same seed = same initialization and batch order), Holm-adjusted p. *Epochs to target*: first epoch reaching 99% of the best accuracy of the cell (93.10%).

| Optimizer | LR | n | Test acc | 95% CI | Δ vs Adam | p (Holm) | Train-test gap | Epochs to target | ms/step | State MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MARS ★ | 0.01 | 5 | 94.04 ± 0.08 | ±0.10 | +0.87 * | 0.001 | 5.96 | 19.2 (5/5) | 13.1 | 6.4 |
| AdamS | 0.003 | 5 | 93.95 ± 0.10 | ±0.12 | +0.78 * | 0.001 | 6.05 | 16.2 (5/5) | 12.8 | 2.1 |
| SGD-M | 0.1 | 5 | 93.95 ± 0.10 | ±0.12 | +0.77 * | 0.001 | 6.05 | 20.4 (5/5) | 11.4 | 2.1 |
| SPAM | 0.003 | 5 | 93.94 ± 0.12 | ±0.15 | +0.76 * | 0.004 | 6.06 | 8.6 (5/5) | 13.8 | 4.3 |
| C-AdamW | 0.01 | 5 | 93.90 ± 0.11 | ±0.14 | +0.73 * | 0.005 | 6.10 | 19.8 (5/5) | 14.3 | 4.3 |
| VRAdam | 0.01 | 5 | 93.90 ± 0.08 | ±0.10 | +0.72 * | 0.002 | 6.10 | 19.6 (5/5) | 15.0 | 4.3 |
| Adan | 0.01 | 5 | 93.87 ± 0.13 | ±0.16 | +0.69 * | 0.003 | 6.13 | 15.8 (5/5) | 13.5 | 8.5 |
| AdamP | 0.003 | 5 | 93.85 ± 0.08 | ±0.10 | +0.67 * | <0.001 | 6.15 | 18.8 (5/5) | 14.7 | 4.3 |
| SF-AdamW | 0.1 | 5 | 93.84 ± 0.27 | ±0.34 | +0.67 * | 0.018 | 0.77 | 4.0 (5/5) | 12.3 | 4.3 |
| AdamW | 0.003 | 5 | 93.84 ± 0.21 | ±0.26 | +0.66 * | 0.016 | 6.16 | 14.8 (5/5) | 11.6 | 4.3 |
| Lion | 0.0003 | 5 | 93.68 ± 0.14 | ±0.17 | +0.51 * | 0.013 | 6.32 | 15.0 (5/5) | 12.1 | 2.1 |
| AdaBelief | 0.003 | 5 | 93.68 ± 0.14 | ±0.17 | +0.50 * | 0.016 | 6.32 | 18.8 (5/5) | 12.7 | 4.3 |
| AdEMAMix | 0.01 | 5 | 93.64 ± 0.08 | ±0.10 | +0.46 * | 0.001 | 6.36 | 16.0 (5/5) | 13.1 | 6.4 |
| AMSGrad | 0.001 | 5 | 93.34 ± 0.16 | ±0.20 | +0.17 | 0.460 | 6.66 | 18.4 (5/5) | 11.7 | 6.4 |
| AdaMax | 0.003 | 5 | 93.25 ± 0.09 | ±0.11 | +0.07 | 0.460 | 6.75 | 21.8 (5/5) | 11.6 | 4.3 |
| NAdam | 0.001 | 5 | 93.19 ± 0.18 | ±0.23 | +0.02 | 1.000 | 6.81 | 21.0 (5/5) | 11.9 | 4.3 |
| Adam | 0.001 | 5 | 93.18 ± 0.06 | ±0.07 | - | - | 6.82 | 23.0 (5/5) | 11.7 | 4.3 |
| RAdam | 0.001 | 5 | 93.14 ± 0.21 | ±0.26 | -0.03 | 1.000 | 6.86 | 22.2 (4/5) | 11.8 | 4.3 |

★ best of the cell; * significantly different from Adam at 0.05 after Holm correction. No epoch selection on the test set.

## Cell B: cifar10 × bncnn

Environment: Tesla T4, torch 2.11.0+cu130, CUDA 13.0. Sweep/main epochs: 50/50, batch 128.

### LR sweep and LR sensitivity

Final-epoch val acc (%), seed 0. *Mean drop*: average decrease relative to the best LR on the 4-point base grid, diverged counted as 0 (smaller = less sensitive). *Within 1 point*: share of base-grid LRs whose result is within 1 percentage point of the best LR.

| Optimizer | Group | LRs tried -> val acc | Selected LR | Mean drop | Within 1 point | Diverged |
|---|---|---|---:|---:|---:|---:|
| Adam | classic | 0.0001->78.3  0.0003->86.4  0.001->89.3  0.003->88.5 | 0.001 | 3.63 | 50% | 0 |
| AdaMax | classic | 0.0001->68.9  0.0003->78.0  0.001->86.9  0.003->89.1  0.01->89.3  0.03->85.9 | 0.01 | 8.56 | 25% | 0 |
| NAdam | classic | 0.0001->78.4  0.0003->86.0  0.001->89.3  0.003->89.3 | 0.001 | 3.59 | 50% | 0 |
| AMSGrad | classic | 0.0001->78.3  0.0003->86.4  0.001->89.4  0.003->88.5 | 0.001 | 3.76 | 50% | 0 |
| AdamW | classic | 0.0001->77.6  0.0003->85.4  0.001->88.8  0.003->89.8  0.01->90.7  0.03->89.1 | 0.01 | 5.28 | 25% | 0 |
| RAdam | classic | 0.0001->77.4  0.0003->85.3  0.001->88.9  0.003->89.4  0.01->86.2 | 0.003 | 4.16 | 50% | 0 |
| AdaBelief | classic | 0.0001->74.8  0.0003->84.2  0.001->88.7  0.003->89.9  0.01->90.6  0.03->87.0 | 0.01 | 6.21 | 25% | 0 |
| AdamP | classic | 0.0001->77.8  0.0003->85.2  0.001->89.5  0.003->90.6  0.01->90.3 | 0.003 | 4.81 | 25% | 0 |
| Adan | modern | 0.0003->79.4  0.001->86.4  0.003->89.0  0.01->89.8  0.03->89.3 | 0.01 | 3.63 | 50% | 0 |
| Lion | modern | 1e-05->73.4  3e-05->81.9  0.0001->87.4  0.0003->89.6  0.001->90.3  0.003->89.7 | 0.001 | 7.19 | 25% | 0 |
| MARS | modern | 0.0001->74.0  0.0003->83.1  0.001->88.5  0.003->89.5  0.01->90.3  0.03->88.6 | 0.01 | 6.54 | 25% | 0 |
| AdEMAMix | modern | 0.0001->81.9  0.0003->86.7  0.001->88.9  0.003->89.6  0.01->90.6  0.03->90.1 | 0.01 | 3.79 | 25% | 0 |
| C-AdamW | modern | 0.0001->78.2  0.0003->85.2  0.001->89.1  0.003->89.5  0.01->90.7  0.03->89.9 | 0.01 | 5.20 | 0% | 0 |
| SF-AdamW | modern | 0.0003->83.0  0.001->87.3  0.003->88.6  0.01->89.6  0.03->91.5  0.1->90.4 | 0.03 | 4.40 | 0% | 0 |
| SPAM | modern | 0.0001->81.6  0.0003->87.3  0.001->89.6  0.003->90.5  0.01->90.6  0.03->88.0 | 0.01 | 3.37 | 25% | 0 |
| AdamS | modern | 0.0001->84.6  0.0003->88.7  0.001->89.7  0.003->90.6  0.01->89.0 | 0.003 | 2.18 | 50% | 0 |
| VRAdam | modern | 0.0003->82.6  0.001->88.6  0.003->89.7  0.01->90.5  0.03->89.6 | 0.01 | 2.66 | 50% | 0 |
| SGD-M | reference | 0.01->87.6  0.03->89.4  0.1->91.0  0.3->90.4 | 0.1 | 1.36 | 50% | 0 |

### Main runs: final-epoch test acc (%)

Mean ± std over seeds, [95% CI]. Δ and p versus **Adam**: paired t-test by seed (same seed = same initialization and batch order), Holm-adjusted p. *Epochs to target*: first epoch reaching 99% of the best accuracy of the cell (89.99%).

| Optimizer | LR | n | Test acc | 95% CI | Δ vs Adam | p (Holm) | Train-test gap | Epochs to target | ms/step | State MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SF-AdamW ★ | 0.03 | 5 | 90.90 ± 0.13 | ±0.16 | +2.23 * | <0.001 | 1.28 | 22.0 (5/5) | 15.4 | 4.3 |
| SPAM | 0.01 | 5 | 90.46 ± 0.18 | ±0.23 | +1.78 * | <0.001 | 8.08 | 41.2 (5/5) | 16.6 | 4.3 |
| C-AdamW | 0.01 | 5 | 90.33 ± 0.20 | ±0.25 | +1.65 * | <0.001 | 8.44 | 44.4 (5/5) | 17.2 | 4.3 |
| SGD-M | 0.1 | 5 | 90.29 ± 0.14 | ±0.17 | +1.62 * | <0.001 | 7.88 | 45.6 (5/5) | 14.5 | 2.1 |
| AdamW | 0.01 | 5 | 90.22 ± 0.19 | ±0.23 | +1.55 * | 0.001 | 8.49 | 44.4 (5/5) | 14.8 | 4.3 |
| MARS | 0.01 | 5 | 90.18 ± 0.15 | ±0.18 | +1.51 * | <0.001 | 8.43 | 43.8 (4/5) | 16.0 | 6.4 |
| AdaBelief | 0.01 | 5 | 90.14 ± 0.10 | ±0.13 | +1.46 * | <0.001 | 7.51 | 46.6 (5/5) | 15.7 | 4.3 |
| AdEMAMix | 0.01 | 5 | 90.13 ± 0.12 | ±0.14 | +1.45 * | <0.001 | 9.34 | 42.4 (5/5) | 16.1 | 6.4 |
| AdamS | 0.003 | 5 | 90.10 ± 0.27 | ±0.34 | +1.42 * | 0.001 | 9.12 | 40.8 (4/5) | 15.7 | 2.1 |
| VRAdam | 0.01 | 5 | 90.07 ± 0.21 | ±0.27 | +1.40 * | <0.001 | 8.63 | 43.0 (4/5) | 17.7 | 4.3 |
| AdamP | 0.003 | 5 | 89.98 ± 0.15 | ±0.18 | +1.30 * | <0.001 | 9.09 | 45.2 (4/5) | 17.6 | 4.3 |
| Lion | 0.001 | 5 | 89.93 ± 0.21 | ±0.26 | +1.26 * | <0.001 | 8.94 | 47.5 (2/5) | 15.0 | 2.1 |
| Adan | 0.01 | 5 | 89.73 ± 0.35 | ±0.44 | +1.05 * | 0.027 | 8.94 | 47.5 (2/5) | 16.4 | 8.5 |
| RAdam | 0.003 | 5 | 88.78 ± 0.19 | ±0.24 | +0.11 | 0.493 | 8.70 | not reached (0/5) | 14.9 | 4.3 |
| Adam | 0.001 | 5 | 88.67 ± 0.09 | ±0.12 | - | - | 9.85 | not reached (0/5) | 15.0 | 4.3 |
| NAdam | 0.001 | 5 | 88.53 ± 0.19 | ±0.23 | -0.14 | 0.493 | 9.99 | not reached (0/5) | 15.1 | 4.3 |
| AMSGrad | 0.001 | 5 | 88.52 ± 0.17 | ±0.21 | -0.15 | 0.493 | 9.91 | not reached (0/5) | 15.0 | 6.4 |
| AdaMax | 0.01 | 5 | 88.31 ± 0.34 | ±0.43 | -0.36 | 0.479 | 7.76 | not reached (0/5) | 14.5 | 4.3 |

★ best of the cell; * significantly different from Adam at 0.05 after Holm correction. No epoch selection on the test set.

## Cell C: cifar10 × resnet18

Environment: Tesla T4, torch 2.11.0+cu128, CUDA 12.8. Sweep/main epochs: 30/100, batch 128.

### LR sweep and LR sensitivity

Final-epoch val acc (%), seed 0. *Mean drop*: average decrease relative to the best LR on the 4-point base grid, diverged counted as 0 (smaller = less sensitive). *Within 1 point*: share of base-grid LRs whose result is within 1 percentage point of the best LR.

| Optimizer | Group | LRs tried -> val acc | Selected LR | Mean drop | Within 1 point | Diverged |
|---|---|---|---:|---:|---:|---:|
| Adam | classic | 0.0001->89.8  0.0003->92.3  0.001->92.3  0.003->89.8 | 0.0003 | 1.27 | 50% | 0 |
| AdaMax | classic | 0.0001->84.4  0.0003->89.9  0.001->92.3  0.003->91.8 | 0.001 | 2.73 | 50% | 0 |
| NAdam | classic | 0.0001->89.9  0.0003->92.7  0.001->92.7  0.003->90.3 | 0.0003 | 1.33 | 50% | 0 |
| AMSGrad | classic | 0.0001->89.8  0.0003->92.5  0.001->92.6  0.003->89.8 | 0.001 | 1.38 | 50% | 0 |
| AdamW | classic | 0.0001->89.8  0.0003->92.3  0.001->92.5  0.003->92.5 | 0.001 | 0.74 | 75% | 0 |
| RAdam | classic | 0.0001->88.8  0.0003->92.1  0.001->93.0  0.003->91.9 | 0.001 | 1.54 | 50% | 0 |
| AdaBelief | classic | 0.0001->89.0  0.0003->91.8  0.001->92.9  0.003->93.5  0.01->93.1 | 0.003 | 1.67 | 50% | 0 |
| AdamP | classic | 0.0001->90.1  0.0003->92.8  0.001->93.3  0.003->93.5  0.01->92.3 | 0.003 | 1.04 | 75% | 0 |
| Adan | modern | 0.0003->90.6  0.001->92.0  0.003->92.2  0.01->92.4  0.03->91.0 | 0.01 | 0.62 | 75% | 0 |
| Lion | modern | 1e-05->87.5  3e-05->91.1  0.0001->92.6  0.0003->92.6  0.001->92.7  0.003->91.5 | 0.001 | 1.80 | 50% | 0 |
| MARS | modern | 0.0001->88.3  0.0003->91.6  0.001->91.9  0.003->91.8 | 0.001 | 0.98 | 75% | 0 |
| AdEMAMix | modern | 0.0001->90.1  0.0003->91.6  0.001->90.2  0.003->89.8 | 0.0003 | 1.19 | 25% | 0 |
| C-AdamW | modern | 0.0001->90.3  0.0003->92.3  0.001->92.3  0.003->92.4  0.01->92.7  0.03->div | 0.01 | 0.86 | 75% | 1 |
| SF-AdamW | modern | 0.0003->90.5  0.001->92.4  0.003->92.6  0.01->94.0  0.03->93.7 | 0.01 | 1.62 | 25% | 0 |
| SPAM | modern | 0.0001->91.4  0.0003->92.8  0.001->92.1  0.003->92.4 | 0.0003 | 0.66 | 75% | 0 |
| AdamS | modern | 3e-05->89.5  0.0001->91.9  0.0003->91.5  0.001->91.1  0.003->90.7 | 0.0001 | 0.63 | 75% | 0 |
| VRAdam | modern | 0.0003->90.1  0.001->92.7  0.003->92.3  0.01->92.9  0.03->90.7 | 0.01 | 0.90 | 75% | 0 |
| SGD-M | reference | 0.01->91.7  0.03->92.6  0.1->92.9  0.3->92.7 | 0.1 | 0.40 | 75% | 0 |

### Main runs: final-epoch test acc (%)

Mean ± std over seeds, [95% CI]. Δ and p versus **Adam**: paired t-test by seed (same seed = same initialization and batch order), Holm-adjusted p. *Epochs to target*: first epoch reaching 99% of the best accuracy of the cell (93.75%).

| Optimizer | LR | n | Test acc | 95% CI | Δ vs Adam | p (Holm) | Train-test gap | Epochs to target | ms/step | State MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AdaBelief ★ | 0.003 | 5 | 94.70 ± 0.06 | ±0.08 | +1.14 * | <0.001 | 5.28 | 80.4 (5/5) | 53.9 | 85.3 |
| SF-AdamW | 0.01 | 5 | 94.69 ± 0.10 | ±0.13 | +1.13 * | <0.001 | 3.59 | 35.4 (5/5) | 53.0 | 85.3 |
| SGD-M | 0.1 | 5 | 94.61 ± 0.13 | ±0.16 | +1.05 * | 0.001 | 5.37 | 82.2 (5/5) | 48.7 | 42.6 |
| AdamP | 0.003 | 5 | 94.26 ± 0.15 | ±0.19 | +0.70 * | 0.011 | 5.71 | 83.6 (5/5) | 62.6 | 85.3 |
| Adan | 0.01 | 5 | 94.26 ± 0.25 | ±0.31 | +0.70 | 0.063 | 5.71 | 88.8 (5/5) | 57.3 | 170.5 |
| VRAdam | 0.01 | 5 | 94.07 ± 0.27 | ±0.33 | +0.51 * | 0.022 | 5.89 | 89.6 (5/5) | 57.8 | 85.3 |
| C-AdamW | 0.01 | 5 | 93.92 ± 0.15 | ±0.19 | +0.36 | 0.110 | 6.04 | 91.0 (5/5) | 58.7 | 85.3 |
| Lion | 0.001 | 5 | 93.58 ± 0.21 | ±0.26 | +0.02 | 1.000 | 6.35 | 98.0 (2/5) | 49.5 | 42.6 |
| Adam | 0.0003 | 5 | 93.56 ± 0.13 | ±0.16 | - | - | 6.42 | 95.0 (1/5) | 50.5 | 85.3 |
| NAdam | 0.0003 | 5 | 93.56 ± 0.17 | ±0.22 | +0.00 | 1.000 | 6.42 | 88.0 (1/5) | 51.3 | 85.3 |
| AdamW | 0.001 | 5 | 93.51 ± 0.13 | ±0.16 | -0.05 | 1.000 | 6.48 | not reached (0/5) | 50.2 | 85.3 |
| MARS | 0.001 | 5 | 93.47 ± 0.26 | ±0.33 | -0.09 | 1.000 | 6.52 | 95.0 (1/5) | 56.4 | 127.9 |
| SPAM | 0.0003 | 5 | 93.41 ± 0.10 | ±0.12 | -0.15 | 0.470 | 6.58 | not reached (0/5) | 58.3 | 85.3 |
| AdaMax | 0.001 | 5 | 93.41 ± 0.26 | ±0.33 | -0.15 | 1.000 | 6.58 | 95.0 (1/5) | 50.1 | 85.3 |
| RAdam | 0.001 | 5 | 93.09 ± 0.14 | ±0.17 | -0.47 | 0.063 | 6.83 | not reached (0/5) | 50.4 | 85.3 |
| AdamS | 0.0001 | 5 | 93.06 ± 0.16 | ±0.20 | -0.50 | 0.067 | 6.93 | not reached (0/5) | 55.2 | 42.6 |
| AMSGrad | 0.001 | 5 | 93.02 ± 0.19 | ±0.23 | -0.54 * | 0.040 | 6.90 | not reached (0/5) | 50.9 | 127.9 |
| AdEMAMix | 0.0003 | 5 | 92.64 ± 0.11 | ±0.13 | -0.92 * | 0.003 | 7.31 | not reached (0/5) | 57.0 | 127.9 |

★ best of the cell; * significantly different from Adam at 0.05 after Holm correction. No epoch selection on the test set.

## Cell D: cifar100 × bncnn

Environment: Tesla T4, torch 2.11.0+cu128, CUDA 12.8. Sweep/main epochs: 50/50, batch 128.

### LR sweep and LR sensitivity

Final-epoch val acc (%), seed 0. *Mean drop*: average decrease relative to the best LR on the 4-point base grid, diverged counted as 0 (smaller = less sensitive). *Within 1 point*: share of base-grid LRs whose result is within 1 percentage point of the best LR.

| Optimizer | Group | LRs tried -> val acc | Selected LR | Mean drop | Within 1 point | Diverged |
|---|---|---|---:|---:|---:|---:|
| Adam | classic | 0.0001->43.6  0.0003->57.7  0.001->64.2  0.003->65.4  0.01->53.5 | 0.003 | 7.67 | 25% | 0 |
| AdaMax | classic | 0.0001->30.5  0.0003->42.4  0.001->57.7  0.003->64.7  0.01->65.3  0.03->57.9 | 0.01 | 16.49 | 25% | 0 |
| NAdam | classic | 0.0001->43.1  0.0003->56.8  0.001->64.5  0.003->65.2  0.01->57.1 | 0.003 | 7.76 | 50% | 0 |
| AMSGrad | classic | 0.0001->43.5  0.0003->57.6  0.001->64.7  0.003->66.0  0.01->54.0 | 0.003 | 8.06 | 25% | 0 |
| AdamW | classic | 0.0001->42.4  0.0003->56.4  0.001->63.9  0.003->65.5  0.01->66.9  0.03->59.9 | 0.01 | 9.89 | 0% | 0 |
| RAdam | classic | 0.0001->41.7  0.0003->55.9  0.001->62.9  0.003->66.4  0.01->61.6 | 0.003 | 9.63 | 25% | 0 |
| AdaBelief | classic | 0.0001->37.6  0.0003->54.3  0.001->63.8  0.003->65.7  0.01->67.3  0.03->56.3 | 0.01 | 12.01 | 0% | 0 |
| AdamP | classic | 0.0001->42.2  0.0003->56.8  0.001->65.0  0.003->67.0  0.01->68.7  0.03->55.5 | 0.01 | 10.99 | 0% | 0 |
| Adan | modern | 0.0003->45.7  0.001->59.4  0.003->64.6  0.01->66.1  0.03->63.6 | 0.01 | 7.14 | 25% | 0 |
| Lion | modern | 1e-05->35.2  3e-05->48.6  0.0001->60.2  0.0003->64.1  0.001->67.3  0.003->68.1 | 0.003 | 16.12 | 0% | 0 |
| MARS | modern | 0.0001->37.0  0.0003->51.2  0.001->61.9  0.003->64.6  0.01->67.2  0.03->59.0 | 0.01 | 13.52 | 0% | 0 |
| AdEMAMix | modern | 0.0001->49.8  0.0003->60.1  0.001->63.6  0.003->64.8  0.01->66.2  0.03->60.4 | 0.01 | 6.59 | 0% | 0 |
| C-AdamW | modern | 0.0001->43.7  0.0003->56.8  0.001->63.9  0.003->65.3  0.01->67.2  0.03->57.1 | 0.01 | 9.77 | 0% | 0 |
| SF-AdamW | modern | 0.0003->51.2  0.001->61.1  0.003->64.1  0.01->67.0  0.03->69.1  0.1->68.6 | 0.03 | 8.20 | 0% | 0 |
| SPAM | modern | 0.0001->48.9  0.0003->60.5  0.001->65.1  0.003->66.7  0.01->67.9  0.03->57.9 | 0.01 | 7.64 | 0% | 0 |
| AdamS | modern | 0.0001->54.5  0.0003->63.0  0.001->64.8  0.003->65.4  0.01->58.8 | 0.003 | 3.45 | 50% | 0 |
| VRAdam | modern | 0.0003->48.5  0.001->62.6  0.003->65.6  0.01->66.6  0.03->60.6 | 0.01 | 5.74 | 50% | 0 |
| SGD-M | reference | 0.01->60.0  0.03->65.4  0.1->67.3  0.3->68.5  1->22.7 | 0.3 | 3.15 | 25% | 0 |

### Main runs: final-epoch test acc (%)

Mean ± std over seeds, [95% CI]. Δ and p versus **Adam**: paired t-test by seed (same seed = same initialization and batch order), Holm-adjusted p. *Epochs to target*: first epoch reaching 99% of the best accuracy of the cell (67.57%).

| Optimizer | LR | n | Test acc | 95% CI | Δ vs Adam | p (Holm) | Train-test gap | Epochs to target | ms/step | State MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SF-AdamW ★ | 0.03 | 5 | 68.25 ± 0.14 | ±0.18 | +3.02 * | <0.001 | 6.77 | 28.6 (5/5) | 14.7 | 4.4 |
| AdamP | 0.01 | 5 | 67.82 ± 0.27 | ±0.34 | +2.58 * | <0.001 | 18.93 | 45.6 (5/5) | 16.8 | 4.4 |
| SGD-M | 0.3 | 5 | 67.49 ± 0.37 | ±0.46 | +2.26 * | 0.004 | 14.77 | 48.0 (2/5) | 13.7 | 2.2 |
| C-AdamW | 0.01 | 5 | 67.11 ± 0.32 | ±0.39 | +1.88 * | 0.002 | 22.58 | 47.0 (1/5) | 16.3 | 4.4 |
| Lion | 0.003 | 5 | 67.11 ± 0.24 | ±0.30 | +1.87 * | 0.004 | 18.43 | not reached (0/5) | 14.3 | 2.2 |
| SPAM | 0.01 | 5 | 66.98 ± 0.23 | ±0.28 | +1.74 * | 0.009 | 22.95 | not reached (0/5) | 16.2 | 4.4 |
| MARS | 0.01 | 5 | 66.93 ± 0.42 | ±0.53 | +1.70 * | 0.003 | 21.09 | 47.0 (1/5) | 15.2 | 6.7 |
| AdaBelief | 0.01 | 5 | 66.89 ± 0.35 | ±0.44 | +1.66 * | 0.017 | 17.42 | not reached (0/5) | 14.8 | 4.4 |
| AdamW | 0.01 | 5 | 66.88 ± 0.26 | ±0.32 | +1.65 * | 0.004 | 22.16 | not reached (0/5) | 13.9 | 4.4 |
| VRAdam | 0.01 | 5 | 66.82 ± 0.30 | ±0.38 | +1.59 * | 0.003 | 22.19 | not reached (0/5) | 16.9 | 4.4 |
| Adan | 0.01 | 5 | 65.95 ± 0.22 | ±0.27 | +0.71 * | 0.039 | 20.71 | not reached (0/5) | 15.6 | 8.9 |
| AdEMAMix | 0.01 | 5 | 65.54 ± 0.44 | ±0.55 | +0.31 | 0.843 | 29.39 | not reached (0/5) | 15.4 | 6.7 |
| NAdam | 0.003 | 5 | 65.34 ± 0.31 | ±0.38 | +0.11 | 0.843 | 19.48 | not reached (0/5) | 14.1 | 4.4 |
| RAdam | 0.003 | 5 | 65.30 ± 0.55 | ±0.68 | +0.07 | 0.843 | 19.75 | not reached (0/5) | 14.1 | 4.4 |
| Adam | 0.003 | 5 | 65.23 ± 0.22 | ±0.28 | - | - | 17.18 | not reached (0/5) | 13.9 | 4.4 |
| AdaMax | 0.01 | 5 | 65.05 ± 0.10 | ±0.13 | -0.19 | 0.843 | 16.34 | not reached (0/5) | 13.7 | 4.4 |
| AMSGrad | 0.003 | 5 | 64.99 ± 0.22 | ±0.28 | -0.24 | 0.843 | 18.25 | not reached (0/5) | 14.0 | 6.7 |
| AdamS | 0.003 | 5 | 64.89 ± 0.25 | ±0.31 | -0.34 | 0.545 | 24.06 | not reached (0/5) | 15.3 | 2.2 |

★ best of the cell; * significantly different from Adam at 0.05 after Holm correction. No epoch selection on the test set.

## Cell E: cifar10 × plaincnn

Environment: Tesla T4, torch 2.11.0+cu128, CUDA 12.8. Sweep/main epochs: 50/50, batch 128.

### LR sweep and LR sensitivity

Final-epoch val acc (%), seed 0. *Mean drop*: average decrease relative to the best LR on the 4-point base grid, diverged counted as 0 (smaller = less sensitive). *Within 1 point*: share of base-grid LRs whose result is within 1 percentage point of the best LR.

| Optimizer | Group | LRs tried -> val acc | Selected LR | Mean drop | Within 1 point | Diverged |
|---|---|---|---:|---:|---:|---:|
| Adam | classic | 0.0001->70.2  0.0003->83.5  0.001->88.0  0.003->84.7 | 0.001 | 6.42 | 25% | 0 |
| AdaMax | classic | 0.0001->57.2  0.0003->69.6  0.001->83.2  0.003->86.9  0.01->9.5 | 0.003 | 12.62 | 25% | 0 |
| NAdam | classic | 0.0001->69.9  0.0003->83.2  0.001->88.6  0.003->87.4 | 0.001 | 6.28 | 25% | 0 |
| AMSGrad | classic | 0.0001->69.6  0.0003->83.0  0.001->87.6  0.003->82.8 | 0.001 | 6.85 | 25% | 0 |
| AdamW | classic | 0.0001->70.0  0.0003->83.4  0.001->87.8  0.003->87.1 | 0.001 | 5.75 | 50% | 0 |
| RAdam | classic | 0.0001->67.5  0.0003->81.4  0.001->88.2  0.003->88.8  0.01->85.8 | 0.003 | 7.34 | 50% | 0 |
| AdaBelief | classic | 0.0001->67.2  0.0003->80.8  0.001->87.3  0.003->88.4  0.01->9.6 | 0.003 | 7.45 | 25% | 0 |
| AdamP | classic | 0.0001->70.0  0.0003->83.4  0.001->88.2  0.003->86.9 | 0.001 | 6.06 | 25% | 0 |
| Adan | modern | 0.0003->69.4  0.001->82.4  0.003->83.1  0.01->9.6 | 0.003 | 22.00 | 50% | 0 |
| Lion | modern | 1e-05->62.9  3e-05->74.8  0.0001->86.1  0.0003->88.4  0.001->85.8 | 0.0003 | 10.34 | 25% | 0 |
| MARS | modern | 0.0001->64.5  0.0003->77.1  0.001->87.3  0.003->86.7 | 0.001 | 8.37 | 50% | 0 |
| AdEMAMix | modern | 0.0001->76.8  0.0003->86.2  0.001->87.6  0.003->84.8 | 0.001 | 3.76 | 25% | 0 |
| C-AdamW | modern | 0.0001->69.5  0.0003->83.2  0.001->88.3  0.003->9.5 | 0.001 | 25.62 | 25% | 0 |
| SF-AdamW | modern | 0.0003->76.8  0.001->86.1  0.003->88.7  0.01->89.9  0.03->div | 0.01 | 4.52 | 25% | 1 |
| SPAM | modern | 0.0001->77.0  0.0003->86.3  0.001->88.4  0.003->85.8 | 0.001 | 4.03 | 25% | 0 |
| AdamS | modern | 0.0001->83.5  0.0003->86.9  0.001->79.7  0.003->9.5 | 0.0003 | 21.96 | 25% | 0 |
| VRAdam | modern | 0.0003->71.5  0.001->88.6  0.003->88.5  0.01->86.9 | 0.001 | 4.74 | 50% | 0 |
| SGD-M | reference | 0.01->86.4  0.03->88.6  0.1->88.7  0.3->div | 0.1 | 22.76 | 50% | 1 |

### Main runs: final-epoch test acc (%)

Mean ± std over seeds, [95% CI]. Δ and p versus **Adam**: paired t-test by seed (same seed = same initialization and batch order), Holm-adjusted p. *Epochs to target*: first epoch reaching 99% of the best accuracy of the cell (88.83%).

| Optimizer | LR | n | Test acc | 95% CI | Δ vs Adam | p (Holm) | Train-test gap | Epochs to target | ms/step | State MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SF-AdamW ★ | 0.01 | 5 | 89.72 ± 0.20 | ±0.25 | +2.19 * | 0.006 | 5.65 | 29.0 (5/5) | 12.6 | 4.3 |
| RAdam | 0.003 | 5 | 88.71 ± 0.18 | ±0.22 | +1.17 * | 0.033 | 6.67 | 44.0 (1/5) | 12.3 | 4.3 |
| SPAM | 0.001 | 5 | 88.52 ± 0.25 | ±0.31 | +0.98 * | 0.031 | 10.07 | 44.0 (1/5) | 13.6 | 4.3 |
| Lion | 0.0003 | 5 | 88.40 ± 0.31 | ±0.38 | +0.87 | 0.379 | 9.67 | not reached (0/5) | 12.4 | 2.1 |
| NAdam | 0.001 | 5 | 88.12 ± 0.18 | ±0.22 | +0.58 | 0.066 | 8.95 | not reached (0/5) | 12.4 | 4.3 |
| C-AdamW | 0.001 | 5 | 87.94 ± 0.36 | ±0.44 | +0.41 | 0.753 | 10.24 | not reached (0/5) | 13.9 | 4.3 |
| AdaBelief | 0.003 | 5 | 87.69 ± 0.55 | ±0.69 | +0.16 | 1.000 | 7.14 | not reached (0/5) | 12.7 | 4.3 |
| VRAdam | 0.001 | 5 | 87.59 ± 0.16 | ±0.20 | +0.06 | 1.000 | 9.88 | not reached (0/5) | 14.1 | 4.3 |
| SGD-M | 0.1 | 5 | 87.54 ± 0.40 | ±0.50 | +0.01 | 1.000 | 6.23 | not reached (0/5) | 11.9 | 2.1 |
| Adam | 0.001 | 5 | 87.53 ± 0.40 | ±0.50 | - | - | 7.28 | not reached (0/5) | 12.2 | 4.3 |
| AdamW | 0.001 | 5 | 87.49 ± 0.21 | ±0.26 | -0.04 | 1.000 | 10.05 | not reached (0/5) | 12.2 | 4.3 |
| AdamP | 0.001 | 5 | 87.45 ± 0.25 | ±0.31 | -0.08 | 1.000 | 10.07 | not reached (0/5) | 15.2 | 4.3 |
| AMSGrad | 0.001 | 5 | 87.30 ± 0.28 | ±0.35 | -0.23 | 0.463 | 7.09 | not reached (0/5) | 12.3 | 6.4 |
| AdEMAMix | 0.001 | 5 | 87.20 ± 0.37 | ±0.45 | -0.33 | 0.234 | 11.43 | not reached (0/5) | 13.2 | 6.4 |
| AdaMax | 0.003 | 5 | 86.80 ± 0.28 | ±0.35 | -0.73 | 0.175 | 6.63 | not reached (0/5) | 12.1 | 4.3 |
| MARS | 0.001 | 5 | 86.77 ± 0.31 | ±0.38 | -0.76 | 0.071 | 7.57 | not reached (0/5) | 13.3 | 6.4 |
| AdamS | 0.0003 | 5 | 85.87 ± 0.53 | ±0.66 | -1.66 * | 0.014 | 7.96 | not reached (0/5) | 12.9 | 2.1 |
| Adan | 0.003 | 5 | 81.19 ± 1.65 | ±2.05 | -6.34 * | 0.006 | 3.27 | not reached (0/5) | 13.2 | 8.5 |

★ best of the cell; * significantly different from Adam at 0.05 after Holm correction. No epoch selection on the test set.
