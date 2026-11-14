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
