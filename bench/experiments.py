"""Follow-up experiments (November 2026) built on the 18-optimizer benchmark.

X1 - Early evaluation: LR search with a COMPLETE but SHORTENED cosine schedule (25% and 50% of the
          epochs) in cells B (with BN) and E (without BN), on the LR grid of the benchmark, 1 seed.
E1 - 2x2 factorial design: update rule x weight-decay form. Runs the two missing corners:
          AdaBelief with decoupled weight decay and SF-AdamW with coupled L2. Cells B, D, E. LR search,
          then 5 seeds (the same number as the existing corners, so that the design is balanced).
E2 - Weight-decay sweep: Adam (coupled L2) and AdamW (decoupled), 3 new coefficients each around
          its own default, cells B and E. The LR is searched again for every coefficient (LR and weight
          decay interact), 3 seeds. The existing coefficient (x1) comes from the benchmark.
E3 - Weight-norm recording: re-run Adam, AdamW, AdaBelief, SF-AdamW at the LR selected in the
          benchmark, cells B and E, 2 seeds. In the same environment the results must match the
          benchmark (reproducibility check), and ||w|| and lr/||w||^2 are logged per epoch.

Each base cell must run in the environment of the benchmark (BASE_ENV): cell B on Colab
(torch cu130), cells D and E on Kaggle (torch cu128).
"""
from . import config

# Environment of the benchmark per cell (from env_lock.json of the 18-optimizer results).
BASE_ENV = {
    "B": dict(gpu="Tesla T4", torch="2.11.0+cu130", cuda="13.0", cudnn=92700),
    "D": dict(gpu="Tesla T4", torch="2.11.0+cu128", cuda="12.8", cudnn=91900),
    "E": dict(gpu="Tesla T4", torch="2.11.0+cu128", cuda="12.8", cudnn=91900),
}

# ---------------------------------------------------------------- X1
X1_OPTS = ("adam", "adamw", "adabelief", "lion", "sfadamw", "cadamw")
# LR grids searched in the benchmark (standard grid + extensions), to compare at the same LR points.
X1_LRS = {
    ("B", "adam"):      [1e-4, 3e-4, 1e-3, 3e-3],
    ("B", "adamw"):     [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    ("B", "adabelief"): [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    ("B", "lion"):      [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
    ("B", "sfadamw"):   [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    ("B", "cadamw"):    [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    ("E", "adam"):      [1e-4, 3e-4, 1e-3, 3e-3],
    ("E", "adamw"):     [1e-4, 3e-4, 1e-3, 3e-3],
    ("E", "adabelief"): [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    ("E", "lion"):      [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
    ("E", "sfadamw"):   [3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    ("E", "cadamw"):    [1e-4, 3e-4, 1e-3, 3e-3],
}
# Suffix of shortened cells: q = 1/4 schedule (13 epochs), h = 1/2 schedule (25 epochs). Base cells B, E have 50 epochs.
X1_BUDGETS = {"q": 13, "h": 25}

# ---------------------------------------------------------------- E1, E2, E3
E1_OPTS = ("adabelief-dw", "sfadamw-l2")
E1_CELLS = ("B", "D", "E")
E1_SEEDS = (0, 1, 2, 3, 4)

# Adam: coupled L2 5e-5, 5e-3, 5e-2 (x0.1, x10, x100); AdamW: decoupled 5e-4, 5e-3, 5e-1 (x0.01, x0.1, x10).
# The two ranges overlap at 5e-4..5e-2, allowing comparison at the same coefficient and at each best one.
E2_OPTS = ("adam-x0.1", "adam-x10", "adam-x100", "adamw-x0.01", "adamw-x0.1", "adamw-x10")
E2_CELLS = ("B", "E")
E2_SEEDS = (0, 1, 2)

E3_OPTS = ("adam", "adamw", "adabelief", "sfadamw")
E3_CELLS = ("B", "E")
E3_SEEDS = (0, 1)
E3_LR = {  # LR selected in the benchmark (best_lr.json)
    "B": {"adam": 1e-3, "adamw": 1e-2, "adabelief": 1e-2, "sfadamw": 3e-2},
    "E": {"adam": 1e-3, "adamw": 1e-3, "adabelief": 3e-3, "sfadamw": 1e-2},
}

# E2b - added after E2: the best L2 coefficient of Adam in cell B lay at the lower edge of the grid (5e-5),
#       so 5e-6 (x0.01) and 5e-7 (x0.001, almost no weight decay) are tried as well. One more LR point (0.1)
#       for AdamW x0.1 in cell B, because its best LR (0.03) lay at the upper edge after 2 extensions.
E2B_OPTS = ("adam-x0.01", "adam-x0.001")
E2B_CELLS = ("B", "E")
E2B_SEEDS = (0, 1, 2)

# E4 - more seeds for the key comparison (Adam with L2 5e-6 vs AdamW): seeds 3, 4 for adam-x0.01
#      (LR selected in E2b), and seeds 2, 3, 4 with weight-norm recording for default Adam and AdamW (as in E3).
E4_SEEDS_X = (3, 4)
E4_LR_X = {"B": 1e-2, "E": 1e-3}
E4_SEEDS_DEF = (2, 3, 4)

EXPERIMENTS = ("X1", "E1", "E2", "E3", "E2b", "E4")


def cell_cfg(cell):
    """Cell configuration; a shortened cell (Bq, Bh, Eq, Eh) = base cell with fewer epochs."""
    base = cell[0]
    cfg = dict(config.CELLS[base])
    if len(cell) == 2:
        cfg["epochs"] = cfg["sweep_epochs"] = X1_BUDGETS[cell[1]]
    return cfg


def base_cell(cell):
    return cell[0]


def units(exp):
    """List of work units. Each unit runs entirely on one worker and does not depend on other units.
    ("fixed", cell, phase, opt, lr, seed)      - one run
    ("tune",  cell, opt, seeds)                - LR search (with grid extension) -> select LR -> main runs
    """
    out = []
    if exp == "X1":
        for b in ("B", "E"):
            for suf in ("q", "h"):
                for o in X1_OPTS:
                    for lr in X1_LRS[(b, o)]:
                        out.append(("fixed", b + suf, "sweep", o, lr, config.SWEEP_SEED))
    elif exp == "E1":
        out = [("tune", c, o, E1_SEEDS) for c in E1_CELLS for o in E1_OPTS]
    elif exp == "E2":
        out = [("tune", c, o, E2_SEEDS) for c in E2_CELLS for o in E2_OPTS]
    elif exp == "E2b":
        out = [("tune", c, o, E2B_SEEDS) for c in E2B_CELLS for o in E2B_OPTS]
        out.append(("fixed", "B", "sweep", "adamw-x0.1", 0.1, config.SWEEP_SEED))
    elif exp == "E4":
        out = [("fixed", c, "final", "adam-x0.01", E4_LR_X[c], s) for c in ("B", "E") for s in E4_SEEDS_X]
        out += [("fixed", c, "final", o, E3_LR[c][o], s)
                for c in ("B", "E") for o in ("adam", "adamw") for s in E4_SEEDS_DEF]
    elif exp == "E3":
        out = [("fixed", c, "final", o, E3_LR[c][o], s)
               for c in E3_CELLS for s in E3_SEEDS for o in E3_OPTS]
    else:
        raise ValueError(f"unknown experiment: {exp}")
    for u in out:
        config.ensure_variant(u[3] if u[0] == "fixed" else u[2])
    return out


def unit_cost(u):
    """Estimated number of epochs (to balance the work across workers)."""
    if u[0] == "fixed":
        return cell_cfg(u[1])["epochs"]
    cfg = cell_cfg(u[1])
    return len(config.lr_grid(u[2])) * cfg["sweep_epochs"] + len(u[3]) * cfg["epochs"]
