"""Experiment configuration: benchmark of Adam-family optimizers (classic vs. modern) on CNNs.

Everything that determines the results lives in this file. Do not edit it while a cell is
running: old and new results would no longer be comparable.
"""

# ------------------------------------------------------------------ optimizer
# group: classic (<=2021) | modern (after 2021) | reference (outside the Adam family)
# wd   : coupled (L2 added to the gradient) | decoupled (subtracted directly from the weights,
#        scaled by lr as in PyTorch) | lion (decoupled, larger because Lion uses a smaller LR)
# grid : LR search grid (see LR_GRIDS)
# sched: cosine (per-step CosineAnnealing) | constant (Schedule-Free handles it, with warm-up)
OPT_SPECS = {
    # --- classic
    "adam":      dict(label="Adam",      group="classic", year=2014, wd="coupled",   grid="std"),
    "adamax":    dict(label="AdaMax",    group="classic", year=2014, wd="coupled",   grid="std"),
    "nadam":     dict(label="NAdam",     group="classic", year=2016, wd="coupled",   grid="std"),
    "amsgrad":   dict(label="AMSGrad",   group="classic", year=2018, wd="coupled",   grid="std"),
    "adamw":     dict(label="AdamW",     group="classic", year=2019, wd="decoupled", grid="std"),
    "radam":     dict(label="RAdam",     group="classic", year=2020, wd="coupled",   grid="std"),
    "adabelief": dict(label="AdaBelief", group="classic", year=2020, wd="coupled",   grid="std"),
    "adamp":     dict(label="AdamP",     group="classic", year=2021, wd="decoupled", grid="std"),
    # --- modern
    "adan":      dict(label="Adan",      group="modern",  year=2022, wd="decoupled", grid="high"),
    "lion":      dict(label="Lion",      group="modern",  year=2023, wd="lion",      grid="lion"),
    "mars":      dict(label="MARS",      group="modern",  year=2024, wd="decoupled", grid="std"),
    "ademamix":  dict(label="AdEMAMix",  group="modern",  year=2024, wd="decoupled", grid="std"),
    "cadamw":    dict(label="C-AdamW",   group="modern",  year=2024, wd="decoupled", grid="std"),
    "sfadamw":   dict(label="SF-AdamW",  group="modern",  year=2024, wd="decoupled", grid="high",
                      sched="constant"),
    # --- added 2026-10-29: Adam-family optimizers released in 2025-2026 (year = first arXiv version)
    "spam":      dict(label="SPAM",      group="modern",  year=2025, wd="decoupled", grid="std"),
    "adams":     dict(label="AdamS",     group="modern",  year=2025, wd="decoupled", grid="std"),
    "vradam":    dict(label="VRAdam",    group="modern",  year=2025, wd="decoupled", grid="high"),
    # --- reference
    "sgd":       dict(label="SGD-M",     group="reference", year=1999, wd="coupled", grid="sgd"),
}
OPTIMIZERS = list(OPT_SPECS)
BASELINE = "adam"

LR_GRIDS = {
    "std":  [1e-4, 3e-4, 1e-3, 3e-3],
    "high": [3e-4, 1e-3, 3e-3, 1e-2],   # Adan and Schedule-Free usually need a larger LR
    "lion": [1e-5, 3e-5, 1e-4, 3e-4],   # Lion: LR 3-10x smaller than AdamW (original paper)
    "sgd":  [1e-2, 3e-2, 1e-1, 3e-1],
}

# Weight decay: not tuned, fixed per form. Stated in the paper.
WEIGHT_DECAY = {"coupled": 5e-4, "decoupled": 5e-2, "lion": 5e-1}

# Optimizer-specific hyperparameters: authors' defaults/recommendations.
BETAS = (0.9, 0.999)
EPS = 1e-8
OPT_EXTRA = {
    "adan":     dict(betas=(0.98, 0.92, 0.99)),
    "lion":     dict(betas=(0.9, 0.99)),
    "mars":     dict(betas=(0.95, 0.99), gamma=0.025),
    "ademamix": dict(betas=(0.9, 0.999, 0.9999), alpha=5.0),  # warm up alpha and beta3 over all T steps
    "sfadamw":  dict(warmup_frac=0.05),
    # SPAM: authors' defaults (reset every 500 steps, 150 warm-up steps, threshold 5000)
    "spam":     dict(spam=dict(delta_t=500, warmup=150, threshold=5000.0)),
    "adams":    dict(betas=(0.9, 0.95)),                  # beta2 = 0.95 as in the AdamS paper
    "vradam":   dict(beta3=1.0, alpha1=29.0),             # beta3 = 1 (recommended); alpha1 from the authors' CNN setup
    "sgd":      dict(momentum=0.9),
}

# LR search phase: the grid of each optimizer x 1 seed.
SWEEP_SEED = 0
# Main phase: best LR (by validation accuracy) x the seeds.
FINAL_SEEDS = [0, 1, 2, 3, 4]

# Fixed validation split taken from the training set (used to select the LR; the test set is NOT used).
VAL_SIZE = 5000
SPLIT_SEED = 12345

# --------------------------------------------------------------------- cells
# Each cell (dataset x model) runs entirely on ONE GPU type.
# require_gpu: substring that must appear in the GPU name; otherwise the runner stops.
# To move a whole cell to another machine: change require_gpu BEFORE the cell starts.
CELLS = {
    "A": dict(dataset="fmnist",   model="bncnn",    epochs=30,  sweep_epochs=30,
              batch_size=128, amp=True, require_gpu="T4"),
    "B": dict(dataset="cifar10",  model="bncnn",    epochs=50,  sweep_epochs=50,
              batch_size=128, amp=True, require_gpu="T4"),
    # Cell C searches the LR with a short 30-epoch schedule to save compute; main runs use 100 epochs.
    # 2026-10-21: moved from an RTX 3050 laptop (faulty GPU, results discarded) to Kaggle T4.
    "C": dict(dataset="cifar10",  model="resnet18", epochs=100, sweep_epochs=30,
              batch_size=128, amp=True, require_gpu="T4"),
    "D": dict(dataset="cifar100", model="bncnn",    epochs=50,  sweep_epochs=50,
              batch_size=128, amp=True, require_gpu="T4"),
    # Ablation: same architecture as cell B but WITHOUT BatchNorm.
    "E": dict(dataset="cifar10",  model="plaincnn", epochs=50,  sweep_epochs=50,
              batch_size=128, amp=True, require_gpu="T4"),
    # --- cells added for the follow-up study (not part of the 18-optimizer benchmark) ---
    # F: ResNet-18 WITHOUT BatchNorm (ablation of cell C). H: the same with a zero-initialized
    # residual scale (fallback, only used if F cannot be trained). G: ResNet-18 on CIFAR-100.
    "F": dict(dataset="cifar10",  model="resnet18nb",  epochs=100, sweep_epochs=30,
              batch_size=128, amp=True, require_gpu="T4"),
    "G": dict(dataset="cifar100", model="resnet18",    epochs=100, sweep_epochs=30,
              batch_size=128, amp=True, require_gpu="T4"),
    "H": dict(dataset="cifar10",  model="resnet18nbz", epochs=100, sweep_epochs=30,
              batch_size=128, amp=True, require_gpu="T4"),
}
# With/without-BN comparison pair (used in analyze.py).
BN_ABLATION = ("B", "E")


def lr_grid(opt):
    return LR_GRIDS[OPT_SPECS[opt]["grid"]]


def weight_decay(opt):
    return WEIGHT_DECAY[OPT_SPECS[opt]["wd"]]


def schedule(opt):
    return OPT_SPECS[opt].get("sched", "cosine")


# ------------------------------------------------- variants (experiments of November 2026)
# Variant name = base name + suffixes, separated by "-":
#   -l2  : L2 weight decay added to the gradient (coupled)
#   -dw  : decoupled weight decay
#   -xK  : multiply the default weight decay of that form by K (e.g. -x10, -x0.1)
# Examples: "sfadamw-l2", "adabelief-dw", "adam-x10", "adamw-x0.01".
# The base optimizers above are NOT changed, so earlier results still match.
VARIANT_BASES = ("adam", "adamw", "adabelief", "sfadamw")


def parse_variant(name):
    """Return (base, weight-decay form, multiplier). A base name returns (name, default form, 1.0)."""
    if name in OPT_SPECS and "base" not in OPT_SPECS[name]:
        return name, OPT_SPECS[name]["wd"], 1.0
    base, *mods = name.split("-")
    if base not in VARIANT_BASES or not mods:
        raise ValueError(f"invalid variant: {name}")
    form, mult = OPT_SPECS[base]["wd"], 1.0
    for m in mods:
        if m == "l2":
            form = "coupled"
        elif m == "dw":
            form = "decoupled"
        elif m.startswith("x"):
            mult = float(m[1:])
            if not mult > 0:
                raise ValueError(f"weight-decay multiplier must be > 0: {name}")
        else:
            raise ValueError(f"unknown suffix '{m}' in {name}")
    return base, form, mult


def ensure_variant(name):
    """Register a variant in OPT_SPECS (not added to OPTIMIZERS). Call before running."""
    if name in OPT_SPECS:
        return name
    base, form, mult = parse_variant(name)
    spec = dict(OPT_SPECS[base])
    spec.update(base=base, wd=form, wd_mult=mult, group="variant",
                label=f"{spec['label']} ({'L2' if form == 'coupled' else 'decoupled'}, ×{mult:g} wd)")
    OPT_SPECS[name] = spec
    return name


def weight_decay(opt):  # noqa: F811  (replaces the version above: adds the multiplier for variants)
    s = OPT_SPECS[opt]
    return WEIGHT_DECAY[s["wd"]] * s.get("wd_mult", 1.0)
