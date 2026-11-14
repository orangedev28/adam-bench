"""Summarize the benchmark results (no GPU/PyTorch needed; matplotlib for the plots).

  python analyze.py --out results            # every cell with results + pooled analysis
  python analyze.py --out results --cell A   # one cell

Each cell -> results/<cell>/analysis/
    summary.md               quick-read tables
    sweep.csv                LR sweep: optimizer, lr, val acc, diverged
    lr_sensitivity.csv       LR sensitivity
    final_runs.csv           every main run
    final_summary.csv        mean, std, 95% CI, difference from Adam, p (paired t + Holm), secondary metrics
  curves_test_acc.png, curves_train_loss.png, lr_sensitivity.png
Pooled -> results/
    summary_all.md           all cells + pooled ranking + BN ablation
    ranking.csv, ranking.png average ranks (Friedman + Nemenyi CD)
    bn_ablation.csv          cell B (with BN) vs cell E (without BN)
"""
import argparse
import csv
import json
import math
from pathlib import Path

from bench import config
from bench import stats as S

SPEC = config.OPT_SPECS
ORDER = config.OPTIMIZERS
BASE = config.BASELINE
LABEL = {o: SPEC[o]["label"] for o in ORDER}

# Colors per optimizer, fixed within each group (8-color palette checked for color blindness).
# Dashes/dots are a second encoding so that black-and-white prints remain readable.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DASHES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
GROUP_PANELS = [("classic", "Classic (<= 2021)"), ("modern", "Modern (> 2021)")]
REF_STYLE = {"adam": dict(color="#3d3d3a", ls="-", lw=1.4), "sgd": dict(color="#8a8984", ls="--", lw=1.4)}
INK, INK2, GRID, SURF = "#1a1a19", "#5f5e5a", "#e6e5e0", "#fcfcfb"


def style_of(o, panel_group):
    """(color, ls, lw, is_reference) for optimizer o in the panel of group panel_group."""
    if SPEC[o]["group"] == panel_group:
        members = [x for x in ORDER if SPEC[x]["group"] == panel_group]
        i = members.index(o)
        return PALETTE[i % 8], DASHES[i % 8], 2.0, False
    st = REF_STYLE.get(o)
    return (st["color"], st["ls"], st["lw"], True) if st else (None, None, None, None)


def in_panel(o, g):
    return SPEC[o]["group"] == g or o in REF_STYLE


def load(cell_dir):
    rows = []
    for p in sorted((cell_dir / "runs").glob("*/result.json")):
        with open(p, encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def opt_sort(names):
    return sorted(names, key=lambda o: (ORDER.index(o) if o in ORDER else 99, o))


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def f2(x, nd=2, scale=100):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{scale * x:.{nd}f}"


def fp(p):
    if p is None or p != p:
        return "-"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("  (matplotlib not available, skipping plots)")
        return None


def _axes_style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(which="both", colors=INK2, labelsize=8)


# ======================================================================= LR sweep
def analyze_sweep(rows, out, md):
    sweep = [r for r in rows if r["phase"] == "sweep"]
    if not sweep:
        return {}
    by = {}
    for r in sweep:
        va = None if r["diverged"] else r["final"]["val_acc"]
        by.setdefault(r["optimizer"], []).append((r["lr"], va))
    write_csv(out / "sweep.csv", ["optimizer", "lr", "val_acc", "diverged"],
              [[o, lr, va, va is None] for o in opt_sort(by) for lr, va in sorted(by[o])])

    sens = {}
    for o in opt_sort(by):
        pts = sorted(by[o])
        ok = [va for _, va in pts if va is not None]
        best = max(ok) if ok else float("nan")
        best_lr = max(pts, key=lambda t: -1 if t[1] is None else t[1])[0]
        # on the optimizer's base grid (extension points are excluded, for a fair 4 points each)
        base = [(lr, va) for lr, va in pts if lr in config.lr_grid(o)]
        drops = [(best - (va if va is not None else 0.0)) for _, va in base]
        sens[o] = dict(best=best, best_lr=best_lr, n=len(pts),
                       n_div=sum(va is None for _, va in pts),
                       mean_drop=sum(drops) / len(drops) if drops else float("nan"),
                       within1=sum((va is not None and best - va <= 0.01) for _, va in base) / max(1, len(base)),
                       pts=pts)
    write_csv(out / "lr_sensitivity.csv",
              ["optimizer", "best_val_acc", "best_lr", "n_lr_tried", "n_diverged",
               "mean_drop_from_best_on_base_grid", "frac_base_grid_within_1pp"],
              [[o, s["best"], s["best_lr"], s["n"], s["n_div"], s["mean_drop"], s["within1"]]
               for o, s in sens.items()])

    md += ["### LR sweep and LR sensitivity", "",
           "Final-epoch val acc (%), seed 0. *Mean drop*: average decrease relative to the best LR on the 4-point base "
           "grid, diverged counted as 0 (smaller = less sensitive). *Within 1 point*: share of base-grid LRs whose result is within "
           "1 percentage point of the best LR.", "",
           "| Optimizer | Group | LRs tried -> val acc | Selected LR | Mean drop | Within 1 point | Diverged |",
           "|---|---|---|---:|---:|---:|---:|"]
    for o, s in sens.items():
        pts = "  ".join(f"{lr:g}->{'div' if va is None else f'{100 * va:.1f}'}" for lr, va in s["pts"])
        md.append(f"| {LABEL[o]} | {SPEC[o]['group']} | {pts} | {s['best_lr']:g} | "
                  f"{100 * s['mean_drop']:.2f} | {100 * s['within1']:.0f}% | {s['n_div']} |")
    md.append("")
    return sens


def plot_sensitivity(sens, out, title):
    plt = _plt()
    if plt is None or not sens:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160, sharey=True)
    for ax, (g, gname) in zip(axes, GROUP_PANELS):
        for o in opt_sort(sens):
            if not in_panel(o, g):
                continue
            c, ls, lw, ref = style_of(o, g)
            s = sens[o]
            xs = [lr / s["best_lr"] for lr, va in s["pts"] if va is not None]
            ys = [100 * va for lr, va in s["pts"] if va is not None]
            ax.plot(xs, ys, color=c, ls=ls, lw=lw, marker="o", ms=4 if not ref else 3,
                    label=LABEL[o] + (" (reference)" if ref else ""), alpha=0.75 if ref else 1)
        ax.set_xscale("log")
        ax.axvline(1, color="#c3c2b7", lw=0.8)
        ax.set_xlabel("LR / best LR (log)", color=INK2)
        ax.set_title(gname, fontsize=9, color=INK2, loc="left")
        _axes_style(ax)
        ax.legend(fontsize=7, frameon=False, loc="lower center", ncol=2)
    axes[0].set_ylabel("Val accuracy (%)", color=INK2)
    fig.suptitle(f"{title}: LR sensitivity (LR sweep, seed 0)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "lr_sensitivity.png", facecolor=SURF)
    plt.close(fig)


# ================================================================ main runs
def time_to_target(hist, target):
    for h in hist:
        if h.get("test_acc") is not None and h["test_acc"] >= target:
            return h["epoch"]
    return None


def analyze_final(rows, out, md, title):
    final = [r for r in rows if r["phase"] == "final"]
    if not final:
        return None
    by = {}
    for r in final:
        by.setdefault(r["optimizer"], []).append(r)
    opts = opt_sort(by)

    acc = {o: {r["seed"]: (None if r["diverged"] else r["final"]["test_acc"]) for r in by[o]} for o in opts}
    means = {o: S.mean_ci(list(acc[o].values()))[0] for o in opts}
    best_mean = max(m for m in means.values() if m == m)
    target = 0.99 * best_mean  # target: 99% of the best accuracy of the cell

    pv, delta = {}, {}
    if BASE in acc:
        for o in opts:
            if o != BASE:
                delta[o], pv[o], _ = S.paired_t(acc[o], acc[BASE])
    padj = S.holm(pv)

    summary, csv_rows = [], []
    for o in opts:
        rs = by[o]
        m, sd, ci, n = S.mean_ci([acc[o][r["seed"]] for r in rs])
        tb = S.mean_ci([(r["best_val"] or {}).get("test_acc") for r in rs if not r["diverged"]])[0]
        tl = S.mean_ci([r["final"]["train_loss"] for r in rs if not r["diverged"]])[0]
        gap = S.mean_ci([r["final"]["train_acc"] - r["final"]["test_acc"] for r in rs
                         if not r["diverged"] and r["final"]["test_acc"] is not None])[0]
        tts = [time_to_target(r["history"], target) for r in rs if not r["diverged"]]
        reached = [t for t in tts if t is not None]
        tt = S.mean_ci(reached)[0] if reached else float("nan")
        step = S.mean_ci([r.get("step_time_ms") for r in rs])[0]
        mem = S.mean_ci([r.get("peak_mem_mb") for r in rs])[0]
        stmb = S.mean_ci([r.get("opt_state_mb") for r in rs])[0]
        ndiv = sum(r["diverged"] for r in rs)
        row = dict(o=o, lr=rs[0]["lr"], n=len(rs), ndiv=ndiv, m=m, sd=sd, ci=ci, tb=tb, tl=tl, gap=gap,
                   tt=tt, n_reach=len(reached), n_tt=len(tts), step=step, mem=mem, state=stmb,
                   d=delta.get(o), p=pv.get(o), padj=padj.get(o))
        summary.append(row)
        csv_rows.append([o, SPEC[o]["group"], row["lr"], row["n"], ndiv, m, sd, ci, row["d"], row["p"],
                         row["padj"], tb, tl, gap, tt, f"{len(reached)}/{len(tts)}", step, mem, stmb])
    write_csv(out / "final_summary.csv",
              ["optimizer", "group", "lr", "n_runs", "n_diverged", "test_acc_mean", "test_acc_std",
               "test_acc_ci95_halfwidth", "delta_vs_adam", "p_paired_t", "p_holm",
               "test_acc_at_best_val_mean", "final_train_loss_mean", "gen_gap_mean",
               "epochs_to_99pct_best_mean", "runs_reaching_target", "step_time_ms", "peak_mem_mb",
               "opt_state_mb"], csv_rows)
    write_csv(out / "final_runs.csv",
              ["optimizer", "lr", "seed", "diverged", "final_test_acc", "final_val_acc", "best_val_epoch",
               "test_acc_at_best_val", "final_train_loss", "time_min", "step_time_ms", "peak_mem_mb"],
              [[r["optimizer"], r["lr"], r["seed"], r["diverged"], r["final"]["test_acc"],
                r["final"]["val_acc"], (r["best_val"] or {}).get("epoch"),
                (r["best_val"] or {}).get("test_acc"), r["final"]["train_loss"],
                round(r["total_time_s"] / 60, 2), r.get("step_time_ms"), r.get("peak_mem_mb")]
               for o in opts for r in sorted(by[o], key=lambda r: r["seed"])])

    md += [f"### Main runs: final-epoch test acc (%)", "",
           f"Mean ± std over seeds, [95% CI]. Δ and p versus **{LABEL.get(BASE, BASE)}**: paired t-test "
           "by seed (same seed = same initialization and batch order), Holm-adjusted p. "
           f"*Epochs to target*: first epoch reaching 99% of the best accuracy of the cell ({100 * target:.2f}%).", "",
           "| Optimizer | LR | n | Test acc | 95% CI | Δ vs Adam | p (Holm) | Train-test gap | "
           "Epochs to target | ms/step | State MB |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(summary, key=lambda r: -r["m"] if r["m"] == r["m"] else 1e9):
        name = LABEL[r["o"]] + (" ★" if r["m"] == best_mean else "")
        div = f" ({r['ndiv']} div)" if r["ndiv"] else ""
        sig = "" if r["padj"] is None or r["padj"] != r["padj"] else (" *" if r["padj"] < 0.05 else "")
        d = "-" if r["d"] is None else f"{100 * r['d']:+.2f}{sig}"
        tt = (f"not reached (0/{r['n_tt']})" if r["tt"] != r["tt"]
              else f"{r['tt']:.1f} ({r['n_reach']}/{r['n_tt']})")
        step = "-" if r["step"] != r["step"] else f"{r['step']:.1f}"
        md.append(f"| {name} | {r['lr']:g} | {r['n']}{div} | {f2(r['m'])} ± {f2(r['sd'])} | "
                  f"±{f2(r['ci'])} | {d} | {fp(r['padj'])} | {f2(r['gap'])} | {tt} | "
                  f"{step} | {r['state']:.1f} |")
    md += ["", "★ best of the cell; * significantly different from Adam at 0.05 after Holm correction. "
           "No epoch selection on the test set.", ""]
    plot_curves(by, out, title)
    return {o: acc[o] for o in opts}


def plot_curves(by, out, title):
    plt = _plt()
    if plt is None:
        return
    curves = {}
    for o, rs in by.items():
        hs = [r["history"] for r in rs if not r["diverged"]]
        if not hs:
            continue
        n_ep = min(len(h) for h in hs)
        curves[o] = {}
        for key in ("test_acc", "train_loss"):
            ms, ss = [], []
            for e in range(n_ep):
                m, sd, _, _ = S.mean_ci([h[e].get(key) for h in hs])
                ms.append(m); ss.append(sd)
            curves[o][key] = (list(range(1, n_ep + 1)), ms, ss, len(hs))
    for key, ylabel, fname, logy, scale in [("test_acc", "Test accuracy (%)", "curves_test_acc.png", False, 100),
                                             ("train_loss", "Train loss", "curves_train_loss.png", True, 1)]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=160, sharey=True)
        allv = []
        for ax, (g, gname) in zip(axes, GROUP_PANELS):
            for o in opt_sort(curves):
                if not in_panel(o, g):
                    continue
                c, ls, lw, ref = style_of(o, g)
                xs, ms, ss, n = curves[o][key]
                ys = [scale * m for m in ms]
                ax.plot(xs, ys, color=c, ls=ls, lw=lw, alpha=0.8 if ref else 1,
                        label=f"{LABEL[o]}{' (reference)' if ref else ''}")
                if not ref:
                    ax.fill_between(xs, [scale * (m - s) for m, s in zip(ms, ss)],
                                    [scale * (m + s) for m, s in zip(ms, ss)], color=c, alpha=0.10, lw=0)
                start = max(0, int(len(ys) * 0.2))
                allv += [y for y in ys[start:] if y == y]
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("Epoch", color=INK2)
            ax.set_title(gname, fontsize=9, color=INK2, loc="left")
            _axes_style(ax)
            ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower right" if not logy else "upper right")
        if not logy and allv:  # drop the first 20% of epochs from the y-range to show late differences
            lo, hi = min(allv), max(allv)
            pad = max((hi - lo) * 0.05, 0.1)
            axes[0].set_ylim(lo - pad, hi + pad)
        axes[0].set_ylabel(ylabel, color=INK2)
        fig.suptitle(f"{title}: {ylabel.lower()}, mean ± std over seeds", fontsize=11, color=INK, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / fname, facecolor=SURF)
        plt.close(fig)


# ================================================================ per cell
def analyze_cell(cell_dir):
    cell = cell_dir.name
    rows = load(cell_dir)
    if not rows:
        return None, None
    out = cell_dir / "analysis"
    out.mkdir(exist_ok=True)
    cfg = config.CELLS.get(cell, {})
    title = f"Cell {cell}: {rows[0]['dataset']} × {rows[0]['model']}"
    md = [f"## {title}", ""]
    lock = cell_dir / "env_lock.json"
    if lock.exists():
        e = json.loads(lock.read_text(encoding="utf-8"))
        md += [f"Environment: {e['gpu']}, torch {e['torch']}, CUDA {e['cuda']}. "
               f"Sweep/main epochs: {cfg.get('sweep_epochs')}/{cfg.get('epochs')}, batch {cfg.get('batch_size')}.", ""]
    sens = analyze_sweep(rows, out, md)
    plot_sensitivity(sens, out, title)
    acc = analyze_final(rows, out, md, title)
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    n_s = sum(r["phase"] == "sweep" for r in rows)
    n_f = sum(r["phase"] == "final" for r in rows)
    print(f"  Cell {cell}: {n_s} LR-sweep runs, {n_f} main runs -> {out}")
    return "\n".join(md), acc


# ================================================================ pooled over cells
def analyze_ranking(accs, root, md):
    """Pooled ranking: each (cell, seed) is a block; only cells with every optimizer are used."""
    cells = [c for c, a in accs.items() if a and all(o in a for o in ORDER)]
    if not cells:
        return
    blocks, labels = [], []
    for c in cells:
        seeds = sorted(set.intersection(*[set(accs[c][o]) for o in ORDER]))
        for s in seeds:
            vals = [accs[c][o][s] for o in ORDER]
            vals = [-1.0 if v is None else v for v in vals]  # diverged = last rank
            blocks.append(vals); labels.append((c, s))
    if len(blocks) < 2:
        return
    avg, chi2, p, cd = S.friedman_nemenyi(blocks)
    order = sorted(range(len(ORDER)), key=lambda i: avg[i])
    write_csv(root / "ranking.csv", ["optimizer", "group", "avg_rank"],
              [[ORDER[i], SPEC[ORDER[i]]["group"], avg[i]] for i in order])
    md += ["## Pooled ranking", "",
           f"Cells: {', '.join(cells)}: {len(blocks)} blocks (cell × seed), {len(ORDER)} optimizers. "
           f"Friedman χ² = {chi2:.1f}, p = {fp(p)}. Nemenyi CD (α = 0.05) = {cd:.2f}: two optimizers whose "
           "average ranks differ by less than CD are *not* significantly different.", "",
           "| Rank | Optimizer | Group | Avg. rank |", "|---:|---|---|---:|"]
    for j, i in enumerate(order, 1):
        md.append(f"| {j} | {LABEL[ORDER[i]]} | {SPEC[ORDER[i]]['group']} | {avg[i]:.2f} |")
    cls = [avg[i] for i, o in enumerate(ORDER) if SPEC[o]["group"] == "classic"]
    mod = [avg[i] for i, o in enumerate(ORDER) if SPEC[o]["group"] == "modern"]
    if cls and mod:
        md += ["", f"Average rank of the classic group: {sum(cls) / len(cls):.2f}; modern group: {sum(mod) / len(mod):.2f}.", ""]

    plt = _plt()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(ORDER) + 1.4), dpi=160)
    gcol = {"classic": PALETTE[0], "modern": PALETTE[1], "reference": "#8a8984"}
    for y, i in enumerate(reversed(order)):
        o = ORDER[i]
        col = gcol[SPEC[o]["group"]]
        ax.plot([avg[i] - cd / 2, avg[i] + cd / 2], [y, y], color=col, lw=1.2, alpha=0.35,
                solid_capstyle="round")
        ax.plot(avg[i], y, "o", color=col, ms=6, mec=SURF, mew=1.5)
        ax.text(avg[i], y + 0.22, f"{avg[i]:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABEL[ORDER[i]] for i in reversed(order)], fontsize=8, color=INK)
    ax.set_xlabel("Average rank (1 = best, left); bars = ± CD/2 (Nemenyi, α = 0.05)", color=INK2, fontsize=8)
    ax.set_xlim(0.5, len(ORDER) + 0.5)
    ax.set_ylim(-0.6, len(order) - 0.2)
    _axes_style(ax)
    ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], marker="o", ls="", color=gcol[g]) for g in ("classic", "modern", "reference")],
              ["Classic", "Modern", "Reference"], fontsize=7, frameon=False, loc="upper right")
    ax.set_title(f"Pooled ranking ({len(blocks)} cell × seed blocks)", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(root / "ranking.png", facecolor=SURF)
    plt.close(fig)


def analyze_bn(accs, root, md):
    cb, ce = config.BN_ABLATION
    a, b = accs.get(cb), accs.get(ce)
    if not a or not b:
        return
    opts = [o for o in ORDER if o in a and o in b]
    ma = {o: S.mean_ci(list(a[o].values()))[0] for o in opts}
    mb = {o: S.mean_ci(list(b[o].values()))[0] for o in opts}
    ra = dict(zip(opts, S.ranks_desc([ma[o] if ma[o] == ma[o] else -1 for o in opts])))
    rb = dict(zip(opts, S.ranks_desc([mb[o] if mb[o] == mb[o] else -1 for o in opts])))
    write_csv(root / "bn_ablation.csv",
              ["optimizer", "group", f"acc_{cb}_bn", f"acc_{ce}_nobn", "delta_bn_minus_nobn",
               f"rank_{cb}", f"rank_{ce}"],
              [[o, SPEC[o]["group"], ma[o], mb[o], ma[o] - mb[o], ra[o], rb[o]] for o in opts])
    md += ["## Ablation BatchNorm", "",
           f"Cell {cb} (with BN) and cell {ce} (without BN): same dataset and architecture; BN is replaced by conv biases. "
           "Question: does the advantage (or disadvantage) of each optimizer depend on BN? "
           "See the change in ranks.", "",
           "| Optimizer | Group | With BN | Without BN | Δ (BN − no BN) | Rank with BN | Rank without BN |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for o in sorted(opts, key=lambda o: ra[o]):
        md.append(f"| {LABEL[o]} | {SPEC[o]['group']} | {f2(ma[o])} | {f2(mb[o])} | "
                  f"{f2(ma[o] - mb[o])} | {ra[o]:g} | {rb[o]:g} |")
    md.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--cell")
    a = ap.parse_args()
    root = Path(a.out)
    cells = [root / a.cell] if a.cell else sorted(p for p in root.iterdir()
                                                    if p.is_dir() and (p / "runs").exists()
                                                    and not p.name.startswith("_"))
    parts, accs = [], {}
    for c in cells:
        m, acc = analyze_cell(c)
        if m:
            parts.append(m)
            accs[c.name] = acc
    if not parts:
        print("No results yet.")
        return
    if not a.cell:
        md = ["# Adam-family benchmark: summary", ""]
        analyze_ranking(accs, root, md)
        analyze_bn(accs, root, md)
        md += parts
        (root / "summary_all.md").write_text("\n".join(md), encoding="utf-8")
        print(f"  Pooled -> {root / 'summary_all.md'}")


if __name__ == "__main__":
    main()
