"""Runner for the follow-up experiments X1, E1, E2, E2b, E3, E4 (see bench/experiments.py).

Examples:
  python run_exp.py --selftest                                   # run first on every machine
  python run_exp.py --plan                                       # list the work and estimated hours
  python run_exp.py --exp E1 E3 E2 X1 --cells B --out results_exp --data data
  python run_exp.py --exp E1 E3 E2 X1 --cells D E --out results_exp --data data --worker 0 --workers 2
  python run_exp.py --exp E1 --status --out results_exp

Rules:
    * Base cell B runs on Colab, D and E on Kaggle, i.e. in the environment of the benchmark
        (the runner checks this and stops on a mismatch; --force-env overrides it if you are sure).
    * Several GPUs (Kaggle T4 x2): one process per GPU, same --out, --worker 0/1 and --workers 2.
        Work is pre-assigned by unit (one run, or one whole optimizer in one cell), without overlap.
    * After an interruption, rerun the same command: finished runs are skipped, unfinished ones resume from their checkpoint.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import math
import sys
import time
from pathlib import Path

import torch

from bench import config
from bench import experiments as X
from bench.data import GPUData
from bench.optim import AdamWScheduleFree, build_optimizer
from bench.train import EnvMismatch, env_info, pick_device, run_id, run_one, set_determinism
from run_cell import MAX_EXTEND, Logger, _step_lr, fail, read_json, write_json

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SEC_PER_EPOCH = {"B": 6.1, "C": 21.5, "D": 6.5, "E": 4.9}  # T4 (measured in the benchmark); for estimates only


# ------------------------------------------------------------- work assignment
def select_units(exp, cells):
    us = X.units(exp)
    if cells:
        us = [u for u in us if X.base_cell(u[1]) in cells]
    return us


def assign(us, n):
    """Assign units to n workers: largest first, each to the currently lightest worker (deterministic)."""
    load, out = [0.0] * n, [[] for _ in range(n)]
    for i in sorted(range(len(us)), key=lambda i: (-X.unit_cost(us[i]), i)):
        k = min(range(n), key=lambda k: (load[k], k))
        load[k] += X.unit_cost(us[i])
        out[k].append(us[i])
    return out, load


def res_path(cell_dir, cell, phase, o, lr, s):
    return cell_dir / "runs" / run_id(cell, phase, o, lr, s) / "result.json"


def tune_state_path(cell_dir, o):
    return cell_dir / "tune" / f"{o}.json"


# ------------------------------------------------------------- environment
def check_env(base, env, args, log, checked):
    if base in checked:
        return
    need = X.BASE_ENV[base]
    diff = {k: (v, env.get(k)) for k, v in need.items() if str(env.get(k)) != str(v)}
    if diff and not args.force_env:
        msg = "\n".join(f"  {k}: required {a!r}, this machine {b!r}" for k, (a, b) in diff.items())
        where = "Colab" if base == "B" else "Kaggle"
        fail(f"Cell {base} must run in the environment of the benchmark ({where}):\n{msg}\n"
             f"Install the matching torch version or switch machines. Use --force-env only if you are sure.")
    if diff:
        log(f"WARNING --force-env, cell {base} differs from the base environment: {diff}")
    checked.add(base)


# ------------------------------------------------------------- running
class Runner:
    def __init__(self, args, log, device, env):
        self.args, self.log, self.device, self.env = args, log, device, env
        self.exp = None
        self.exp_dir = None
        self.data, self.data_name = None, None
        self.t0, self.n_done = time.time(), 0
        self.checked = set()

    def set_exp(self, exp):
        self.exp = exp
        self.exp_dir = Path(self.args.out) / exp

    def stop(self):
        a = self.args
        if a.max_runs is not None and self.n_done >= a.max_runs:
            self.log(f"Reached --max-runs={a.max_runs}, stopping.")
            return True
        if a.max_hours is not None and (time.time() - self.t0) / 3600 >= a.max_hours:
            self.log(f"Reached --max-hours={a.max_hours}, stopping.")
            return True
        return False

    def get_data(self, ds):
        if self.data_name != ds:
            self.data = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self.data = GPUData(ds, self.args.data, self.device, config.VAL_SIZE, config.SPLIT_SEED)
            self.data_name = ds
            self.log(f"Data {ds}: train {self.data.n_train}, val {len(self.data.y_val)}, "
                     f"test {len(self.data.y_test)}")
        return self.data

    def run(self, cell, phase, o, lr, s):
        """Run one run. Returns False if the runner must stop (time/run limit)."""
        cell_dir = self.exp_dir / cell
        if res_path(cell_dir, cell, phase, o, lr, s).exists():
            return True
        if self.stop():
            return False
        check_env(X.base_cell(cell), self.env, self.args, self.log, self.checked)
        cfg = X.cell_cfg(cell)
        ep = cfg["sweep_epochs"] if phase == "sweep" else cfg["epochs"]
        self.log(f"[{self.exp} {cell} {phase}] {o} lr={lr:g} seed={s} ({ep} epoch)")
        try:
            st = run_one(cell=cell, phase=phase, cell_cfg=cfg, opt_name=o, lr=lr, seed=s, epochs=ep,
                         data=self.get_data(cfg["dataset"]), device=self.device, out_dir=cell_dir,
                         env=self.env, log=self.log)
        except EnvMismatch as e:
            fail(str(e))
        if st == "done":
            self.n_done += 1
        return True

    def sweep_rows(self, cell, o, grid):
        cell_dir = self.exp_dir / cell
        rows = []
        for lr in grid:
            r = read_json(res_path(cell_dir, cell, "sweep", o, lr, config.SWEEP_SEED))
            va = r["final"]["val_acc"]
            rows.append((lr, -1.0 if r["diverged"] or va is None else va))
        return rows

    def tune(self, cell, o, seeds):
        """LR search (standard grid, extended at the edges as in the benchmark) -> select by final-epoch
        validation accuracy -> main runs for the seeds. State is stored in tune/<opt>.json."""
        cell_dir = self.exp_dir / cell
        sp = tune_state_path(cell_dir, o)
        st = read_json(sp) if sp.exists() else {"ext": []}
        base = config.lr_grid(o)
        while "best_lr" not in st:
            grid = sorted(set(base) | set(st["ext"]))
            for lr in grid:
                if not self.run(cell, "sweep", o, lr, config.SWEEP_SEED):
                    return False
            rows = self.sweep_rows(cell, o, grid)
            best = max(rows, key=lambda t: t[1])[0]  # LRs ascending; ties go to the smaller LR
            n_up = sum(lr > max(base) for lr in grid)
            n_dn = sum(lr < min(base) for lr in grid)
            new = None
            if best == max(grid) and n_up < MAX_EXTEND:
                new = _step_lr(best, up=True)
            elif best == min(grid) and n_dn < MAX_EXTEND:
                new = _step_lr(best, up=False)
            if new is not None and new not in grid:
                st["ext"].append(new)
                self.log(f"  {cell}/{o}: best LR {best:g} at the grid edge -> adding lr={new:g}")
            else:
                st.update(best_lr=best, rows=rows, at_edge=best in (min(grid), max(grid)),
                          criterion="final-epoch val_acc, sweep seed")
                self.log(f"  {cell}/{o}: " + "  ".join(f"{lr:g}:{va:.4f}" for lr, va in rows)
                         + f"  -> selected {best:g}" + ("  (WARNING: still at the edge)" if st["at_edge"] else ""))
            write_json(st, sp)
        for s in seeds:
            if not self.run(cell, "final", o, st["best_lr"], s):
                return False
        return True

    def do_unit(self, u):
        if u[0] == "fixed":
            _, cell, phase, o, lr, s = u
            return self.run(cell, phase, o, lr, s)
        _, cell, o, seeds = u
        return self.tune(cell, o, seeds)


def cmd_run(args, log):
    """Run the experiments in the order given by --exp. Each experiment is split evenly
    across the workers, so the workers finish each experiment at about the same time."""
    n = args.workers or 1
    k = args.worker or 0
    if not 0 <= k < n:
        fail("--worker must be in 0..workers-1")
    plans = []
    for exp in args.exp:
        us = select_units(exp, args.cells)
        if us:
            plans.append((exp, assign(us, n)[0][k], len(us)))
    if not plans:
        fail("No work matches --exp/--cells.")
    device = pick_device(args.device)
    set_determinism()
    env = env_info(device)
    log(f"Worker {k}/{n} on {env['gpu']}, torch {env['torch']}: "
        + ", ".join(f"{e} {len(m)}/{t} units" for e, m, t in plans))
    r = Runner(args, log, device, env)
    for exp, mine, _ in plans:
        r.set_exp(exp)
        for u in mine:
            if not r.do_unit(u):
                return
        log(f"{exp}: worker {k}/{n} finished its share.")
    log(f"Worker {k}/{n}: all done ({r.n_done} new runs).")


def unit_runs(exp_dir, u):
    """(finished runs, expected total runs) of a unit, based on the result files."""
    if u[0] == "fixed":
        _, cell, phase, o, lr, s = u
        return int(res_path(exp_dir / cell, cell, phase, o, lr, s).exists()), 1
    _, cell, o, seeds = u
    cd = exp_dir / cell
    sp = tune_state_path(cd, o)
    st = read_json(sp) if sp.exists() else {"ext": []}
    grid = sorted(set(config.lr_grid(o)) | set(st["ext"]))
    done = sum(res_path(cd, cell, "sweep", o, lr, config.SWEEP_SEED).exists() for lr in grid)
    total = len(grid) + len(seeds)
    if "best_lr" in st:
        done += sum(res_path(cd, cell, "final", o, st["best_lr"], s).exists() for s in seeds)
    return done, total


def cmd_status(args):
    for exp in (args.exp or X.EXPERIMENTS):
        us = select_units(exp, args.cells)
        if not us:
            continue
        ed = Path(args.out) / exp
        d = t = 0
        by_cell = {}
        for u in us:
            a, b = unit_runs(ed, u)
            d, t = d + a, t + b
            c = by_cell.setdefault(u[1], [0, 0])
            c[0] += a; c[1] += b
        print(f"{exp}: {d}/{t} runs done (the total may grow if an LR grid is extended)")
        print("   " + "  ".join(f"{c}: {a}/{b}" for c, (a, b) in sorted(by_cell.items())))


def cmd_plan(args):
    tot = 0.0
    for exp in (args.exp or X.EXPERIMENTS):
        us = select_units(exp, args.cells)
        if args.exp is not None and not us:
            continue
        h = {}
        for u in us:
            b = X.base_cell(u[1])
            h[b] = h.get(b, 0) + X.unit_cost(u) * SEC_PER_EPOCH.get(b, 6.5) / 3600
        tot += sum(h.values())
        print(f"{exp}: {len(us)} units, ~{sum(h.values()):.1f} GPU hours  ("
              + ", ".join(f"cell {b}: {v:.1f} h" for b, v in sorted(h.items())) + ")")
    print(f"Total ~{tot:.1f} GPU hours (excluding LR-grid extensions, if any).")


# ------------------------------------------------------------- selftest
def cmd_selftest(args, log):
    device = pick_device(args.device)
    set_determinism()
    env = env_info(device)
    log(f"SELFTEST run_exp on {env['gpu']}, torch {env['torch']}, CUDA {env['cuda']}")
    fails = []

    # 1. Variants build the right optimizer with the right weight-decay coefficient
    p = [torch.nn.Parameter(torch.randn(3, 3))]
    expect = {"adam-x10": (torch.optim.Adam, 5e-3), "adamw-x0.01": (torch.optim.AdamW, 5e-4),
              "adam-x0.1": (torch.optim.Adam, 5e-5), "adamw-x10": (torch.optim.AdamW, 0.5)}
    for name, (cls, wd) in expect.items():
        config.ensure_variant(name)
        o = build_optimizer(name, p, 1e-3, total_steps=100)
        ok = type(o) is cls and math.isclose(o.param_groups[0]["weight_decay"], wd, rel_tol=1e-9)
        log(f"   [{'ok ' if ok else 'ERR'}] {name}: {type(o).__name__}, wd={o.param_groups[0]['weight_decay']:g}")
        if not ok:
            fails.append(f"{name}: wrong optimizer or wrong weight-decay coefficient")
    config.ensure_variant("adabelief-dw")
    o = build_optimizer("adabelief-dw", p, 1e-3, total_steps=100)
    ok = o.param_groups[0]["decoupled"] is True and math.isclose(o.param_groups[0]["weight_decay"], 0.05)
    log(f"   [{'ok ' if ok else 'ERR'}] adabelief-dw: decoupled={o.param_groups[0]['decoupled']}, "
        f"wd={o.param_groups[0]['weight_decay']:g}")
    if not ok:
        fails.append("adabelief-dw: not decoupled weight decay 0.05")

    # 2. SF-AdamW with coupled L2 == SF-AdamW without weight decay fed the gradient g + lambda*w (at the point y)
    torch.manual_seed(0)
    w0 = torch.randn(64, device=device)
    a = torch.nn.Parameter(w0.clone()); b = torch.nn.Parameter(w0.clone())
    lam = 5e-4
    oa = AdamWScheduleFree([a], lr=1e-2, weight_decay=lam, warmup_steps=5, coupled=True)
    ob = AdamWScheduleFree([b], lr=1e-2, weight_decay=0.0, warmup_steps=5)
    gen = torch.Generator(device="cpu").manual_seed(1)
    for _ in range(20):
        g = torch.randn(64, generator=gen).to(device)
        a.grad = g.clone()
        b.grad = g + lam * b.detach()
        oa.step(); ob.step()
    diff = (a - b).abs().max().item()
    ok = diff < 1e-6
    log(f"   [{'ok ' if ok else 'ERR'}] sfadamw-l2 matches the coupled-L2 definition (max deviation {diff:.1e})")
    if not ok:
        fails.append(f"sfadamw-l2: deviates {diff:.1e} from the definition")
    # The default (decoupled) version is unchanged by the new `coupled` argument
    c = torch.nn.Parameter(w0.clone()); d = torch.nn.Parameter(w0.clone())
    oc = AdamWScheduleFree([c], lr=1e-2, weight_decay=0.05, warmup_steps=5)
    od = AdamWScheduleFree([d], lr=1e-2, weight_decay=0.05, warmup_steps=5, coupled=False)
    gen = torch.Generator(device="cpu").manual_seed(2)
    for _ in range(20):
        g = torch.randn(64, generator=gen).to(device)
        c.grad = g.clone(); d.grad = g.clone()
        oc.step(); od.step()
    ok = torch.equal(c, d) and not torch.equal(c, a)
    log(f"   [{'ok ' if ok else 'ERR'}] default sfadamw is still decoupled weight decay")
    if not ok:
        fails.append("default sfadamw has changed")

    # 3. Real 3-epoch runs on fake data: learns, records weight norms, resume equals an uninterrupted run
    root = Path(args.out) / "_selftest_exp" / str(int(time.time()))
    quiet = lambda m: None
    from run_cell import _same
    # Fake data with 10 classes (100 classes learn too slowly in a few epochs, so the "learns" criterion would misfire).
    # Each case uses its own cell name (T1, T2) so that run_ids do not collide across architectures.
    for tag, model, ds in (("T1", "bncnn", "cifar10"), ("T2", "plaincnn", "cifar10")):
        cfg = dict(config.CELLS["B"], dataset=ds, model=model, batch_size=32)
        data = GPUData(ds, None, device, 200, config.SPLIT_SEED, fake=True)
        for o in ("adabelief-dw", "sfadamw-l2", "adam-x10", "adamw-x0.1"):
            config.ensure_variant(o)
            for lr in (config.lr_grid(o)[2], config.lr_grid(o)[1]):
                kw = dict(cell=tag, phase="st", cell_cfg=cfg, opt_name=o, lr=lr, seed=3, epochs=4,
                          data=data, device=device, env=env, log=quiet)
                run_one(out_dir=root / "straight", **kw)
                rid = run_id(tag, "st", o, lr, 3)
                r = read_json(root / "straight" / "runs" / rid / "result.json")
                h = r["history"]
                learned = not r["diverged"] and h[-1]["train_loss"] < 0.95 * h[0]["train_loss"]
                if learned:
                    break
            st = run_one(out_dir=root / "resumed", interrupt_after=1, **kw)
            run_one(out_dir=root / "resumed", **kw)
            h2 = read_json(root / "resumed" / "runs" / rid / "result.json")["history"]
            # every epoch must carry the norms and the update measurements (the LR at the end of the
            # cosine schedule is 0, so eta_eff_mean > 0 is not checked)
            norms_ok = all(
                len(e.get("w_norms", [])) > 0
                and e.get("eta_eff_mean") is not None
                and len(e.get("rel_upd", [])) == len(e["w_norms"])
                and len(e.get("ang_upd", [])) == len(e["w_norms"])
                and len(e.get("rho_step", [])) == len(e["w_norms"])
                and all(math.isfinite(x) and x > 0 for k in ("rel_upd", "ang_upd", "rho_step") for x in e[k])
                for e in h
            )
            upd_same = all(
                e1.get(k) == e2.get(k)
                for e1, e2 in zip(h, h2)
                for k in ("w_norms", "rel_upd", "ang_upd", "rho_step")
            )
            same = st == "interrupted" and _same(h, h2) and upd_same
            ok = learned and norms_ok and same and r["wd_form"] == config.OPT_SPECS[o]["wd"]
            log(f"   [{'ok ' if ok else 'ERR'}] {ds}/{model}/{o} lr={lr:g}: loss {h[0]['train_loss']:.3f}->"
                f"{h[-1]['train_loss']:.3f}, {len(h[-1]['w_norms'])} conv layers, "
                f"||w|| {h[0]['w_norms'][0]:.2f}->{h[-1]['w_norms'][0]:.2f}, "
                f"rel/rho {h[0]['rel_upd'][0]/h[0]['rho_step'][0]:.2f}, "
                f"resume {'same' if same else 'DIFFERENT'}")
            if not ok:
                fails.append(f"{ds}/{model}/{o}: learned={learned} norms={norms_ok} resume={same}")

    # 4. Work lists can be built
    for exp in X.EXPERIMENTS:
        log(f"   {exp}: {len(X.units(exp))} units")
    if fails:
        log("SELFTEST FAILED - errors:")
        for f in fails:
            log(f"   - {f}")
    else:
        log("SELFTEST PASSED")
    log(f"(test directory: {root}, can be deleted)")
    sys.exit(1 if fails else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", nargs="+", choices=X.EXPERIMENTS,
                    help="one or more experiments, run in order (e.g. --exp E1 E3 E2 X1)")
    ap.add_argument("--cells", nargs="+", choices=["B", "C", "D", "E"], help="run only these base cells")
    ap.add_argument("--out", default="results_exp")
    ap.add_argument("--data", default="data")
    ap.add_argument("--worker", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--max-runs", type=int)
    ap.add_argument("--max-hours", type=float)
    ap.add_argument("--device")
    ap.add_argument("--force-env", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--log", help="log file (default <out>/<exp>/log_w<worker>.txt)")
    args = ap.parse_args()
    if args.plan:
        return cmd_plan(args)
    if args.status:
        return cmd_status(args)
    if args.selftest:
        return cmd_selftest(args, Logger(Path(args.out) / "selftest_exp_log.txt"))
    if not args.exp:
        ap.error("--exp is required (or --plan / --status / --selftest)")
    log_path = args.log or Path(args.out) / f"log_w{args.worker or 0}.txt"
    cmd_run(args, Logger(log_path, tag=f"[w{args.worker or 0}]"))


if __name__ == "__main__":
    main()
