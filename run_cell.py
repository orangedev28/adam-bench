"""Runner for the Adam-family benchmark. Each cell (A-E) runs entirely on one GPU type.

Examples:
  python run_cell.py --selftest                       # run first on every machine
  python run_cell.py --cell C --smoke --data data     # try real data + estimate the runtime
  python run_cell.py --cell C --phase all --out results --data data
  python run_cell.py --cell C --phase status --out results

Several GPUs on the SAME machine (e.g. Kaggle T4 x2): one process per GPU,
same --out, different --worker. The processes claim unfinished runs (claim files) without overlap:
  CUDA_VISIBLE_DEVICES=0 python run_cell.py --cell C --phase all --out results --worker 0
  CUDA_VISIBLE_DEVICES=1 python run_cell.py --cell C --phase all --out results --worker 1
"""
import contextlib
import os

# Must be set before importing torch / initializing CUDA so that cuBLAS is deterministic.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from bench import config
from bench.data import N_TRAIN_FULL, NUM_CLASSES, GPUData
from bench.train import EnvMismatch, env_info, pick_device, run_id, run_one, set_determinism

try:
    import fcntl  # Linux/macOS; unavailable on Windows -> no multi-worker mode
except ImportError:
    fcntl = None


# The Windows console is not UTF-8 by default -> non-ASCII output may fail.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ---------------------------------------------------------------- utilities
def fail(msg):
    """Configuration/environment error: exit code 2 so that auto-restart scripts do NOT retry."""
    print(msg, file=sys.stderr, flush=True)
    sys.exit(2)


class Logger:
    def __init__(self, path=None, tag=""):
        self.path = Path(path) if path else None
        self.tag = tag
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg):
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}]{self.tag} {msg}"
        print(line, flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, p):
    """Write a temporary file and replace: other processes never read a partially written file."""
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)


# ------------------------------------------- multi-worker mode (same machine)
@contextlib.contextmanager
def cell_lock(cell_dir):
    """Exclusive lock for read-modify-write steps on the shared files of a cell (sweep_ext.json, best_lr.json,
    env_lock.json). Without fcntl (Windows) there is no lock; single-process mode is still correct."""
    if fcntl is None:
        yield
        return
    cell_dir.mkdir(parents=True, exist_ok=True)
    with open(cell_dir / ".lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


_HOST = __import__("platform").node() or "host"


def claim_run(claim_dir, rid, worker):
    """Claim the run `rid`. True if the claim succeeds. Claims of dead processes
    (killed, timed out) or of another machine/session (results restored from an earlier session) are
    reclaimed so that another worker can continue from the checkpoint."""
    claim_dir.mkdir(parents=True, exist_ok=True)
    path = claim_dir / f"{rid}.claim"
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                parts = path.read_text().split()
                pid, host = int(parts[0]), parts[2]
            except (OSError, ValueError, IndexError):
                return False          # file is being created by another worker
            if host == _HOST and (pid == os.getpid() or _pid_alive(pid)):
                return False
            with contextlib.suppress(FileNotFoundError):
                path.unlink()         # orphaned claim -> reclaim and retry
            continue
        with os.fdopen(fd, "w") as f:
            f.write(f"{os.getpid()} worker{worker} {_HOST}\n")
        return True
    return False


def release_run(claim_dir, rid):
    with contextlib.suppress(FileNotFoundError):
        (claim_dir / f"{rid}.claim").unlink()


def sweep_complete(cell, cell_dir, opts=None):
    return all(sweep_scores(cell, cell_dir, o) is not None for o in (opts or config.OPTIMIZERS))


def pending_opts(cell_dir):
    """Optimizers without a selected LR. Cell without selection: all. Cell with a selection (best_lr.json) to
    which the config later added optimizers: only the new ones; the LRs of the old ones are kept."""
    path = cell_dir / "best_lr.json"
    if not path.exists():
        return list(config.OPTIMIZERS)
    best = read_json(path)["best_lr"]
    return [o for o in config.OPTIMIZERS if o not in best]


# ---------------------------------------------------------- run planning
MAX_EXTEND = 2  # maximum number of LR-grid extensions per side


def _nice(x):
    """Round to the form 1e k or 3e k."""
    k = math.floor(math.log10(x) + 1e-9)
    m = x / 10 ** k
    m = 1 if m < 2 else (3 if m < 6.5 else 10)
    return float(f"{m * 10 ** k:.6g}")


def _step_lr(lr, up):
    k = math.floor(math.log10(lr) + 1e-9)
    m = round(lr / 10 ** k)
    if up:
        return _nice(3 * 10 ** k) if m == 1 else _nice(10 ** (k + 1))
    return _nice(10 ** k) if m == 3 else _nice(3 * 10 ** (k - 1))


def grid_for(cell_dir, o):
    """LR grid of optimizer o in this cell = base grid + extensions (if any)."""
    ext_path = cell_dir / "sweep_ext.json"
    ext = read_json(ext_path).get(o, []) if ext_path.exists() else []
    return sorted(set(config.lr_grid(o)) | set(ext))


def sweep_plan(cell, cell_dir=None, opts=None):
    ep = config.CELLS[cell]["sweep_epochs"]
    return [("sweep", o, lr, config.SWEEP_SEED, ep)
            for o in (opts or config.OPTIMIZERS)
            for lr in (grid_for(cell_dir, o) if cell_dir else config.lr_grid(o))]


def final_plan(cell, best_lr):
    ep = config.CELLS[cell]["epochs"]
    return [("final", o, best_lr[o], s, ep)
            for s in config.FINAL_SEEDS for o in config.OPTIMIZERS if o in best_lr]


def result_path(cell_dir, cell, phase, o, lr, s):
    return cell_dir / "runs" / run_id(cell, phase, o, lr, s) / "result.json"


def sweep_scores(cell, cell_dir, o):
    """[(lr, val_acc)] of the finished sweep runs; diverged = -1. None if runs are missing."""
    rows = []
    for lr in grid_for(cell_dir, o):
        rp = result_path(cell_dir, cell, "sweep", o, lr, config.SWEEP_SEED)
        if not rp.exists():
            return None
        r = read_json(rp)
        va = r["final"]["val_acc"]
        rows.append((lr, -1.0 if r["diverged"] or va is None else va))
    return rows


def extend_grids(cell, cell_dir, log, opts=None):
    """If the best LR of an optimizer lies at the edge of its grid, add one LR beyond the edge
    (at most MAX_EXTEND times per side). Returns True if new runs were added."""
    ext_path = cell_dir / "sweep_ext.json"
    ext = read_json(ext_path) if ext_path.exists() else {}
    added = False
    for o in (opts or config.OPTIMIZERS):
        rows = sweep_scores(cell, cell_dir, o)
        if rows is None:
            continue
        lrs = [lr for lr, _ in rows]
        best = max(rows, key=lambda t: t[1])[0]
        base = config.lr_grid(o)
        n_up = sum(lr > max(base) for lr in lrs)
        n_dn = sum(lr < min(base) for lr in lrs)
        new = None
        if best == max(lrs) and n_up < MAX_EXTEND:
            new = _step_lr(best, up=True)
        elif best == min(lrs) and n_dn < MAX_EXTEND:
            new = _step_lr(best, up=False)
        if new is not None and new not in lrs:
            ext.setdefault(o, []).append(new)
            added = True
            log(f"  {o}: best LR {best:g} at the grid edge -> adding lr={new:g}")
    if added:
        write_json(ext, ext_path)
    return added


def select_best_lr(cell, cell_dir, log):
    """Select the LR by validation accuracy at the last epoch of the sweep phase. Writes best_lr.json,
    which stays fixed from then on (no re-selection)."""
    path = cell_dir / "best_lr.json"
    old = read_json(path) if path.exists() else None
    todo = pending_opts(cell_dir)
    if old is not None and not todo:
        return old["best_lr"]
    best, edge, missing = {}, [], []
    for o in todo:
        rows = sweep_scores(cell, cell_dir, o)
        if rows is None:
            missing.append(o)
            continue
        best[o] = max(rows, key=lambda t: t[1])[0]  # LRs ascending; ties go to the smaller LR
        log(f"  {o:9s} " + "  ".join(f"{lr:g}:{va:.4f}" for lr, va in rows) + f"  -> selected {best[o]:g}")
        lrs = [lr for lr, _ in rows]
        if best[o] in (min(lrs), max(lrs)):
            edge.append(o)
            log(f"  WARNING {o}: best LR still at the grid edge after the maximum number of extensions.")
    if missing:
        fail(f"LR sweep not finished for: {missing}")
    if old is not None:
        # Cell with LRs already fixed: only add new optimizers, never touch the old LRs.
        old["best_lr"].update(best)
        old["lr_at_grid_edge"] = old.get("lr_at_grid_edge", []) + edge
        old.setdefault("added", []).append({"opts": todo, "selected_at": datetime.now().isoformat()})
        write_json(old, path)
        return old["best_lr"]
    write_json({"best_lr": best, "lr_at_grid_edge": edge, "selected_at": datetime.now().isoformat(),
                "criterion": "final-epoch val_acc, sweep seed"}, path)
    return best


# ---------------------------------------------------- environment lock of a cell
def check_env(cell, cell_dir, env, args, log):
    need = config.CELLS[cell]["require_gpu"]
    if not args.any_gpu and need.lower() not in env["gpu"].lower():
        fail(
            f"Cell {cell} only runs on a GPU containing '{need}'; this machine has '{env['gpu']}'. Stopping to keep the cell homogeneous.\n"
            f"(Colab: Runtime > Disconnect and delete runtime then reconnect to get a T4.)")
    lock = cell_dir / "env_lock.json"
    keys = ("gpu", "torch", "cuda", "cudnn")
    if lock.exists():
        old = read_json(lock)
        diff = {k: (old.get(k), env.get(k)) for k in keys if old.get(k) != env.get(k)}
        if diff and not args.force_env:
            msg = "\n".join(f"  {k}: locked={a!r}  current={b!r}" for k, (a, b) in diff.items())
            fail(
                f"Environment differs from the first run of cell {cell}:\n{msg}\n"
                f"Install the versions listed in {lock} (see README). Use --force-env only if you are sure.")
        if diff:
            log(f"WARNING --force-env, environment differs from the lock: {diff}")
    else:
        write_json(env, lock)
        log(f"Created the environment lock of cell {cell}: {env['gpu']} / torch {env['torch']}")


# --------------------------------------------------------------- commands
def cmd_status(cell, cell_dir):
    def count(plan):
        done = part = 0
        for ph, o, lr, s, _ in plan:
            d = cell_dir / "runs" / run_id(cell, ph, o, lr, s)
            if (d / "result.json").exists():
                done += 1
            elif (d / "ckpt.pt").exists():
                part += 1
        return done, part, len(plan) - done - part

    d, p, w = count(sweep_plan(cell, cell_dir))
    print(f"Cell {cell} - LR sweep: done {d}, partial {p}, pending {w} / {d + p + w}")
    if (cell_dir / "best_lr.json").exists():
        best = read_json(cell_dir / "best_lr.json")["best_lr"]
        d, p, w = count(final_plan(cell, best))
        print(f"Cell {cell} - main runs: done {d}, partial {p}, pending {w} / {d + p + w}")
        print("Selected LRs: " + ", ".join(f"{o}={lr:g}" for o, lr in best.items()))
    else:
        print(f"Cell {cell} - main runs: none yet (LR sweep must finish first)")
    if (cell_dir / "env_lock.json").exists():
        e = read_json(cell_dir / "env_lock.json")
        print(f"Environment lock: {e['gpu']} / torch {e['torch']} / CUDA {e['cuda']}")


def cmd_run(args, log):
    cell = args.cell
    cfg = config.CELLS[cell]
    cell_dir = Path(args.out) / cell
    multi = args.worker is not None
    if multi and fcntl is None:
        fail("--worker needs fcntl (Linux/macOS). On Windows, run a single process.")
    claim_dir = cell_dir / "_claims"
    device = pick_device(args.device)
    set_determinism()
    env = env_info(device)
    log(f"Cell {cell} ({cfg['dataset']} x {cfg['model']}) on {env['gpu']}, torch {env['torch']}"
        + (f", worker {args.worker} (pid {os.getpid()})" if multi else ""))
    with cell_lock(cell_dir):
        check_env(cell, cell_dir, env, args, log)

    data = GPUData(cfg["dataset"], args.data, device, config.VAL_SIZE, config.SPLIT_SEED)
    log(f"Data: train {data.n_train}, val {len(data.y_val)}, test {len(data.y_test)}")

    phases = ["sweep", "final"] if args.phase == "all" else [args.phase]
    t_start = time.time()
    n_done = 0

    def run_plan(ph, plan):
        """Run the unfinished runs. Returns False if stopped by --max-runs/--max-hours."""
        nonlocal n_done
        total = len(plan)
        for i, (_, o, lr, s, ep) in enumerate(plan, 1):
            if args.max_runs is not None and n_done >= args.max_runs:
                log(f"Reached --max-runs={args.max_runs}, stopping."); return False
            if args.max_hours is not None and (time.time() - t_start) / 3600 >= args.max_hours:
                log(f"Reached --max-hours={args.max_hours}, stopping."); return False
            if result_path(cell_dir, cell, ph, o, lr, s).exists():
                continue
            rid = run_id(cell, ph, o, lr, s)
            if multi:
                if not claim_run(claim_dir, rid, args.worker):
                    continue              # another worker is running this run
                if result_path(cell_dir, cell, ph, o, lr, s).exists():
                    release_run(claim_dir, rid)   # finished just before the claim
                    continue
            log(f"[{ph} {i}/{total}] {o} lr={lr:g} seed={s} ({ep} epoch)")
            try:
                status = run_one(cell=cell, phase=ph, cell_cfg=cfg, opt_name=o, lr=lr, seed=s,
                                 epochs=ep, data=data, device=device, out_dir=cell_dir,
                                 env=env, log=log)
            except EnvMismatch as e:
                fail(str(e))
            finally:
                if multi:
                    release_run(claim_dir, rid)
            if status == "done":
                n_done += 1
        return True

    def over_time():
        return args.max_hours is not None and (time.time() - t_start) / 3600 >= args.max_hours

    for ph in phases:
        if ph == "sweep":
            todo = pending_opts(cell_dir)
            if not todo:
                log("LRs already selected (best_lr.json) for every optimizer; skipping the sweep phase.")
                continue
            if len(todo) < len(config.OPTIMIZERS):
                log(f"LRs already fixed for this cell; sweeping only the new optimizers: {todo}")
            waiting = False
            while True:
                if not run_plan("sweep", sweep_plan(cell, cell_dir, todo)):
                    return
                with cell_lock(cell_dir):
                    added = extend_grids(cell, cell_dir, log, todo)
                if added:
                    waiting = False
                    continue
                if not multi or sweep_complete(cell, cell_dir, todo):
                    break
                # Several workers: the remaining sweep runs belong to other workers. Wait so that the grid can be
                # extended / the LR selected when they finish (only happens at the end of the sweep phase).
                if over_time():
                    log(f"Reached --max-hours={args.max_hours}, stopping."); return
                if not waiting:
                    log("No sweep runs left to claim; waiting for other workers to finish theirs...")
                    waiting = True
                time.sleep(args.poll)
        else:
            with cell_lock(cell_dir):
                log("Selecting LRs from the sweep:")
                best = select_best_lr(cell, cell_dir, log)
            if not run_plan("final", final_plan(cell, best)):
                return
    log(f"Cell {cell}: finished phases {phases}.")


def cmd_smoke(args, log):
    """Real data, 2048 training images, 2 epochs, 1 run per optimizer. Estimates the runtime of the whole cell."""
    cell = args.cell
    cfg = config.CELLS[cell]
    device = pick_device(args.device)
    set_determinism()
    env = env_info(device)
    log(f"SMOKE cell {cell} on {env['gpu']}, torch {env['torch']}")
    limit = 2048
    data = GPUData(cfg["dataset"], args.data, device, config.VAL_SIZE, config.SPLIT_SEED,
                   train_limit=limit)
    n_full = N_TRAIN_FULL[cfg["dataset"]] - config.VAL_SIZE
    out = Path(args.out) / "_smoke" / f"{cell}_{int(time.time())}"
    t_train, t_eval = [], []
    for o in config.OPTIMIZERS:
        run_one(cell=cell, phase="smoke", cell_cfg=cfg, opt_name=o, lr=1e-3, seed=0,
                epochs=2, data=data, device=device, out_dir=out, env=env, log=log)
        r = read_json(out / "runs" / run_id(cell, "smoke", o, 1e-3, 0) / "result.json")
        h = r["history"][-1]  # epoch 2 (warmed up)
        t_train.append(h["train_time_s"])
        t_eval.append(h["epoch_time_s"] - h["train_time_s"])
    med = lambda v: sorted(v)[len(v) // 2]
    # full epoch = training (scaled by the number of images) + val/test evaluation (fixed)
    t_epoch = med(t_train) * n_full / limit + med(t_eval)
    runs_sweep = sum(len(config.lr_grid(o)) for o in config.OPTIMIZERS)
    runs_final = len(config.OPTIMIZERS) * len(config.FINAL_SEEDS)
    h_sweep = runs_sweep * cfg["sweep_epochs"] * t_epoch / 3600
    h_final = runs_final * cfg["epochs"] * t_epoch / 3600
    log(f"Training on 2048 images ~{med(t_train):.2f}s, eval ~{med(t_eval):.2f}s -> estimated full epoch ~{t_epoch:.0f}s")
    log(f"Estimate for cell {cell}: LR sweep ~{h_sweep:.1f} h, main runs ~{h_final:.1f} h, total ~{h_sweep + h_final:.1f} h")
    log(f"SMOKE OK - test results are in {out} (can be deleted).")


def _same(h1, h2):
    keys = ("train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc")
    return len(h1) == len(h2) and all(a.get(k) == b.get(k) for a, b in zip(h1, h2) for k in keys)


def cmd_selftest(args, log):
    """Checks on fake data (no network needed):
        1. all 18 optimizers x 4 architecture/dataset pairs run and LEARN (training loss drops >= 5%);
        2. an interrupted and resumed run equals an uninterrupted one (optimizers with complex state);
        3. re-running with the same seed gives identical results (deterministic)."""
    device = pick_device(args.device)
    set_determinism()
    env = env_info(device)
    log(f"SELFTEST on {env['gpu']}, torch {env['torch']}, CUDA {env['cuda']}")
    root = Path(args.out) / "_selftest" / str(int(time.time()))
    quiet = lambda m: None
    fails = []

    # The fake data has only 1000 training images: batch size 32 gives enough steps (~100) to learn.
    # The "learns" criterion uses the training loss. Validation accuracy is informational only: after ~100
    # steps the BatchNorm running statistics have not converged, so evaluation can still be at chance level
    # (with real data an epoch has ~350 steps, so this does not occur).
    cases = [("A", "fmnist", "bncnn", 4), ("D", "cifar100", "bncnn", 4),
             ("E", "cifar10", "plaincnn", 4), ("C", "cifar10", "resnet18", 3)]
    for cell, ds, model, ep in cases:
        cfg = dict(config.CELLS[cell], dataset=ds, model=model, batch_size=32)
        data = GPUData(ds, None, device, 200, config.SPLIT_SEED, fake=True)
        log(f"-- {ds} x {model}: {len(config.OPTIMIZERS)} optimizer, {ep} epoch, batch 32")
        for o in config.OPTIMIZERS:
            # try the 3rd LR of the grid; if it diverges or does not learn, try the 2nd
            for lr in (config.lr_grid(o)[2], config.lr_grid(o)[1]):
                run_one(cell=cell, phase="st", cell_cfg=cfg, opt_name=o, lr=lr, seed=0, epochs=ep,
                        data=data, device=device, out_dir=root / "all", env=env, log=quiet)
                r = read_json(root / "all" / "runs" / run_id(cell, "st", o, lr, 0) / "result.json")
                h = r["history"]
                learned = (not r["diverged"] and math.isfinite(h[-1]["train_loss"])
                           and h[-1]["train_loss"] < 0.95 * h[0]["train_loss"])
                if learned:
                    break
            tag = "ok " if learned else "ERR"
            if not learned:
                fails.append(f"{ds}/{model}/{o}: training loss did not decrease")
            va = r["final"]["val_acc"]
            log(f"   [{tag}] {o:9s} lr={lr:<7g} loss {h[0]['train_loss']:.3f}->{h[-1]['train_loss']:.3f} "
                f"train_acc {h[-1]['train_acc']:.2f} val {va if va is None else round(va, 2)}  "
                f"{r['step_time_ms']:.1f} ms/step  state {r['opt_state_mb']:.1f} MB")

    # Resume + determinism for optimizers with complex state
    checks = [("A", "fmnist", "bncnn", ["adamp", "adan", "mars", "ademamix", "lion", "sfadamw", "sgd", "spam", "adams", "vradam"]),
              ("C", "cifar10", "resnet18", ["adamp", "sfadamw"])]
    for cell, ds, model, opts in checks:
        cfg = dict(config.CELLS[cell], dataset=ds, model=model)
        data = GPUData(ds, None, device, 200, config.SPLIT_SEED, fake=True)
        log(f"-- {ds} x {model}: resume (interrupted after epoch 1/3) and repeat with the same seed")
        for o in opts:
            lr = config.lr_grid(o)[1]
            kw = dict(cell=cell, phase="st", cell_cfg=cfg, opt_name=o, lr=lr, seed=7, epochs=3,
                      data=data, device=device, env=env, log=quiet)
            run_one(out_dir=root / "straight", **kw)
            st = run_one(out_dir=root / "resumed", interrupt_after=1, **kw)
            assert st == "interrupted", st
            run_one(out_dir=root / "resumed", **kw)
            run_one(out_dir=root / "repeat", **kw)
            rid = run_id(cell, "st", o, lr, 7)
            h1, h2, h3 = (read_json(root / d / "runs" / rid / "result.json")["history"]
                          for d in ("straight", "resumed", "repeat"))
            r_ok, d_ok = _same(h1, h2), _same(h1, h3)
            if not r_ok:
                fails.append(f"{ds}/{model}/{o}: resumed run differs from the uninterrupted run")
            if not d_ok:
                dl = max(abs(a["train_loss"] - b["train_loss"]) for a, b in zip(h1, h3))
                fails.append(f"{ds}/{model}/{o}: repeating the same seed gave different results "
                             f"(max train_loss deviation {dl:.1e})")
            log(f"   {o:9s} resume {'PASS' if r_ok else 'FAIL'}   deterministic {'PASS' if d_ok else 'FAIL'}")

    if fails:
        log("SELFTEST FAILED - errors:")
        for f in fails:
            log(f"   - {f}")
    else:
        log("SELFTEST PASSED")
    log(f"(test directory: {root}, can be deleted)")
    sys.exit(1 if fails else 0)


def cmd_det_check(args, log):
    """Reproducibility diagnosis: runs the same run several times and reports the largest deviation
    between repetitions and any 'nondeterministic' warnings issued by PyTorch."""
    import warnings as W
    device = pick_device(args.device)
    set_determinism()
    env = env_info(device)
    log(f"DET-CHECK on {env['gpu']}, torch {env['torch']}, CUDA {env['cuda']}")
    root = Path(args.out) / "_detcheck" / str(int(time.time()))
    keys = ("train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc")
    cases = [("A", "fmnist", "bncnn", o) for o in ("sgd", "adam", "adamw")] + \
            [("C", "cifar10", "resnet18", "sgd")]
    seen = {}
    for cell, ds, model, o in cases:
        cfg = dict(config.CELLS[cell], dataset=ds, model=model)
        data = GPUData(ds, None, device, 200, config.SPLIT_SEED, fake=True)
        lr = config.lr_grid(o)[1]
        hs = []
        for rep in range(args.det_reps):
            with W.catch_warnings(record=True) as caught:
                W.simplefilter("always")
                run_one(cell=cell, phase="dc", cell_cfg=cfg, opt_name=o, lr=lr, seed=7, epochs=3,
                        data=data, device=device, out_dir=root / f"r{rep}", env=env, log=lambda m: None)
            for w in caught:
                msg = str(w.message)
                if "determinis" in msg.lower():
                    seen[msg.split(".")[0][:160]] = seen.get(msg.split(".")[0][:160], 0) + 1
            hs.append(read_json(root / f"r{rep}" / "runs" / run_id(cell, "dc", o, lr, 7) / "result.json")["history"])
        diffs = {k: max(abs(h[e][k] - hs[0][e][k]) for h in hs[1:] for e in range(len(h))) for k in keys}
        n_same = sum(_same(hs[0], h) for h in hs[1:])
        log(f"  {ds}/{model}/{o:6s} lr={lr:g}: {n_same}/{len(hs) - 1} repetitions identical to the first | "
            f"max deviation train_loss {diffs['train_loss']:.2e}, val_acc {diffs['val_acc']:.2e}, "
            f"test_acc {diffs['test_acc']:.2e}")
    if seen:
        log("'nondeterministic' warnings issued by PyTorch:")
        for m, c in seen.items():
            log(f"  ({c}x) {m}")
    else:
        log("No 'nondeterministic' warnings from PyTorch.")


def cmd_check_ref(args, log):
    """Compare the re-implemented optimizers with the original packages (if installed): 40 steps on the same
    parameters and the same random gradient sequence, reporting the largest relative parameter deviation.
    Only 2-D parameters are used, to avoid differences in how packages treat biases/BN.
    Reference packages: pip install timm schedulefree adabelief-pytorch adamp pytorch-optimizer"""
    from bench import optim as O
    torch.manual_seed(0)
    shapes = [(32, 16), (16, 8), (8, 4, 3, 3)]
    init = [torch.randn(sh) for sh in shapes]
    grads = [[torch.randn(sh) for sh in shapes] for _ in range(40)]
    T = 40

    def run(make):
        ps = [torch.nn.Parameter(t.clone()) for t in init]
        opt = make(ps)
        if hasattr(opt, "train"):  # Schedule-Free: the original package requires .train() before step
            opt.train()
        for gs in grads:
            for p, g in zip(ps, gs):
                p.grad = g.clone()
            opt.step()
        if hasattr(opt, "eval"):
            opt.eval()
        return [p.detach() for p in ps]

    def timm_make(name, **kw):
        from timm.optim import create_optimizer_v2
        return lambda ps: create_optimizer_v2(ps, opt=name, **kw)

    lr, wd = 1e-3, 1e-2
    cases = [
        ("adamp", lambda ps: O.AdamP(ps, lr=lr, weight_decay=wd),
         lambda ps: __import__("adamp").AdamP(ps, lr=lr, weight_decay=wd)),
        ("adabelief", lambda ps: O.AdaBelief(ps, lr=lr, weight_decay=wd),
         lambda ps: __import__("adabelief_pytorch").AdaBelief(
             ps, lr=lr, eps=1e-8, weight_decay=wd, weight_decouple=False, rectify=False,
             fixed_decay=False, amsgrad=False, print_change_log=False)),
        ("adan", lambda ps: O.Adan(ps, lr=lr, weight_decay=wd),
         timm_make("adan", lr=lr, weight_decay=wd, betas=(0.98, 0.92, 0.99))),
        ("lion", lambda ps: O.Lion(ps, lr=lr, weight_decay=wd),
         timm_make("lion", lr=lr, weight_decay=wd, betas=(0.9, 0.99))),
        ("mars", lambda ps: O.MARS(ps, lr=lr, weight_decay=wd),
         timm_make("mars", lr=lr, weight_decay=wd, betas=(0.95, 0.99), gamma=0.025)),
        ("cadamw", lambda ps: O.CAdamW(ps, lr=lr, weight_decay=wd),
         timm_make("cadamw", lr=lr, weight_decay=wd)),
        ("ademamix", lambda ps: O.AdEMAMix(ps, lr=lr, weight_decay=wd, warmup_steps=T),
         lambda ps: __import__("pytorch_optimizer").AdEMAMix(
             ps, lr=lr, betas=(0.9, 0.999, 0.9999), alpha=5.0, T_alpha_beta3=T,
             weight_decay=wd, weight_decouple=True)),
        ("sfadamw", lambda ps: O.AdamWScheduleFree(ps, lr=lr, weight_decay=wd, warmup_steps=5),
         lambda ps: __import__("schedulefree").AdamWScheduleFree(
             ps, lr=lr, weight_decay=wd, warmup_steps=5)),
    ]
    import importlib.util, io, contextlib

    def official(fname, cls):
        spec = importlib.util.spec_from_file_location(fname, Path(__file__).parent / "ref" / f"{fname}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, cls)

    def spam_ref(ps):
        with contextlib.redirect_stdout(io.StringIO()):
            return official("spam_official", "AdamW")(
                ps, lr=lr, eps=1e-8, weight_decay=wd, warmup_steps=5, threshold=2.0, DeltaT=15,
                grad_accu_steps=3, no_deprecation_warning=True)

    class _Quiet:   # the original SPAM prints "Mask Update" at every reset
        def __init__(self, o): self.o = o
        def step(self):
            with contextlib.redirect_stdout(io.StringIO()):
                self.o.step()

    def vradam_loop(ps):
        """VRAdam rewritten line by line from Algorithm 1 (independent of bench/optim.py)."""
        class Loop:
            def __init__(self):
                self.m = [torch.zeros_like(p) for p in ps]
                self.v = [torch.zeros_like(p) for p in ps]
                self.t = 0
            @torch.no_grad()
            def step(self):
                self.t += 1
                b1, b2, eps, b3, a1 = 0.9, 0.999, 1e-8, 1.0, 29.0
                for i, p in enumerate(ps):
                    self.m[i] = b1 * self.m[i] + (1 - b1) * p.grad
                    self.v[i] = b2 * self.v[i] + (1 - b2) * p.grad ** 2
                n2 = sum(float((x ** 2).sum()) for x in self.m)
                eta = lr / (1 + min(b3 * n2, a1))
                for i, p in enumerate(ps):
                    mh = self.m[i] / (1 - b1 ** self.t)
                    vh = self.v[i] / (1 - b2 ** self.t)
                    p.mul_(1 - eta * wd)
                    p.sub_(eta * mh / (vh.sqrt() + eps))
        return Loop()

    cases += [
        ("spam", lambda ps: O.SPAM(ps, lr=lr, weight_decay=wd, warmup=5, threshold=2.0, delta_t=15,
                                   grad_accu_steps=3),
         lambda ps: _Quiet(spam_ref(ps))),
        ("adams", lambda ps: O.AdamS(ps, lr=lr, betas=(0.9, 0.95), weight_decay=wd),
         lambda ps: official("adams_official", "AdamS")(
             [{"params": ps, "weight_decay": wd}], lr=lr, betas=(0.9, 0.95), eps=1e-8)),
        ("vradam", lambda ps: O.VRAdam(ps, lr=lr, weight_decay=wd), vradam_loop),
    ]
    for name, mine, ref in cases:
        try:
            a = run(mine)
        except Exception as e:
            log(f"  {name:9s} ERROR in the re-implementation: {e!r}")
            continue
        try:
            b = run(ref)
        except ImportError as e:
            log(f"  {name:9s} skipped (reference package not installed: {e.name})")
            continue
        except Exception as e:
            log(f"  {name:9s} skipped (the reference package does not accept this argument: {e!r})")
            continue
        rel = max(((x - y).abs().max() / y.abs().max().clamp(min=1e-12)).item() for x, y in zip(a, b))
        moved = max(((x - i).abs().max()).item() for x, i in zip(a, init))
        verdict = "MATCH" if rel < 1e-5 else ("CLOSE" if rel < 1e-3 else "DIFFERS")
        log(f"  {name:9s} {verdict:8s} relative deviation {rel:.2e} (parameters moved {moved:.2e})")
    log("Note: 'DIFFERS' may be due to different defaults in the reference package (betas, weight-decay convention, ...)."
        "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", choices=sorted(config.CELLS))
    ap.add_argument("--phase", choices=["sweep", "final", "all", "status"], default="all")
    ap.add_argument("--out", default="results", help="results directory (Colab: on Google Drive)")
    ap.add_argument("--data", default="data", help="directory holding/downloading the datasets")
    ap.add_argument("--max-runs", type=int, help="stop after N finished runs")
    ap.add_argument("--max-hours", type=float, help="do not start new runs after N hours")
    ap.add_argument("--device", help="force the device: cuda / mps / cpu")
    ap.add_argument("--any-gpu", action="store_true", help="skip the GPU-type check (for testing only)")
    ap.add_argument("--force-env", action="store_true", help="ignore the environment lock (not recommended)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check-ref", action="store_true", help="compare the re-implemented optimizers with the original packages")
    ap.add_argument("--det-check", action="store_true", help="reproducibility diagnosis")
    ap.add_argument("--det-reps", type=int, default=4)
    ap.add_argument("--worker", type=int, help="worker index when several processes (one per GPU) "
                    "run one cell on one machine; see the top of this file")
    ap.add_argument("--poll", type=float, default=30, help="seconds between checks (multi-worker mode)")
    args = ap.parse_args()

    if args.selftest:
        return cmd_selftest(args, Logger())
    if args.det_check:
        return cmd_det_check(args, Logger())
    if args.check_ref:
        return cmd_check_ref(args, Logger())
    if not args.cell:
        ap.error("--cell is required (A-E)")
    if args.smoke:
        return cmd_smoke(args, Logger())
    if args.phase == "status":
        return cmd_status(args.cell, Path(args.out) / args.cell)
    tag = f" [w{args.worker}]" if args.worker is not None else ""
    cmd_run(args, Logger(Path(args.out) / args.cell / "log.txt", tag=tag))


if __name__ == "__main__":
    main()
