"""Run ONE training run (one optimizer, one LR, one seed), with a checkpoint every epoch and resume."""
import contextlib
import json
import math
import os
import platform
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

from . import config
from .data import IN_CHANNELS, NUM_CLASSES
from .models import build_model
from .optim import build_optimizer, is_schedule_free, state_bytes, weight_decay_for

warnings.filterwarnings("ignore", message=".*lr_scheduler.step\\(\\).*")


class EnvMismatch(RuntimeError):
    """The checkpoint was created on a different machine/version."""


# ----------------------------------------------------------------- environment
def pick_device(prefer=None):
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_determinism():
    """Call before any CUDA computation. CUBLAS_WORKSPACE_CONFIG is already set by
    run_cell.py before torch is imported."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Disable TF32 so that every GPU computes plain FP32 for the parts that do not use AMP.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def env_info(device):
    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        gpu = f"Apple MPS ({platform.machine()})"
    else:
        gpu = f"CPU ({platform.processor() or platform.machine()})"
    try:
        import torchvision
        tv = torchvision.__version__
    except Exception:
        tv = None
    return {
        "gpu": gpu,
        "torch": torch.__version__,
        "torchvision": tv,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if device.type == "cuda" else None,
        "python": platform.python_version(),
        "os": platform.platform(),
    }


# -------------------------------------------------------------------- utilities
def run_id(cell, phase, opt, lr, seed):
    return f"{cell}_{phase}_{opt}_lr{lr:g}_s{seed}"


def _atomic_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_json(obj, path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def cross_entropy(logits, y, reduction="mean"):
    """Cross-entropy written with one-hot targets: only element-wise ops + reductions, so the
    backward pass is deterministic on CUDA (nll_loss / gather may use atomicAdd)."""
    logp = F.log_softmax(logits.float(), dim=1)
    nll = -(F.one_hot(y, logits.size(1)).to(logp.dtype) * logp).sum(1)
    return nll.sum() if reduction == "sum" else nll.mean()


def _autocast(use_amp):
    if use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def evaluate(model, data, split, use_amp):
    model.eval()
    loss_sum = torch.zeros((), device=data.device)
    correct = torch.zeros((), device=data.device, dtype=torch.long)
    n = 0
    for x, y in data.eval_batches(split):
        with _autocast(use_amp):
            out = model(x)
        loss_sum += cross_entropy(out, y, reduction="sum")
        correct += (out.argmax(1) == y).sum()
        n += len(y)
    return loss_sum.item() / n, correct.item() / n


# ------------------------------------------------------------------- one run
@torch.no_grad()
def recalibrate_bn(model, data, batch_size, n_batches=50):
    """Recompute the BatchNorm statistics (cumulative average over n_batches fixed training
    batches). Needed for Schedule-Free: the weights used for evaluation (x) differ from those
    used for training (y), so the running statistics accumulated during training do not match."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    for i, (x, _) in enumerate(data.train_batches(batch_size, seed=987654, epoch=0)):
        if i >= n_batches:
            break
        model(x)
    for m, mo in zip(bns, saved):
        m.momentum = mo


@torch.no_grad()
def conv_weight_norms(model):
    """Frobenius norm ||w|| of every 4-D conv weight, in declaration order."""
    return [p.detach().float().norm().item() for p in model.parameters() if p.ndim == 4]


def _sample_steps(steps_per_epoch, n=10):
    """Pick at most n step indices spread evenly over an epoch, at which the actual update is measured."""
    n = min(n, steps_per_epoch)
    if n <= 0:
        return set()
    stride = max(1, steps_per_epoch // n)
    return {min(steps_per_epoch - 1, stride // 2 + i * stride) for i in range(n)}


@torch.no_grad()
def _measure_step_updates(conv_params, w_prev, eta_step, conv_c_sqrt):
    """Return (rel_upd, ang_upd, rho_step) for every conv layer at a step that has just been applied:
    the measured relative update ||w1 - w0|| / ||w0||, the angle between w0 and w1 (radians), and the
    proxy eta * sqrt(C) / ||w0|| at the same step."""
    rels, angs, rhos = [], [], []
    for p0, p1, c_sqrt in zip(w_prev, conv_params, conv_c_sqrt):
        a = p0.double()
        d = p1.detach().double() - a
        na = a.norm()
        if na > 0:
            dw = d.norm()
            ahat = a / na
            d_par = (d * ahat).sum()
            d_perp = (d - d_par * ahat).norm()
            rels.append((dw / na).item())
            angs.append(torch.atan2(d_perp, na + d_par).item())
            rhos.append(eta_step * c_sqrt / na.item())
        else:
            rels.append(0.0)
            angs.append(0.0)
            rhos.append(0.0)
    return rels, angs, rhos


def run_one(*, cell, phase, cell_cfg, opt_name, lr, seed, epochs, data,
            device, out_dir, env, interrupt_after=None, log=print):
    """Returns 'done' | 'skipped' | 'interrupted'.

    interrupt_after: stop after N epochs of this call (used by the selftest to simulate
    an interruption).
    """
    rid = run_id(cell, phase, opt_name, lr, seed)
    run_dir = Path(out_dir) / "runs" / rid
    result_path = run_dir / "result.json"
    ckpt_path = run_dir / "ckpt.pt"
    if result_path.exists():
        return "skipped"
    run_dir.mkdir(parents=True, exist_ok=True)

    use_amp = bool(cell_cfg["amp"]) and device.type == "cuda"
    bs = cell_cfg["batch_size"]
    ds = cell_cfg["dataset"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Same seed -> same initial weights for every optimizer.
    torch.manual_seed(seed)
    model = build_model(cell_cfg["model"], IN_CHANNELS[ds], NUM_CLASSES[ds]).to(device)
    steps_per_epoch = data.steps_per_epoch(bs)
    total_steps = epochs * steps_per_epoch
    opt = build_optimizer(opt_name, model.parameters(), lr, total_steps=total_steps)
    sf = is_schedule_free(opt)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
             if config.schedule(opt_name) == "cosine" else None)
    scaler = _make_scaler(use_amp)

    history, start_epoch, elapsed, peak_mem = [], 0, 0.0, 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck["env"]["gpu"] != env["gpu"] or ck["env"]["torch"] != env["torch"]:
            raise EnvMismatch(
                f"{rid}: checkpoint created on {ck['env']['gpu']} / torch {ck['env']['torch']}, "
                f"but the current machine is {env['gpu']} / torch {env['torch']}. Checkpoints are not resumed across machines.")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if sched is not None:
            sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        history, start_epoch, elapsed = ck["history"], ck["epoch"], ck["elapsed"]
        peak_mem = ck.get("peak_mem", 0)
        log(f"  resume {rid} from epoch {start_epoch}/{epochs}")

    def current_lr():
        g = opt.param_groups[0]
        return g.get("scheduled_lr", g["lr"]) if sf else g["lr"]

    # Measurement of the actual update (read-only; does not change training): ~10 steps per epoch.
    conv_params = [p for p in model.parameters() if p.ndim == 4]
    conv_c_sqrt = [math.sqrt(p.numel()) for p in conv_params]
    sample_steps = _sample_steps(steps_per_epoch, n=10)

    diverged = any(h.get("diverged") for h in history)
    done_this_call = 0
    for epoch in range(start_epoch, epochs):
        if diverged:
            break
        t0 = time.perf_counter()
        model.train()
        if sf:
            opt.train()
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device, dtype=torch.long)
        n_conv = len(conv_params)
        sum_rel = [0.0] * n_conv
        sum_ang = [0.0] * n_conv
        sum_rho = [0.0] * n_conv
        n_meas = 0
        measure_next = False
        for step_idx, (x, y) in enumerate(data.train_batches(bs, seed, epoch)):
            opt.zero_grad(set_to_none=True)
            with _autocast(use_amp):
                out = model(x)
                loss = cross_entropy(out, y)
            scaler.scale(loss).backward()
            want_measure = n_conv > 0 and (measure_next or (step_idx in sample_steps))
            if want_measure:
                w_prev = [p.detach().clone() for p in conv_params]
            old_scale = scaler.get_scale() if use_amp else None
            scaler.step(opt)
            scaler.update()
            # With mixed precision the scaler skips a step when it finds inf/NaN gradients (and lowers
            # the scale); such a step changes no weight, so the measurement moves to the next step.
            step_ran = (not use_amp) or (scaler.get_scale() >= old_scale)
            if want_measure:
                if step_ran:
                    measure_next = False
                    rels, angs, rhos = _measure_step_updates(conv_params, w_prev, current_lr(), conv_c_sqrt)
                    for li in range(n_conv):
                        sum_rel[li] += rels[li]
                        sum_ang[li] += angs[li]
                        sum_rho[li] += rhos[li]
                    n_meas += 1
                else:
                    measure_next = True
                del w_prev
            if sched is not None:
                sched.step()
            loss_sum += loss.detach() * len(y)
            correct += (out.argmax(1) == y).sum()
        train_loss = loss_sum.item() / data.n_train
        train_acc = correct.item() / data.n_train
        t_train = time.perf_counter() - t0

        rec = {"epoch": epoch + 1, "lr_end": current_lr(),
               "train_loss": train_loss, "train_acc": train_acc}
        # Norms of the conv weights (measured at the training point; y for Schedule-Free) and the
        # SGD-style effective learning rate lr/||w||^2 averaged over layers. Read-only; does not affect training.
        norms = conv_weight_norms(model)
        rec["w_norms"] = norms
        rec["eta_eff_mean"] = (sum(current_lr() / n ** 2 for n in norms if n > 0) / len(norms)
                               if norms else None)
        if n_meas > 0:
            rec["rel_upd"] = [v / n_meas for v in sum_rel]
            rec["ang_upd"] = [v / n_meas for v in sum_ang]
            rec["rho_step"] = [v / n_meas for v in sum_rho]
        if not math.isfinite(train_loss):
            rec["diverged"] = True
            diverged = True
            log(f"  {rid}: non-finite loss at epoch {epoch + 1} -> marked as diverged")
        else:
            if sf:
                opt.eval()
                recalibrate_bn(model, data, bs)
            rec["val_loss"], rec["val_acc"] = evaluate(model, data, "val", use_amp)
            rec["test_loss"], rec["test_acc"] = evaluate(model, data, "test", use_amp)
            if sf:
                opt.train()
        rec["train_time_s"] = t_train
        rec["epoch_time_s"] = time.perf_counter() - t0
        elapsed += rec["epoch_time_s"]
        if device.type == "cuda":
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(device))
        history.append(rec)
        if "val_acc" in rec:
            log(f"  {rid} ep {epoch + 1:3d}/{epochs} loss {train_loss:.4f} "
                f"val {rec['val_acc']:.4f} test {rec['test_acc']:.4f} ({rec['epoch_time_s']:.1f}s)")

        _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(),
                      "sched": sched.state_dict() if sched is not None else None,
                      "scaler": scaler.state_dict(),
                      "history": history, "epoch": epoch + 1, "elapsed": elapsed,
                      "peak_mem": peak_mem, "env": env}, ckpt_path)

        done_this_call += 1
        if interrupt_after is not None and done_this_call >= interrupt_after and epoch + 1 < epochs:
            return "interrupted"

    good = [h for h in history if "val_acc" in h]
    best = max(good, key=lambda h: h["val_acc"]) if good else None
    last = history[-1] if history else {}
    train_times = [h["train_time_s"] for h in history[1:]] or [h["train_time_s"] for h in history]
    result = {
        "run_id": rid, "cell": cell, "phase": phase,
        "optimizer": opt_name, "group": config.OPT_SPECS[opt_name]["group"],
        "lr": lr, "seed": seed,
        "weight_decay": weight_decay_for(opt_name),
        "wd_form": config.OPT_SPECS[opt_name]["wd"],
        "schedule": config.schedule(opt_name),
        "dataset": ds, "model": cell_cfg["model"],
        "epochs": epochs, "batch_size": bs, "amp": use_amp,
        "steps_per_epoch": steps_per_epoch,
        "diverged": diverged,
        "final": {k: last.get(k) for k in ("train_loss", "train_acc", "val_loss", "val_acc",
                                            "test_loss", "test_acc")},
        "best_val": ({"epoch": best["epoch"], "val_acc": best["val_acc"],
                      "test_acc": best["test_acc"]} if best else None),
        "total_time_s": elapsed,
        # time per training step (first epoch skipped because of warm-up effects)
        "step_time_ms": (1000 * sum(train_times) / len(train_times) / steps_per_epoch
                         if train_times else None),
        "peak_mem_mb": peak_mem / 2 ** 20 if peak_mem else None,
        "opt_state_mb": state_bytes(opt) / 2 ** 20,
        "n_params": sum(p.numel() for p in model.parameters()),
        "env": env,
        "history": history,
    }
    _atomic_json(result, result_path)
    try:
        ckpt_path.unlink()
    except FileNotFoundError:
        pass
    return "done"
