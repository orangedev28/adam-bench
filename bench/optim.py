"""Builds the 18 optimizers of the benchmark.

- Adam, AdaMax, NAdam, AMSGrad, AdamW, RAdam, SGD: standard torch.optim versions.
- The other optimizers are re-implemented from the algorithms in the papers and the
    authors' official code (no external packages, so that every machine runs the same code):
    AdamP     — Heo et al., ICLR 2021        github.com/clovaai/AdamP
    AdaBelief — Zhuang et al., NeurIPS 2020  github.com/juntang-zhuang/Adabelief-Optimizer
    Adan      — Xie et al., TPAMI 2024       github.com/sail-sg/Adan
    Lion      — Chen et al., NeurIPS 2023    github.com/google/automl/tree/master/lion
    MARS      — Yuan et al., ICML 2025       github.com/AGI-Arena/MARS  (MARS-AdamW, approximate version)
    AdEMAMix  — Pagliardini et al., ICLR 2025 github.com/apple/ml-ademamix
    C-AdamW   — Liang et al., ICLR 2026      github.com/kyleliang919/C-Optim
    SF-AdamW  — Defazio et al., NeurIPS 2024 github.com/facebookresearch/schedule_free
    SPAM      — Huang et al., ICLR 2025      github.com/TianjinYellow/SPAM-Optimizer
    AdamS     — Zhang et al., EMNLP 2025     github.com/pkuhuzhang/AdamS
    VRAdam    — Vaidhyanathan et al., ICLR 2026  (no public code; follows Algorithm 1)
  `python run_cell.py --check-ref` compares with the original packages if installed (timm, schedulefree,
  adabelief-pytorch...).
"""
import math

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

from . import config


class AdamP(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, delta=0.1, wd_ratio=0.1, nesterov=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        delta=delta, wd_ratio=wd_ratio, nesterov=nesterov)
        super().__init__(params, defaults)

    @staticmethod
    def _channel_view(x):
        return x.view(x.size(0), -1)

    @staticmethod
    def _layer_view(x):
        return x.view(1, -1)

    @staticmethod
    def _cosine_similarity(x, y, eps, view_func):
        return F.cosine_similarity(view_func(x), view_func(y), dim=1, eps=eps).abs_()

    def _projection(self, p, grad, perturb, delta, wd_ratio, eps):
        wd = 1.0
        expand_size = [-1] + [1] * (len(p.shape) - 1)
        for view_func in (self._channel_view, self._layer_view):
            cosine_sim = self._cosine_similarity(grad, p.data, eps, view_func)
            if cosine_sim.max() < delta / math.sqrt(view_func(p.data).size(1)):
                p_n = p.data / view_func(p.data).norm(dim=1).view(expand_size).add_(eps)
                perturb = perturb - p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
                wd = wd_ratio
                return perturb, wd
        return perturb, wd

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                bc1 = 1 - beta1 ** state["step"]
                bc2 = 1 - beta2 ** state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(group["eps"])
                step_size = group["lr"] / bc1

                if group["nesterov"]:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                wd_ratio = 1.0
                if len(p.shape) > 1:
                    perturb, wd_ratio = self._projection(
                        p, grad, perturb, group["delta"], group["wd_ratio"], group["eps"])

                if group["weight_decay"] > 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"] * wd_ratio)
                p.add_(perturb, alpha=-step_size)
        return loss


class AdaBelief(Optimizer):
    """AdaBelief: replaces g^2 by (g - m)^2, the "belief" in the gradient.
    CIFAR setup from the authors' README: eps 1e-8, L2-style weight decay, no rectification."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 decoupled=False):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                      decoupled=decoupled))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd:
                    if group["decoupled"]:
                        p.mul_(1 - lr * wd)
                    else:
                        g = g.add(p, alpha=wd)
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_var"] = torch.zeros_like(p)
                m, s = st["exp_avg"], st["exp_avg_var"]
                st["step"] += 1
                bc1 = 1 - b1 ** st["step"]
                bc2 = 1 - b2 ** st["step"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                r = g - m
                s.mul_(b2).addcmul_(r, r, value=1 - b2).add_(eps)
                denom = (s.sqrt() / math.sqrt(bc2)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bc1)
        return loss


class Adan(Optimizer):
    """Adan: Nesterov momentum for both the first and second moments via the gradient difference (g_t - g_{t-1}).
    betas follow the convention of the official code (0.98, 0.92, 0.99); proximal weight decay."""

    def __init__(self, params, lr=1e-3, betas=(0.98, 0.92, 0.99), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2, b3 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_diff"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                    st["pre_grad"] = g.clone()
                st["step"] += 1
                t = st["step"]
                bc1, bc2, bc3 = 1 - b1 ** t, 1 - b2 ** t, 1 - b3 ** t
                diff = g - st["pre_grad"]
                st["exp_avg"].mul_(b1).add_(g, alpha=1 - b1)
                st["exp_avg_diff"].mul_(b2).add_(diff, alpha=1 - b2)
                upd = g + b2 * diff
                st["exp_avg_sq"].mul_(b3).addcmul_(upd, upd, value=1 - b3)
                denom = (st["exp_avg_sq"].sqrt() / math.sqrt(bc3)).add_(eps)
                p.addcdiv_(st["exp_avg"], denom, value=-lr / bc1)
                p.addcdiv_(st["exp_avg_diff"], denom, value=-lr * b2 / bc2)
                if wd:
                    p.div_(1 + lr * wd)
                st["pre_grad"].copy_(g)
        return loss


class Lion(Optimizer):
    """Lion: update = sign(beta1*m + (1-beta1)*g); a single state variable."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["exp_avg"] = torch.zeros_like(p)
                m = st["exp_avg"]
                if wd:
                    p.mul_(1 - lr * wd)
                c = m.mul(b1).add_(g, alpha=1 - b1)
                p.add_(torch.sign(c), alpha=-lr)
                m.mul_(b2).add_(g, alpha=1 - b2)
        return loss


class MARS(Optimizer):
    """MARS-AdamW (approximate version, one backward pass): moments are computed on the variance-reduced
    gradient c_t = g_t + gamma*beta1/(1-beta1)*(g_t - g_{t-1}), with its norm clipped to <= 1.
    As in the official code, this applies only to parameters with >= 2 dimensions; 1-D parameters (biases, BN)
    use AdamW (here with the same betas/LR for simplicity; stated in the paper)."""

    def __init__(self, params, lr=3e-3, betas=(0.95, 0.99), eps=1e-8, weight_decay=0.0,
                 gamma=0.025):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                      gamma=gamma))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps, wd, gamma = group["lr"], group["eps"], group["weight_decay"], group["gamma"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                    st["last_grad"] = torch.zeros_like(p)
                st["step"] += 1
                bc1 = 1 - b1 ** st["step"]
                bc2 = 1 - b2 ** st["step"]
                if p.dim() >= 2:
                    c = g + (gamma * b1 / (1 - b1)) * (g - st["last_grad"])
                    c = c / torch.clamp(c.norm(), min=1.0)
                else:
                    c = g
                st["exp_avg"].mul_(b1).add_(c, alpha=1 - b1)
                st["exp_avg_sq"].mul_(b2).addcmul_(c, c, value=1 - b2)
                denom = (st["exp_avg_sq"].sqrt() / math.sqrt(bc2)).add_(eps)
                if wd:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(st["exp_avg"], denom, value=-lr / bc1)
                st["last_grad"].copy_(g)
        return loss


def _ademamix_beta3(step, total, beta_start, beta_end):
    """beta3 schedule of AdEMAMix (official code): increases from beta1 to beta3 over T steps."""
    if step >= total:
        return beta_end
    tau = step / total
    lb1, lb3 = math.log(beta_start), math.log(beta_end)
    return min(math.exp(lb1 * lb3 / ((1 - tau) * lb3 + tau * lb1)), beta_end)


class AdEMAMix(Optimizer):
    """AdEMAMix: mixes a fast EMA (beta1) and a slow EMA (beta3) of the gradient.
    alpha and beta3 are warmed up over T steps (T = total number of steps), as the authors recommend."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9999), alpha=5.0, eps=1e-8,
                 weight_decay=0.0, warmup_steps=None):
        super().__init__(params, dict(lr=lr, betas=betas, alpha=alpha, eps=eps,
                                      weight_decay=weight_decay, warmup_steps=warmup_steps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2, b3 = group["betas"]
            lr, eps, wd, alpha = group["lr"], group["eps"], group["weight_decay"], group["alpha"]
            T = group["warmup_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg_fast"] = torch.zeros_like(p)
                    st["exp_avg_slow"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                st["step"] += 1
                t = st["step"]
                bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
                if T:
                    alpha_t = alpha * min(1.0, t / T)
                    b3_t = _ademamix_beta3(t, T, b1, b3)
                else:
                    alpha_t, b3_t = alpha, b3
                m1, m2, nu = st["exp_avg_fast"], st["exp_avg_slow"], st["exp_avg_sq"]
                m1.mul_(b1).add_(g, alpha=1 - b1)
                m2.mul_(b3_t).add_(g, alpha=1 - b3_t)
                nu.mul_(b2).addcmul_(g, g, value=1 - b2)
                denom = (nu.sqrt() / math.sqrt(bc2)).add_(eps)
                upd = (m1 / bc1 + alpha_t * m2) / denom
                if wd:
                    upd.add_(p, alpha=wd)
                p.add_(upd, alpha=-lr)
        return loss


class CAdamW(Optimizer):
    """Cautious AdamW: AdamW, but masks the coordinates whose momentum has the opposite sign to the
    current gradient and rescales the rest (official C-Optim code)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                st["step"] += 1
                bc1 = 1 - b1 ** st["step"]
                bc2 = 1 - b2 ** st["step"]
                if wd:
                    p.mul_(1 - lr * wd)
                m, v = st["exp_avg"], st["exp_avg_sq"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                mask = (m * g > 0).to(g.dtype)
                mask = mask * (mask.numel() / (mask.sum() + 1))
                p.addcdiv_(m * mask, denom, value=-lr / bc1)
        return loss


class AdamWScheduleFree(Optimizer):
    """Schedule-Free AdamW (the authors' reference `AdamWScheduleFree`).
    No external LR schedule; linear warm-up. The model parameters are the point y
    during training; call .eval() to switch to the averaged point x before evaluation,
    and .train() to switch back. BatchNorm statistics must be recomputed at x (train.py does this)."""

    def __init__(self, params, lr=2.5e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 warmup_steps=0, r=0.0, weight_lr_power=2.0, coupled=False):
        # coupled=True (variant "sfadamw-l2"): L2 weight decay is added to the gradient BEFORE
        # updating v and dividing by the denominator, as in Adam. Default False = the authors' version (decoupled).
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                      coupled=coupled, warmup_steps=warmup_steps, r=r,
                                      weight_lr_power=weight_lr_power, k=0, train_mode=True,
                                      weight_sum=0.0, lr_max=-1.0, scheduled_lr=0.0))

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            if group["train_mode"]:
                b1 = group["betas"][0]
                for p in group["params"]:
                    st = self.state[p]
                    if "z" in st:
                        p.lerp_(end=st["z"].to(p.device), weight=1 - 1 / b1)
                group["train_mode"] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            if not group["train_mode"]:
                b1 = group["betas"][0]
                for p in group["params"]:
                    st = self.state[p]
                    if "z" in st:
                        p.lerp_(end=st["z"].to(p.device), weight=1 - b1)
                group["train_mode"] = True

    @torch.no_grad()
    def step(self, closure=None):
        if not self.param_groups[0]["train_mode"]:
            raise RuntimeError("AdamWScheduleFree.step() called in eval mode; call .train() first.")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            eps, wd = group["eps"], group["weight_decay"]
            k, warm = group["k"], group["warmup_steps"]
            sched = (k + 1) / warm if k < warm else 1.0
            # As in the current schedulefree package: the bias correction is inside the denominator,
            # not multiplied into lr (older versions multiplied sqrt(bc2) into lr, which changes the averaging weights).
            bc2 = 1 - b2 ** (k + 1)
            lr = group["lr"] * sched
            group["scheduled_lr"] = lr
            lr_max = group["lr_max"] = max(lr, group["lr_max"])
            weight = ((k + 1) ** group["r"]) * (lr_max ** group["weight_lr_power"])
            group["weight_sum"] = group["weight_sum"] + weight
            ckp1 = weight / group["weight_sum"] if group["weight_sum"] != 0 else 0.0
            for p in group["params"]:
                if p.grad is None:
                    continue
                y, g = p, p.grad
                st = self.state[p]
                if "z" not in st:
                    st["z"] = torch.clone(y)
                    st["exp_avg_sq"] = torch.zeros_like(y)
                z, v = st["z"], st["exp_avg_sq"]
                coupled = group.get("coupled", False)
                if wd and coupled:
                    g = g.add(y, alpha=wd)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                gn = g / v.div(bc2).sqrt_().add_(eps)
                if wd and not coupled:
                    gn.add_(y, alpha=wd)
                y.lerp_(end=z, weight=ckp1)
                y.add_(gn, alpha=lr * (b1 * (1 - ckp1) - 1))
                z.sub_(gn, alpha=lr)
            group["k"] = k + 1
        return loss


class SPAM(Optimizer):
    """SPAM (Huang et al., ICLR 2025): AdamW + momentum reset every DeltaT steps (with a cosine
    warm-up of the step factor over `warmup` steps after each reset) + element-wise clipping of "spiked" gradients
    (g^2 > threshold * v  =>  g = sign(g) * sqrt(threshold * v)). Follows the authors' galore_torch/SPAM.py
    (version without sparse momentum, density = 1). The counters are stored in param_groups[0]
    so that checkpoint/resume preserves them."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 warmup=150, threshold=5000.0, delta_t=500, grad_accu_steps=20):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        g0 = self.param_groups[0]
        g0.update(spam_total=0, spam_cur=warmup + 1)
        self.warmup, self.thres, self.delta_t, self.accu = warmup, threshold, delta_t, grad_accu_steps

    def _scale(self, cur):
        # the authors' CosineDecay(0.99, warmup): dr decays 0.99 -> 0 over `warmup` steps; factor = 1 - dr
        if cur >= self.warmup:
            return 1.0
        return 1.0 - 0.99 * (1 + math.cos(math.pi * cur / (self.warmup + 1))) / 2

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        g0 = self.param_groups[0]
        total, cur = g0["spam_total"], g0["spam_cur"]
        scale = self._scale(cur)
        reset = (total + 1) % self.delta_t == 0
        k = total + 1
        clip = self.thres != 0 and k >= self.accu and (self.delta_t == 0 or k % self.delta_t >= self.accu)
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                if reset:
                    st["exp_avg"].zero_()
                    st["exp_avg_sq"].zero_()
                m, v = st["exp_avg"], st["exp_avg_sq"]
                st["step"] += 1
                if clip:
                    lim = torch.sqrt(v * self.thres)
                    mask = g * g > self.thres * v
                    g = torch.where(mask, torch.sign(g) * lim, g)
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                denom = v.sqrt().add_(eps)
                step_size = lr * math.sqrt(1 - b2 ** st["step"]) / (1 - b1 ** st["step"])
                p.addcdiv_(m, denom, value=-step_size * scale)
                if wd > 0:
                    p.add_(p, alpha=-lr * wd)
        total += 1
        cur += 1
        if total != 0 and (total + 1) % self.delta_t == 0:
            cur = 0
        g0["spam_total"], g0["spam_cur"] = total, cur
        return loss


class AdamS(Optimizer):
    """AdamS (Zhang et al., EMNLP 2025): no second-moment buffer; the denominator is
    sqrt(beta2 * m_hat_{t-1}^2 + (1 - beta2) * g_t^2). Decoupled weight decay.
    Follows the authors' new_optimizers.py (bias correction for m only)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                st["step"] += 1
                bc1 = 1 - b1 ** st["step"]
                if wd:
                    p.mul_(1 - lr * wd)
                m = st["exp_avg"]
                m_hat_prev = m / bc1
                denom = (b2 * m_hat_prev * m_hat_prev + (1 - b2) * g * g).sqrt_().add_(eps)
                m.lerp_(g, 1 - b1)
                p.addcdiv_(m / bc1, denom, value=-lr)
        return loss


class VRAdam(Optimizer):
    """VRAdam (Vaidhyanathan et al., ICLR 2026), Algorithm 1: AdamW with learning rate
        eta_t = lr / (1 + min(beta3 * ||m_t||^2, alpha1)),
    ||m_t|| is the L2 norm of the (uncorrected) momentum over ALL parameters. lr plays the role of alpha0
    (the maximum LR); decoupled weight decay uses eta_t. The authors released no code: the scope of the norm
    (whole model) follows the notation ||v_t||, with v_t the vector of all parameters in Algorithm 1."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 beta3=1.0, alpha1=29.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                      beta3=beta3, alpha1=alpha1))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        sq = None
        live = []
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                st["step"] += 1
                st["exp_avg"].mul_(b1).add_(g, alpha=1 - b1)
                st["exp_avg_sq"].mul_(b2).addcmul_(g, g, value=1 - b2)
                s = st["exp_avg"].float().pow(2).sum()
                sq = s if sq is None else sq + s
                live.append((group, p))
        if sq is None:
            return loss
        for group, p in live:
            b1, b2 = group["betas"]
            st = self.state[p]
            eta = group["lr"] / (1 + torch.clamp(group["beta3"] * sq, max=group["alpha1"]))
            eta = eta.to(p.dtype)
            m_hat = st["exp_avg"] / (1 - b1 ** st["step"])
            v_hat = st["exp_avg_sq"] / (1 - b2 ** st["step"])
            if group["weight_decay"]:
                p.mul_(1 - eta * group["weight_decay"])
            p.sub_(eta * m_hat / (v_hat.sqrt() + group["eps"]))
        return loss


def weight_decay_for(name):
    return config.weight_decay(name)


def build_optimizer(name, params, lr, total_steps=None):
    """total_steps is needed by AdEMAMix (alpha/beta3 warm-up) and Schedule-Free (LR warm-up)."""
    wd = config.weight_decay(name)
    spec = config.OPT_SPECS[name]
    if "base" in spec:
        # Variants (see config.ensure_variant): only the weight-decay form/coefficient of the 4 base optimizers changes.
        base, form = spec["base"], spec["wd"]
        ex = config.OPT_EXTRA.get(base, {})
        kw = dict(lr=lr, betas=ex.get("betas", config.BETAS), eps=config.EPS, weight_decay=wd)
        if base in ("adam", "adamw"):
            return (torch.optim.Adam if form == "coupled" else torch.optim.AdamW)(params, **kw)
        if base == "adabelief":
            return AdaBelief(params, decoupled=(form == "decoupled"), **kw)
        if base == "sfadamw":
            warm = int(round(ex.get("warmup_frac", 0.05) * total_steps)) if total_steps else 0
            return AdamWScheduleFree(params, warmup_steps=warm, coupled=(form == "coupled"), **kw)
        raise ValueError(f"unsupported variant: {name}")
    ex = config.OPT_EXTRA.get(name, {})
    kw = dict(lr=lr, betas=ex.get("betas", config.BETAS), eps=config.EPS, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, **kw)
    if name == "amsgrad":
        return torch.optim.Adam(params, amsgrad=True, **kw)
    if name == "adamax":
        return torch.optim.Adamax(params, **kw)
    if name == "nadam":
        return torch.optim.NAdam(params, **kw)
    if name == "adamw":
        return torch.optim.AdamW(params, **kw)
    if name == "radam":
        return torch.optim.RAdam(params, **kw)
    if name == "adamp":
        return AdamP(params, **kw)
    if name == "adabelief":
        return AdaBelief(params, decoupled=False, **kw)
    if name == "adan":
        return Adan(params, **kw)
    if name == "lion":
        kw.pop("eps")
        return Lion(params, **kw)
    if name == "mars":
        return MARS(params, gamma=ex.get("gamma", 0.025), **kw)
    if name == "ademamix":
        return AdEMAMix(params, alpha=ex.get("alpha", 5.0), warmup_steps=total_steps, **kw)
    if name == "cadamw":
        return CAdamW(params, **kw)
    if name == "sfadamw":
        warm = int(round(ex.get("warmup_frac", 0.05) * total_steps)) if total_steps else 0
        return AdamWScheduleFree(params, warmup_steps=warm, **kw)
    if name == "spam":
        return SPAM(params, **ex.get("spam", {}), **kw)
    if name == "adams":
        return AdamS(params, **kw)
    if name == "vradam":
        return VRAdam(params, beta3=ex.get("beta3", 1.0), alpha1=ex.get("alpha1", 29.0), **kw)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=ex.get("momentum", 0.9), weight_decay=wd)
    raise ValueError(f"unsupported optimizer: {name}")


def is_schedule_free(opt):
    return isinstance(opt, AdamWScheduleFree)


def state_bytes(opt):
    """Optimizer state memory (bytes); hardware-independent."""
    n = 0
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                n += v.numel() * v.element_size()
    return n
