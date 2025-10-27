# optimizers/dog_coordinate.py
"""
Coordinate-wise DoG (Algorithm 1, without projection)

w_{t+1} = w_t - eta_t * g_t / sqrt(v_t)
  where:
    v_{-1} = eps^2 * 1
    eta_{-1} = r_eps  (default r_eps = alpha * (1 + ||w0||_inf))
    v_t = v_{t-1} + g_t ⊙ g_t
    eta_t = max(eta_{t-1}, ||w_t - w0||_inf)

Notes:
- Per-parameter state keeps: v (accumulated g^2) and w0 (initial snapshot).
- A single global eta_t is kept in the optimizer's state (like the paper).
- Optional weight_decay (L2) is applied directly to gradients before updates.
"""

from typing import Optional
import torch
from torch.optim import Optimizer


class DoGCoordinate(Optimizer):
    def __init__(
        self,
        params,
        eps: float = 1e-8,
        alpha: float = 1e-6,
        init_eta: Optional[float] = None,
        weight_decay: float = 0.0,
    ):
        if eps <= 0:
            raise ValueError("eps must be > 0")
        if alpha <= 0 and init_eta is None:
            # If user doesn't set init_eta, we need alpha to construct r_eps
            raise ValueError("alpha must be > 0 when init_eta is None")
        if weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")

        defaults = dict(eps=eps, alpha=alpha, init_eta=init_eta, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Global scalars
        self._initialized = False
        # self.state["eta"] will be set in _maybe_initialize

    @torch.no_grad()
    def _maybe_initialize(self):
        if self._initialized:
            return

        # 1) Initialize per-parameter buffers:
        #       v_{-1} = eps^2 * 1,  w0 = snapshot of initial parameters
        #    Also compute max ||w0||_inf to set r_eps if needed.
        max_w0_inf = 0.0
        for group in self.param_groups:
            eps = group["eps"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                st = self.state[p]
                st["v"] = torch.full_like(p, eps * eps)
                st["w0"] = p.detach().clone()
                with torch.no_grad():
                    max_w0_inf = max(max_w0_inf, p.detach().abs().max().item())

        # 2) Set eta_{-1}
        init_eta = None
        # If user explicitly supplied init_eta in any group, honor it
        for group in self.param_groups:
            if group.get("init_eta", None) is not None:
                init_eta = group["init_eta"]
                break

        if init_eta is None:
            # Default r_eps per paper suggestion
            # r_eps = alpha * (1 + ||w0||_inf)
            # Use max alpha across groups (usually they are equal)
            alpha_max = max(g["alpha"] for g in self.param_groups)
            init_eta = alpha_max * (1.0 + max_w0_inf)

        self.state["eta"] = float(init_eta)
        self._initialized = True

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Follows Algorithm 1 (coordinate-wise DoG):
          - v_t = v_{t-1} + g_t^2
          - eta_t = max(eta_{t-1}, ||w_t - w0||_inf)  (global over all params)
          - w_{t+1} = w_t - eta_t * g_t / sqrt(v_t)
        """
        self._maybe_initialize()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Compute global ||w_t - w0||_inf across all params
        max_inf = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                diff = (p.detach() - st["w0"]).abs().max().item()
                if diff > max_inf:
                    max_inf = diff

        # Update eta_t
        eta_prev = float(self.state["eta"])
        eta_t = max(eta_prev, max_inf)
        self.state["eta"] = float(eta_t)

        # Parameter updates
        for group in self.param_groups:
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    g = g.add(p, alpha=wd)

                st = self.state[p]
                st["v"].add_(g * g)              # v_t = v_{t-1} + g^2
                denom = st["v"].sqrt_()          # sqrt(v_t) in-place
                p.addcdiv_(g, denom, value=-eta_t)

        return loss


def build_dog_coordinate(params, args):
    """
    Builder for the refactored codebase registry.

    CLI mapping:
      --eps           : numerical stabilizer (default 1e-8)
      --alpha         : scales r_eps = alpha * (1 + ||w0||_inf) if --init-eta not set
      --init-eta      : optional explicit initial eta (overrides alpha-based r_eps)
      --weight-decay  : optional L2

    Backward-compat: if args has --reps-rel (from your DoG/LDoG), we treat it
    as alpha when --alpha is not provided.
    """
    eps = getattr(args, "eps", 1e-8)

    # Prefer explicit --alpha; else fall back to --reps-rel; else 1e-6
    alpha = getattr(args, "alpha", None)
    if alpha is None:
        alpha = getattr(args, "reps_rel", 1e-6)

    init_eta = getattr(args, "init_eta", None)
    weight_decay = getattr(args, "weight_decay", 0.0)

    return DoGCoordinate(
        params,
        eps=eps,
        alpha=alpha,
        init_eta=init_eta,
        weight_decay=weight_decay,
    )

