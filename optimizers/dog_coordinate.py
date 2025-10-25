import torch
from torch.optim import Optimizer

class DoGCoordinate(Optimizer):
    """
    Coordinate-wise DoG (no projection).
    v_{-1} = eps^2 * 1
    eta_{-1} = max_group alpha * (1 + ||w0||_inf)
    For t:
      v_t = v_{t-1} + g_t^2
      eta_t = max(eta_{t-1}, ||w_t - w0||_inf)
      w_{t+1} = w_t - eta_t * g_t / sqrt(v_t)
    """
    def __init__(self, params, eps: float = 1e-8, alpha: float = 1e-6):
        if eps <= 0 or alpha <= 0:
            raise ValueError("eps and alpha must be positive.")
        super().__init__(params, dict(eps=eps, alpha=alpha))
        self._initialized = False

    @torch.no_grad()
    def _maybe_initialize(self):
        if self._initialized:
            return
        max_w0_inf = 0.0
        for group in self.param_groups:
            eps = group["eps"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                st = self.state[p]
                st["v"] = torch.full_like(p, eps * eps)   # v_{-1}
                st["w0"] = p.detach().clone()             # snapshot
                max_w0_inf = max(max_w0_inf, p.detach().abs().max().item())
        # eta_{-1} = max over groups alpha*(1+||w0||_inf)
        r_eps = 0.0
        for group in self.param_groups:
            r_eps = max(r_eps, group["alpha"] * (1.0 + max_w0_inf))
        self.state["eta"] = r_eps
        self._initialized = True

    @torch.no_grad()
    def step(self, closure=None):
        self._maybe_initialize()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # global max ||w_t - w0||_inf
        max_inf = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                diff = (p.detach() - st["w0"]).abs().max().item()
                if diff > max_inf:
                    max_inf = diff

        eta_prev = self.state["eta"]
        eta_t = max(eta_prev, max_inf)
        self.state["eta"] = eta_t

        # update: w <- w - eta * g / sqrt(v)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                g = p.grad
                st["v"].add_(g * g)                         # v_t
                p.addcdiv_(g, st["v"].sqrt(), value=-eta_t) # step
        return loss

def build_dog_coordinate(params, args):
    # Uses shared CLI flags: --eps and --alpha
    eps = getattr(args, "eps", 1e-8)
    alpha = getattr(args, "alpha", 1e-6)
    return DoGCoordinate(params, eps=eps, alpha=alpha)

