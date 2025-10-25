import torch
from torch.optim import Optimizer

class DoGMomentum(Optimizer):
    def __init__(self, params, eps: float = 1e-8, alpha: float = 1e-7, beta: float = 0.9):
        if eps <= 0 or alpha <= 0 or not (0.0 <= beta <= 1.0):
            raise ValueError("eps, alpha > 0 and 0<=beta<=1 required.")
        defaults = dict(eps=eps, alpha=alpha, beta=beta)
        super().__init__(params, defaults)
        self._initialized = False

    @torch.no_grad()
    def _maybe_initialize(self):
        if self._initialized:
            return
        # Initialize per-parameter state
        for group in self.param_groups:
            eps = group["eps"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                st = self.state[p]
                st["v"] = torch.full_like(p, eps * eps)   # v_{-1}
                st["m"] = torch.zeros_like(p)             # m_{-1}
                st["w0"] = p.detach().clone()             # snapshot
        # Start eta with a tiny positive (can also use alpha*(1+||w0||_inf))
        self.state["eta"] = 1e-6
        self._initialized = True

    @torch.no_grad()
    def step(self, closure=None):
        self._maybe_initialize()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Global ||w_t - w0||_inf for eta update
        max_inf = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                diff = (p.detach() - self.state[p]["w0"]).abs().max().item()
                if diff > max_inf:
                    max_inf = diff

        eta_prev = self.state["eta"]
        self.state["eta"] = max(eta_prev, max_inf)
        eta_t = self.state["eta"]

        for group in self.param_groups:
            beta = group["beta"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                g = p.grad
                st["m"].mul_(beta).add_(g, alpha=(1.0 - beta))   # m_t
                st["v"].add_(st["m"] * st["m"])                  # v_t += m_t^2
                p.addcdiv_(st["m"], st["v"].sqrt(), value=-eta_t)  # w <- w - eta * m / sqrt(v)
        return loss

def build_dog_momentum(params, args):
    return DoGMomentum(params, eps=args.eps, alpha=args.alpha, beta=args.beta)

