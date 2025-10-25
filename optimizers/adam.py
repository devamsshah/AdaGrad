import torch
from torch.optim import Adam

def build_adam(params, args):
    # Generic knobs; fall back if not provided
    lr = getattr(args, "lr", 1e-3)
    beta1 = getattr(args, "beta1", 0.9)
    beta2 = getattr(args, "beta2", 0.999)
    eps = getattr(args, "eps", 1e-8)            # reuses global --eps if set
    wd = getattr(args, "weight_decay", 0.0)
    return Adam(params, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=wd)

