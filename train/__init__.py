from .loops import train_one_epoch, eval_one_epoch, train_eval
from .loops import train_eval as _train_cls
from .loops_lm import train_eval_lm as _train_lm

def train_eval(model, train_loader, test_loader, optimizer, args, meta=None):
    # route based on task
    task = (meta or {}).get("task", "cls")
    if task == "lm":
        return _train_lm(model, train_loader, test_loader, optimizer, args)
    return _train_cls(model, train_loader, test_loader, optimizer, args)


__all__ = ["train_one_epoch", "eval_one_epoch", "train_eval"]

