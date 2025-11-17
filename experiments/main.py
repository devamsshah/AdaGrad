# experiments/main.py
from parser.args import parse_args
from utils.seed import set_seed
from datasets import get_dataset
from models import get_model
from optimizers import get_optimizer
from train import train_eval
from utils.plot import make_run_artifacts
from utils.paths import resolve_data_dir

import torch
import sys

def main():
    # 1) Parse CLI
    args = parse_args()

    # 2) Resolve data dir (optional data_root support; ok if you pass only --data-dir)
    args.data_dir = resolve_data_dir(
        getattr(args, "dataset", None),
        getattr(args, "data_root", None),
        getattr(args, "data_dir", None),
    )

    # 3) Repro & device
    set_seed(args.seed, args.device)
    device = torch.device(args.device)

    # 4) Dataset & loaders (+ meta to route LM vs CLS)
    try:
        make_loaders = get_dataset(args.dataset)
    except Exception as e:
        print(f"[DATASET] Unknown dataset '{args.dataset}': {e}", file=sys.stderr)
        raise
    train_loader, test_loader, meta = make_loaders(args)

    # 5) Model
    try:
        build_model = get_model(args.model)
    except Exception as e:
        print(f"[MODEL] Unknown model '{args.model}': {e}", file=sys.stderr)
        raise
    # builders ignore input_dim/num_classes for LM tasks; we still pass meta fields
    input_dim = meta.get("input_dim", None)
    num_classes = meta.get("num_classes", None)
    model = build_model(input_dim, num_classes, args).to(device)

    # 6) Optimizer (must be built **before** calling train_eval)
    try:
        build_opt = get_optimizer(args.optimizer)
    except Exception as e:
        print(f"[OPTIM] Unknown optimizer '{args.optimizer}': {e}", file=sys.stderr)
        raise
    optimizer = build_opt(model.parameters(), args)

    # 7) Train + Eval (train.train_eval routes by meta['task'])
    history, metrics, best_state = train_eval(model, train_loader, test_loader, optimizer, args, meta)

    # 8) Save artifacts (PDF/JSON + MIN_LOSS/MAX_ACCY already printed by loops)
    make_run_artifacts(args, history, metrics)

    # 9) Save best weights next to results (or adapt to your results dir)
    torch.save(best_state, f"best_{args.dataset}_{args.model}_{args.optimizer}.pth")

if __name__ == "__main__":
    main()

