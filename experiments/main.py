from parser.args import parse_args 
from utils.seed import set_seed
from datasets import get_dataset
from models import get_model
from optimizers import get_optimizer
from train import train_eval
from utils.plot import make_run_artifacts
from utils.paths import resolve_data_dir

import torch

def main():
    args = parse_args()
    args.data_dir = resolve_data_dir(args.dataset, args.data_root, args.data_dir)
    set_seed(args.seed, args.device)

    # Dataset & loaders
    make_loaders = get_dataset(args.dataset)
    train_loader, test_loader, meta = get_dataset(args.dataset)(args)
 

    # Model
    build_model = get_model(args.model)
    model = build_model(meta["input_dim"], meta["num_classes"], args)

    # Optimizer
    build_opt = get_optimizer(args.optimizer)
    optimizer = build_opt(model.parameters(), args)

    # Train + Eval
    history, metrics, best_state = train_eval(model, train_loader, test_loader, optimizer, args)

    # Save artifacts
    make_run_artifacts(args, history, metrics)

    # (optional) save best weights
    torch.save(best_state, f"best_{args.dataset}_{args.optimizer}.pth")

if __name__ == "__main__":
    main()

