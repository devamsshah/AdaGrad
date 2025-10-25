import argparse
import torch

DEFAULT_DATA_DIR = "../../datasets/UCI_HAR/UCI_HAR_Dataset"

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HAR training runner")

    # Core selection
    p.add_argument("--dataset", type=str, default="uci_har",
                   help="Dataset name (e.g., uci_har)")
    p.add_argument("--optimizer", type=str, default="dog_momentum",
                   help="Optimizer name (e.g., dog_momentum)")
    p.add_argument("--model", type=str, default="mlp",
                   help="Model name (e.g., mlp)")

    # Common hyperparams
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)

    # Dataset args
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)

    # Device/seed
    p.add_argument("--device", type=str,
                   default=("cuda" if torch.cuda.is_available() else "cpu"),
                   choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", type=str, default="")

    # Optimizer-agnostic lr-ish knobs (superset; your optimizer can pick what it needs)
    p.add_argument("--alpha", type=float, default=1e-8, help="Base DoG step size")
    p.add_argument("--eps",   type=float, default=1e-8, help="Numerical stabilizer")
    p.add_argument("--beta",  type=float, default=0.9,  help="Momentum coeff")
    
    # Adam knobs (used if --optimizer adam)
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    p.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    p.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")
    p.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay")


    return p

def parse_args():
    return build_parser().parse_args()

