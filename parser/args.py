import argparse
import torch

# One root for all datasets; subfolders are resolved per dataset
DEFAULT_DATA_ROOT = "../../datasets"

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HAR training runner")

    # Core selection
    p.add_argument("--dataset", type=str, default="uci_har",
                   help="Dataset name (e.g., uci_har, cifar10, mnist)")
    p.add_argument("--optimizer", type=str, default="dog_momentum",
                   help="Optimizer name (e.g., dog_momentum, dog, adam)")
    p.add_argument("--model", type=str, default="mlp",
                   help="Model name (e.g., mlp, cnn, resnet)")

    # Common hyperparams
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)

    # Dataset paths
    p.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT,
                   help="Root folder under which dataset subfolders live")
    p.add_argument("--data-dir", type=str, default=None,
                   help="Full path to a specific dataset directory (overrides --data-root)")

    # Device/seed
    p.add_argument("--device", type=str,
                   default=("mps" if torch.backends.mps.is_available() else "cpu"),
                   choices=["mps", "cpu"])
    p.add_argument("--seed", type=int, default=55)
    p.add_argument("--run-name", type=str, default="")

    # DoG/Adam knobs
    p.add_argument("--alpha", type=float, default=1e-8, help="Base DoG step size")
    p.add_argument("--eps",   type=float, default=1e-8, help="Numerical stabilizer")
    p.add_argument("--beta",  type=float, default=0.9,  help="Momentum coeff")

    # Adam knobs
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    p.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    p.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")
    p.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay")

    # CNN knobs
    p.add_argument("--cnn-width", type=int, default=64, help="Base channel width for CNN")
    p.add_argument("--in-channels", type=int, default=3, help="Input channels fallback for image models")

    # ResNet knobs
    p.add_argument("--resnet-depth", type=int, default=18, choices=[18,34], help="ResNet depth")
    p.add_argument("--resnet-width", type=int, default=64, help="ResNet base width")

    # DoG / LDoG knobs
    p.add_argument("--reps-rel", type=float, default=None,
               help="DoG r_eps relative factor; if None, falls back to --alpha or 1e-6")
    p.add_argument("--init-eta", type=float, default=None,
               help="Override initial eta for DoG/LDoG (optional)")
    p.add_argument("--dog-layerwise", action="store_true",
               help="Use LDoG (layer-wise DoG) instead of global DoG")
    
    # LM (GPT) knobs
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head",  type=int, default=6)
    p.add_argument("--n-embd",  type=int, default=384)
    p.add_argument("--bias", action="store_true", help="use bias in LayerNorm/Linear")
    p.add_argument("--gpt2-preset", type=str, default="gpt2", choices=["gpt2","gpt2-medium","gpt2-large","gpt2-xl"])


    return p

def parse_args():
    return build_parser().parse_args()

