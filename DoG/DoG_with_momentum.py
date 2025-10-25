"""
DoG_with_momentum.py

Same HAR pipeline as your script; 
now takes parameters via CLI:
  --alpha, --eps, --beta, --epochs, --batch-size, --hidden, --dropout,
  --data-dir, --device, --seed
"""
import os
import io
import sys
import time
import json
import zipfile
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import re

from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

DEFAULT_DATA_DIR = "../datasets/UCI_HAR/UCI_HAR_Dataset"
eta, info, ret, epoch = False, False, True, False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1e-8, help="Base DoG step size")
    parser.add_argument("--eps",   type=float, default=1e-8, help="Numerical stabilizer")
    parser.add_argument("--beta",  type=float, default=0.9,  help="Momentum coefficient")
    parser.add_argument("--epochs", type=int, default=1000,  help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden layer width")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout prob")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="UCI-HAR root dir")
    parser.add_argument("--device", type=str,
                        default=("cuda" if torch.cuda.is_available() else "cpu"),
                        choices=["cuda", "cpu"], help="Device override")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-name", type=str, default="", help="Optional run name tag")
    return parser.parse_args()

def load_ucihar_all(data_dir: str) -> Tuple[np.ndarray, np.ndarray, list]:
    x_train = os.path.join(data_dir, "train", "X_train.txt")
    y_train = os.path.join(data_dir, "train", "y_train.txt")
    x_test  = os.path.join(data_dir, "test",  "X_test.txt")
    y_test  = os.path.join(data_dir, "test",  "y_test.txt")
    labels  = os.path.join(data_dir, "activity_labels.txt")

    X_tr = pd.read_csv(x_train, sep=r"\s+", header=None).values
    y_tr = pd.read_csv(y_train, sep=r"\s+", header=None).values.ravel()
    X_te = pd.read_csv(x_test,  sep=r"\s+", header=None).values
    y_te = pd.read_csv(y_test,  sep=r"\s+", header=None).values.ravel()

    X_all = np.vstack([X_tr, X_te]).astype(np.float32)
    y_all = np.concatenate([y_tr, y_te]).astype(np.int64)  # labels 1..6
    act_df = pd.read_csv(labels, sep=r"\s+", header=None, names=["id", "name"])
    act_df = act_df.sort_values("id")
    label_names = act_df["name"].str.replace("_", " ").tolist()
    return X_all, y_all, label_names


# ---------------------------
# Torch dataset/model
# ---------------------------
class HARSet(Dataset):
    def __init__(self, X: np.ndarray, y_zero_based: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_zero_based, dtype=torch.long)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]



class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.2, num_classes: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, x): return self.net(x)

# ---------------------------
# DoG with Momentum (Algorithm 2)
# ---------------------------
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
        max_w0_inf = 0.0
        for group in self.param_groups:
            eps = group["eps"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                st = self.state[p]
                st["v"] = torch.full_like(p, eps * eps)   # v_{-1}
                st["m"] = torch.zeros_like(p)             # m_{-1} = 0
                st["w0"] = p.detach().clone()             # snapshot
                max_w0_inf = max(max_w0_inf, p.detach().abs().max().item())
        # eta_{-1} = r_eps = alpha * (1 + ||w0||_inf)
        r_eps = 1e-6
        #for group in self.param_groups:
        #    r_eps = max(r_eps, group["alpha"] * (1.0 + max_w0_inf))
        self.state["eta"] = r_eps
        self._initialized = True

    @torch.no_grad()
    def step(self, closure=None):
        self._maybe_initialize()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # η_t update needs global ||w_t - w0||_∞
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
        if eta:
            print(f"[ETA] {eta_t}")

        # Parameter updates using m_t and v_t accumulated with m_t^2
        for group in self.param_groups:
            beta = group["beta"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                g = p.grad
                st["m"].mul_(beta).add_(g, alpha=(1.0 - beta))   # m_t
                st["v"].add_(st["m"] * st["m"])                  # v_t += m_t^2
                denom = st["v"].sqrt()                           # sqrt(v_t)
                p.addcdiv_(st["m"], denom, value=-eta_t)         # w <- w - eta * m / sqrt(v)
        return loss


# ---------------------------
# Train / Eval helpers
# ---------------------------
def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    running = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        running += loss.item() * xb.size(0)
        n += xb.size(0)
    return running / n

def eval_one_epoch(model, loader, loss_fn, device):
    model.eval()
    running_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            running_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            n += xb.size(0)
    return running_loss / n, correct / n


# ---------------------------
# Main
# ---------------------------
def main(args):
    # Repro
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    X, y, _ = load_ucihar_all(args.data_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    train_ds = HARSet(X_train, y_train - 1)
    test_ds  = HARSet(X_test,  y_test  - 1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device(args.device)
    model = MLP(in_dim=X.shape[1], hidden=args.hidden, dropout=args.dropout, num_classes=6).to(device)

    optimizer = DoGMomentum(model.parameters(), eps=args.eps, alpha=args.alpha, beta=args.beta)
    loss_fn = nn.CrossEntropyLoss()

    epochs = args.epochs
    train_losses, test_losses = [], []
    best_test_acc, best_state = 0.0, None
    min_test_loss = float("inf")

    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        te_loss, te_acc = eval_one_epoch(model, test_loader, loss_fn, device)
        train_losses.append(tr)
        test_losses.append(te_loss)
        min_test_loss = min(min_test_loss, te_loss)

        if te_acc > best_test_acc:
            best_test_acc = te_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch:
            print(f"Epoch {ep:03d}/{epochs} | train loss: {tr:.6f} | test loss: {te_loss:.6f} | test acc: {te_acc*100:.2f}%")

    t1 = time.time()
    if info:
        print(f"[INFO] Finished in {t1 - t0:.1f}s")
    if best_state is not None:
        model.load_state_dict(best_state)
        if info:
            print(f"[INFO] Restored best checkpoint (test acc = {best_test_acc*100:.2f}%)")
        torch.save(best_state, "best_har_dog.pth")

    # --- Naming helpers: include args in PDF name ---
    def slug_float(x: float) -> str:
        return re.sub(r"[^0-9a-zA-Z\-\._]", "", f"{x:.1e}")
    def slug(s: str) -> str:
        return re.sub(r"[^0-9a-zA-Z\-\._]", "_", s)[:50]

    pdf_name = (
        f"har_loss_dog"
        f"_a={slug_float(args.alpha)}"
        f"_e={slug_float(args.eps)}"
        f"_b={slug_float(args.beta)}"
        f"_hid={args.hidden}"
        f"_do={slug_float(args.dropout)}"
        f"_bs={args.batch_size}"
        f"_seed={args.seed}"
        + (f"_run={slug(args.run_name)}" if args.run_name else "")
        + ".pdf"
    )

    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("UCI HAR — DoG optimizer (Train/Test Loss)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pdf_name, format="pdf", bbox_inches="tight")
    plt.close()
    if info:
        print(f"[INFO] Saved loss curves to {pdf_name}")

    # Write a small JSON result
    result = {
        "min_test_loss": float(min_test_loss),
        "best_test_acc": float(best_test_acc),
        "alpha": args.alpha,
        "eps": args.eps,
        "beta": args.beta,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "pdf": pdf_name,
        "run_name": args.run_name,
    }
    out_json = os.path.splitext(pdf_name)[0] + ".json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    if info:
        print(f"[INFO] Wrote {out_json}")

    # Print a machine-parsable final line (sweeper will read this)
    if ret:
        print(f"MIN_LOSS:{min_test_loss:.8f}")
        print(f"MAX_ACCY:{best_test_acc:.8f}")
    return best_test_acc 


if __name__ == "__main__":
    args = parse_args()
    val = main(args)
