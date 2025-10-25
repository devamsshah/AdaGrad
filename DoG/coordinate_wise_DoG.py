#!/usr/bin/env python3
"""
har_pytorch_dog.py

UCI HAR classification using a custom Coordinate-wise DoG optimizer
(Algorithm 1: without projection). Trains for 100 epochs and saves a
PDF chart of Epoch vs Loss (train & test): har_loss_dog.pdf

- Dataset: UCI HAR (561 engineered features)
- Split: 80/20 stratified (combining official train/test first)
- Loss: CrossEntropyLoss
- Optimizer: DoG (this file)

Requirements:
  python>=3.9, numpy, pandas, matplotlib, torch, scikit-learn, requests
"""

import os
import io
import sys
import time
import zipfile
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "../datasets/UCI_HAR/UCI_HAR_Dataset"

def load_ucihar_all() -> Tuple[np.ndarray, np.ndarray, list]:
    x_train = os.path.join(DATA_DIR, "train", "X_train.txt")
    y_train = os.path.join(DATA_DIR, "train", "y_train.txt")
    x_test  = os.path.join(DATA_DIR, "test",  "X_test.txt")
    y_test  = os.path.join(DATA_DIR, "test",  "y_test.txt")
    labels  = os.path.join(DATA_DIR, "activity_labels.txt")

    # Use the modern sep=r"\s+" instead of deprecated delim_whitespace
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

    def __len__(self): 
        return self.X.shape[0]
    def __getitem__(self, i): 
        return self.X[i], self.y[i]


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
    def forward(self, x): 
        return self.net(x)


# ---------------------------
# Custom Optimizer: Coordinate-wise DoG (no projection)
# Algorithm (per your figure):
#   v_{-1} = eps^2 * 1_d
#   eta_{-1} = r_eps (we set r_eps = alpha * (1 + ||w0||_inf))
#   for t:
#     g_t = mean minibatch gradient
#     v_t = v_{t-1} + (g_t ⊙ g_t)
#     Λ_t = diag(sqrt(v_t))
#     η_t = max(η_{t-1}, ||w_t - w0||_∞)
#     w_{t+1} = w_t - η_t Λ_t^{-1} g_t
# Notes:
# - We implement per-parameter buffers (v, w0) and a global eta.
# - Gradients provided by PyTorch (with CrossEntropyLoss reduction='mean')
#   are already batch-averaged.
# ---------------------------
class DoG(Optimizer):
    def __init__(self, params, eps: float = 10e-8, alpha: float = 10e-7):
        if eps <= 0 or alpha <= 0:
            raise ValueError("eps and alpha must be positive.")
        defaults = dict(eps=eps, alpha=alpha)
        super().__init__(params, defaults)

        # Initialize state: v = eps^2, w0 snapshot, eta = r_eps
        self._initialized = False

    @torch.no_grad()
    def _maybe_initialize(self):
        if self._initialized:
            return
        max_w0_inf = 0.0
        for group in self.param_groups:
            eps = group["eps"]
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    state["v"] = torch.full_like(p, eps * eps)  # v_{-1}
                    state["w0"] = p.detach().clone()            # snapshot of init
                    max_w0_inf = max(max_w0_inf, p.detach().abs().max().item())
        # η_{-1} = r_eps = alpha * (1 + ||w0||_inf)
        r_eps = 0.0
        for group in self.param_groups:
            alpha = group["alpha"]
            r_eps = max(r_eps, alpha * (1.0 + max_w0_inf))
        self.state["eta"] = r_eps
        self._initialized = True

    @torch.no_grad()
    def step(self, closure=None):
        self._maybe_initialize()

        # Optional closure (not used here)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Compute current ||w_t - w0||_inf across all params
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
        eta_prev = self.state["eta"]
        eta_t = max(eta_prev, max_inf)
        self.state["eta"] = eta_t

        # Parameter updates: w <- w - eta * g / sqrt(v)
        for group in self.param_groups:
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                g = p.grad
                st["v"].add_(g * g)                     # v_t = v_{t-1} + g^2
                denom = st["v"].sqrt().add_(0.0)        # Λ_t^{-1} ~ 1/sqrt(v)
                p.addcdiv_(g, denom, value=-eta_t)      # p = p - eta * g / sqrt(v)
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
def main():
    X, y, label_names = load_ucihar_all()

    # Stratified 80/20 split on combined set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Standardize by training statistics
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # Tensors/datasets
    train_ds = HARSet(X_train, y_train - 1)  # labels 0..5
    test_ds  = HARSet(X_test,  y_test  - 1)

    batch_size = 128
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=X.shape[1], hidden=128, dropout=0.2, num_classes=6).to(device)

    # ---- DoG optimizer params ----
    # eps ~ Adam epsilon scale; alpha controls initial r_eps = alpha * (1 + ||w0||_inf)
    optimizer = DoG(model.parameters(), eps=1e-8, alpha=1e-6)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 1000
    train_losses, test_losses = [], []
    best_test_acc, best_state = 0.0, None

    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        te_loss, te_acc = eval_one_epoch(model, test_loader, loss_fn, device)
        train_losses.append(tr)
        test_losses.append(te_loss)

        # checkpoint best
        if te_acc > best_test_acc:
            best_test_acc = te_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {ep:03d}/{epochs} | train loss: {tr:.4f} | test loss: {te_loss:.4f} | test acc: {te_acc*100:.2f}%")

    t1 = time.time()
    print(f"[INFO] Finished in {t1 - t0:.1f}s")
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[INFO] Restored best checkpoint (test acc = {best_test_acc*100:.2f}%)")
        torch.save(best_state, "best_har_dog.pth")
    # Save Epoch vs Loss to PDF (as requested in earlier step)
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("UCI HAR — DoG optimizer (Train/Test Loss)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("har_loss_dog_a=10e-7.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print("[INFO] Saved loss curves to har_loss_dog.pdf")


if __name__ == "__main__":
    main()
