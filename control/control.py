#!/usr/bin/env python3
"""
har_pytorch_adam.py

Human Activity Recognition (UCI HAR) with PyTorch + Adam.

- Task: Predict activity (e.g., WALKING, SITTING) from smartphone accelerometer/gyroscope features.
- Data: UCI HAR "UCI HAR Dataset" (561 features already precomputed by the dataset authors).
- Split: 80/20 STRATIFIED split (we combine the official train/test and then split as requested).
- Optimizer: torch.optim.Adam
- Loss: CrossEntropyLoss
- Output: Prints accuracy and plots Epoch vs Loss (train & test).

NO SYNTHETIC FALLBACK: If download fails, the script exits with an error.

Requirements:
  - python 3.9+
  - numpy, pandas, matplotlib, torch, scikit-learn, requests

Run:
  python har_pytorch_adam.py
"""

import os
import io
import sys
import zipfile
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

def load_ucihar_all() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load UCI HAR features and labels by combining official train & test splits.

    Returns:
      X_all: (N, 561) float32 features
      y_all: (N,) int64 labels in [1..6]
      label_names: (6,) list of activity names indexed by label-1
    """
    # Paths
    DATA_DIR = "../datasets/UCI_HAR/UCI_HAR_Dataset"
    x_train = os.path.join(DATA_DIR, "train", "X_train.txt")
    y_train = os.path.join(DATA_DIR, "train", "y_train.txt")
    x_test = os.path.join(DATA_DIR, "test", "X_test.txt")
    y_test = os.path.join(DATA_DIR, "test", "y_test.txt")
    lbl_path = os.path.join(DATA_DIR, "activity_labels.txt")

    # Load
    X_tr = pd.read_csv(x_train, sep=r"\s+", header=None).values
    y_tr = pd.read_csv(y_train, sep=r"\s+", header=None).values.ravel()
    X_te = pd.read_csv(x_test, sep=r"\s+", header=None).values
    y_te = pd.read_csv(y_test, sep=r"\s+", header=None).values.ravel()

    X_all = np.vstack([X_tr, X_te]).astype(np.float32)
    y_all = np.concatenate([y_tr, y_te]).astype(np.int64)  # labels 1..6

    # Load label names
    act_df = pd.read_csv(lbl_path, sep=r"\s+", header=None, names=["id", "name"])
    # Ensure order matches label ids
    act_df = act_df.sort_values("id")
    label_names = act_df["name"].str.replace("_", " ").tolist()  # e.g., WALKING -> "WALKING"

    return X_all, y_all, label_names


# ---------------------------
# Torch dataset / model
# ---------------------------
class HARSet(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

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
# Training utilities
# ---------------------------
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    running_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        running_loss += loss.item() * xb.size(0)
    return running_loss / len(loader.dataset)


def eval_one_epoch(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            running_loss += loss.item() * xb.size(0)
            running_acc += (torch.argmax(logits, dim=1) == yb).sum().item()
            n += xb.size(0)
    return running_loss / n, running_acc / n


# ---------------------------
# Main
# ---------------------------
def main():
    X, y, label_names = load_ucihar_all()

    # Stratified 80/20 split (shuffle once, stratify by label)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Standardize using TRAIN stats only
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # Torch loaders
    train_ds = HARSet(X_train, y_train - 1)  # make labels 0..5 for CrossEntropy
    test_ds = HARSet(X_test, y_test - 1)

    batch_size = 128
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # Model / Optimizer / Loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=X.shape[1], hidden=128, dropout=0.2, num_classes=6).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Train
    epochs = 1000
    train_losses, test_losses = [], []
    best_test_acc, best_state = 0.0, None

    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, device)
        te_loss, te_acc = eval_one_epoch(model, test_loader, loss_fn, device)

        train_losses.append(tr_loss)
        test_losses.append(te_loss)

        if te_acc > best_test_acc:
            best_test_acc = te_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {ep:02d}/{epochs} "
              f"| train loss: {tr_loss:.4f} "
              f"| test loss: {te_loss:.4f} "
              f"| test acc: {te_acc*100:.2f}%")

    t1 = time.time()
    print(f"\n[INFO] Training finished in {t1 - t0:.1f}s. Best test accuracy: {best_test_acc*100:.2f}%")

    # Restore best model (optional)
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, "best_har.pth")

    # Final evaluation
    final_test_loss, final_test_acc = eval_one_epoch(model, test_loader, loss_fn, device)
    print(f"[INFO] Test loss: {final_test_loss:.4f} | Test accuracy: {final_test_acc*100:.2f}%")

    # Plot Epoch vs Loss (train & test)
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("UCI HAR — Adam convergence (Train/Test Loss)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("har_loss.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    plt.show()
    print("[INFO] Saved loss curves to har_loss.pdf")

if __name__ == "__main__":
    main()
