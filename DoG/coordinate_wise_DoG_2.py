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

# --- DoG Optimizer Class (Correction/Verification) ---
# The class below is the complete implementation that correctly handles the 
# 'alpha' parameter in __init__ and follows Algorithm 1 steps in 'step'.

class DoG(Optimizer):
    """
    Implements the Coordinate-wise DoG (Difference of Gradients) optimizer 
    as described in Algorithm 1 (without projection).
    
    This optimizer uses a coordinate-wise second moment estimate (v) and 
    updates the step size (eta) based on the maximum L-infinity norm of the 
    weight deviation from the initial state (w0).
    
    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): The initial fixed learning rate (r_epsilon). If 'alpha' is provided, 
                              this value is overridden by the alpha-based calculation. (default: 1e-3)
        eps (float, optional): Epsilon for numerical stability, used in v_-1 initialization. (default: 1e-8)
        alpha (float, optional): Scaling factor for initial learning rate calculation: 
                                 lr = alpha * (1 + max(||w0||_inf)). (default: None)
    """

    def __init__(self, params, lr=1e-3, eps=1e-8, alpha=None):
        
        # CRITICAL FIX: Convert the iterable 'params' (usually a generator) into a persistent list.
        # This prevents the list from being exhausted by the alpha calculation loop.
        param_groups = list(params)
        
        # --- 1. Handle alpha-based learning rate calculation (Algorithm 1, Line 1 hint) ---
        if alpha is not None:
            if not 0.0 <= alpha:
                raise ValueError(f"Invalid alpha value: {alpha}")
            
            # Find the maximum L-infinity norm of all initial parameters (w0)
            max_w0_linf = torch.tensor(0.0)
            
            # Iterate over the persistent list of parameter groups/tensors
            for group in param_groups:
                
                # Determine if 'group' is a parameter dict (user-defined group) or a raw Tensor
                if isinstance(group, dict):
                    tensor_list = group['params']
                else:
                    tensor_list = [group]
                
                for p in tensor_list:
                    if p.ndim > 0 and p.data.numel() > 0:
                        # Calculate ||w0||_inf for this parameter
                        # We use .max() and .item() to ensure we compare scalar values across groups
                        linf_val = torch.abs(p.data).max().item()
                        if linf_val > max_w0_linf.item():
                            max_w0_linf = torch.tensor(linf_val)

            # Calculate the new learning rate r_epsilon (lr)
            # r_epsilon = alpha * (1 + max(||w0||_inf))
            lr = float(alpha * (1.0 + max_w0_linf.item()))
            
        # --- 2. Validation and Initialization ---
            
        if not 0.0 <= lr:
            raise ValueError(f"Invalid initial learning rate (r_epsilon): {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        
        defaults = dict(lr=lr, eps=eps)
        
        # Pass the persistent list of parameter groups to the base class
        super(DoG, self).__init__(param_groups, defaults)

        # Initialize global state for tracking eta_t-1 (Line 2: eta_-1 = r_epsilon)
        self.state['current_eta'] = lr

        # Initialize per-parameter state for v_-1 and w_0
        eps_sq = eps * eps
        
        for group in self.param_groups:
            for p in group['params']:
                if p.ndim == 0 or p.data.numel() == 0:
                    continue 
                
                state = self.state[p]
                
                # Line 2: Initialize v_-1 = epsilon^2 * 1_d. 'v' stores v_t-1.
                state['v'] = torch.full_like(p.data, eps_sq, memory_format=torch.preserve_format)
                
                # Line 8 requires storing w_0 (initial weights)
                state['w0'] = p.data.clone().detach()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step (Algorithm 1)."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        if not self.param_groups:
            return loss

        # --- 1. Calculate global L-infinity norm term for step size update (Line 8) ---
        
        # Determine the device for the global L-infinity norm calculation
        first_param = self.param_groups[0]['params'][0]
        device = first_param.device if first_param.is_cuda else torch.device('cpu')

        # Calculate max(||w_t - w_0||_inf) across all parameters
        max_deviation_linf = torch.tensor(0.0, device=device)
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                w0 = state['w0']
                
                # Calculate ||w_t - w_0||_inf for this parameter
                deviation_linf = torch.abs(p.data - w0).max()
                
                # Update the global maximum deviation
                if deviation_linf > max_deviation_linf:
                    max_deviation_linf = deviation_linf

        # --- 2. Update global step size eta_t (Line 8) ---
        
        # Get eta_t-1 (which is stored in self.state['current_eta'])
        prev_eta = self.state['current_eta']
        
        # Line 8: eta_t = max{eta_t-1, ||w_t - w_0||_inf}
        new_eta = torch.max(torch.tensor(prev_eta, device=device), max_deviation_linf)
        
        # Store the new eta for the next step (eta_t becomes eta_t-1 for the next step)
        self.state['current_eta'] = new_eta.item()
        
        # --- 3. Perform coordinate-wise updates (Lines 6 and 9) ---
        
        eta_t = new_eta
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                v = state['v'] # v is v_t-1 here

                # Line 6: v_t = v_t-1 + (g_t .odot g_t)
                v.addcmul_(grad, grad, value=1) 

                # Line 9: w_t+1 = w_t - eta_t * Lambda_t^-1 * g_t
                # Lambda_t^-1 * g_t is equivalent to g_t / sqrt(v_t).
                denom = v.sqrt()
                
                # Update the parameter in place: p.data = p.data - eta_t * grad / denom
                p.data.addcdiv_(grad, denom, value=-eta_t)

        return loss

# --- End of DoG Optimizer Class ---

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


# --------------------------
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

@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, device):
    model.eval()
    running = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        running += loss.item() * xb.size(0)
        n += xb.size(0)
    return running / n


# ---------------------------
# Main
# ---------------------------
def main():
    if not os.path.isdir(DATA_DIR):
        print(f"[ERROR] Data directory not found: {DATA_DIR}. Please ensure the UCI HAR Dataset is available.")
        # Attempt to download/unzip if missing (for completeness, assuming this is common in PyTorch examples)
        try:
            download_url = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
            print(f"[INFO] Attempting to download data from {download_url}")
            r = requests.get(download_url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            # Extract to the parent directory to match the DATA_DIR path
            z.extractall(os.path.dirname(os.path.dirname(DATA_DIR)))
            print("[INFO] Downloaded and extracted UCI HAR Dataset.")
        except Exception as e:
            print(f"[ERROR] Failed to download or extract data: {e}. Exiting.")
            sys.exit(1)


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
    train_ds = HARSet(X_train, y_train - 1)  # labels 0..5 (UCI labels are 1-based)
    test_ds  = HARSet(X_test,  y_test  - 1)

    batch_size = 128
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=X.shape[1], hidden=128, dropout=0.2, num_classes=6).to(device)

    # ---- DoG optimizer params ----
    # eps ~ Adam epsilon scale; alpha controls initial r_eps = alpha * (1 + ||w0||_inf)
    optimizer = DoG(model.parameters(), eps=1e-8, alpha=1e-6)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 100
    train_losses, test_losses = [], []

    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        te = eval_one_epoch(model, test_loader, loss_fn, device)
        train_losses.append(tr)
        test_losses.append(te)
        print(f"Epoch {ep:03d}/{epochs} | train loss: {tr:.4f} | test loss: {te:.4f} | eta_t: {optimizer.state['current_eta']:.4e}")
    t1 = time.time()
    print(f"[INFO] Finished in {t1 - t0:.1f}s")

    # Save Epoch vs Loss to PDF
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("UCI HAR — DoG optimizer (Train/Test Loss)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("har_loss_dog.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print("[INFO] Saved loss curves to har_loss_dog.pdf")


if __name__ == "__main__":
    main()

