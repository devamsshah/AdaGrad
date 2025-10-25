import os
import re
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader

class HARSet(Dataset):
    def __init__(self, X: np.ndarray, y_zero_based: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_zero_based, dtype=torch.long)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

def _load_ucihar_all(data_dir: str) -> Tuple[np.ndarray, np.ndarray, list]:
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
    y_all = np.concatenate([y_tr, y_te]).astype(np.int64)
    act_df = pd.read_csv(labels, sep=r"\s+", header=None, names=["id", "name"]).sort_values("id")
    label_names = act_df["name"].str.replace("_", " ").tolist()
    return X_all, y_all, label_names

def load_and_make_loaders(args):
    X, y, labels = _load_ucihar_all(args.data_dir)

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

    meta = {
        "input_dim": X.shape[1],
        "num_classes": 6,
        "label_names": labels
    }
    return train_loader, test_loader, meta

