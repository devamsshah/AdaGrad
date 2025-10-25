#!/usr/bin/env python3
"""
sweep_dog.py

Runs DoG_with_momentum.py over 20 “linspace” settings per param
(α, ε in log space; β in linear space), zipped to 20 total runs.
Records the parameters that give the least test loss.
"""

import os
import csv
import json
import math
import subprocess
import sys
import numpy as np
from pathlib import Path

PYTHON = sys.executable  # use current interpreter
SCRIPT = "DoG_with_momentum.py"  # path to your training script

sweep, command, info, res = True, False, False, True
# Reasonable sweep ranges (tweak if you like)
ALPHA_MIN, ALPHA_MAX = 1e-5, 1e-2
EPS_MIN,   EPS_MAX   = 1e-6, 1e-3
BETA_MIN,  BETA_MAX  = 0.0, 0.99

NUM = 10
EPOCHS = 50  # fewer epochs for sweep; adjust as needed
BATCH_SIZE = 128
HIDDEN = 128
DROPOUT = 0.2
SEED = 55 
DEVICE = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES") is not None) else "cpu"

def linspace(start, end, num):
    if num == 1: return [start]
    step = (end - start) / (num - 1)
    return [start + i * step for i in range(num)]

def log_linspace(start, end, num):
    # linear in log10
    log_start, log_end = math.log10(start), math.log10(end)
    logs = linspace(log_start, log_end, num)
    return [10 ** v for v in logs]

def main():
    alphas = log_linspace(ALPHA_MIN, ALPHA_MAX, NUM)
    epses  = log_linspace(EPS_MIN, EPS_MAX, NUM)
    betas  = linspace(BETA_MIN, BETA_MAX, NUM)

    results = []
    best = {"min_test_loss": float("inf")}

    out_dir = Path("sweep_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep_log.csv"
    best_json_path = out_dir / "best_result.json"

    for i, (a, e, b) in enumerate(zip(*[x.ravel() for x in np.meshgrid(alphas, epses, betas, indexing='ij')]),start=1):
        run_tag = f"sweep{i:02d}"
        cmd = [
            PYTHON, SCRIPT,
            "--alpha", str(a),
            "--eps", str(e),
            "--beta", str(b),
            "--epochs", str(EPOCHS),
            "--batch-size", str(BATCH_SIZE),
            "--hidden", str(HIDDEN),
            "--dropout", str(DROPOUT),
            "--seed", str(SEED),
            "--device", DEVICE,
            "--run-name", run_tag,
        ]
        if command:
            print(f"[COMMAND] {cmd}")
        if sweep:
            print(f"[SWEEP] Running {i}/{NUM}: alpha={a:.3e}, eps={e:.3e}, beta={b:.3f}")

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        stdout = proc.stdout
        if res:
            print(f"[RESULT] min error for run {i}: {stdout}")
        # Save full log for each run (optional but useful)
        (out_dir / f"log_{run_tag}.txt").write_text(stdout)

        # Parse MIN_LOSS:<value> from the last lines
        min_loss = None
        for line in stdout.strip().splitlines()[::-1]:
            if line.startswith("MIN_LOSS:"):
                try:
                    min_loss = float(line.split(":", 1)[1].strip())
                except:
                    pass
                break

        if min_loss is None:
            print(f"[WARN] Could not parse MIN_LOSS from run {run_tag}. Skipping scoring.")
            continue

        row = {
            "run": run_tag,
            "alpha": a,
            "eps": e,
            "beta": b,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "hidden": HIDDEN,
            "dropout": DROPOUT,
            "seed": SEED,
            "device": DEVICE,
            "min_test_loss": min_loss,
        }
        results.append(row)
        if min_loss < best["min_test_loss"]:
            best = row.copy()
            print(f"[SWEEP] New best: loss={min_loss:.6f} at (alpha={a:.3e}, eps={e:.3e}, beta={b:.3f})")

    # Write CSV of all runs
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)
        print(f"[SWEEP] Wrote {csv_path}")

        # Write best JSON
        with open(best_json_path, "w") as f:
            json.dump(best, f, indent=2)
        print(f"[SWEEP] Best result -> {best_json_path}")
    else:
        print("[SWEEP] No successful runs recorded.")

if __name__ == "__main__":
    main()
