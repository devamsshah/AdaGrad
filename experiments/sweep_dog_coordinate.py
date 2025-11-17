"""
Sweep script for Coordinate-wise DoG optimizer (Algorithm 1)

- Dataset: MNIST
- Model:   CNN
- Optimizer: dog_coordinate
- Fixed:   epochs=5, seed=55, cnn-width default, reps-rel fixed (uses default alpha)
- Sweeps:  alpha, eps, weight-decay, init-eta
- Logs:    results/sweep_results/sweep_log.csv
- Saves:   best run summary as results/sweep_results/best_result.json
"""

import os
import sys
import csv
import json
import subprocess
from itertools import product

# ------------------------
# Constants / Defaults
# ------------------------
DATA_DIR = "./datasets/MNIST/raw"      # adjust if raw IDX files are inside ./datasets/MNIST/raw
EPOCHS = 2 
SEED = 55
MODEL = "cnn"
OPTIMIZER = "dog_coordinate"
RUN_PREFIX = "sweep_dogcoord_cnn_mnist"

RESULTS_DIR = "experiments/results/sweep_results"
BEST_JSON = os.path.join(RESULTS_DIR, "best_result.json")
CSV_PATH = os.path.join(RESULTS_DIR, "sweep_log.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------
# Sweep grids (modify freely)
# ------------------------
ALPHA_VALUES = [1e-8, 3e-8, 7e-8, 1e-7, 3e-7, 7e-7, 1e-6, 3e-6, 7e-6]
EPS_VALUES   = [1e-10, 5e-9, 1e-9, 5e-8, 1e-8, 5e-7, 1e-7, 5e-6, 1e-6]
WD_VALUES    = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
INITETA_VALUES = [None, 1e-6, 1e-4]  # optional override of init_eta

# ------------------------
# Core helpers
# ------------------------
def run_one(alpha, eps, wd, init_eta, idx):
    run_name = f"{RUN_PREFIX}_a={alpha:.1e}_e={eps:.1e}_wd={wd:.1e}_ie={init_eta if init_eta else 'None'}_i={idx}"
    cmd = [
        sys.executable, "-m", "experiments.main",
        "--dataset", "mnist",
        "--model", MODEL,
        "--optimizer", OPTIMIZER,
        "--data-dir", DATA_DIR,
        "--epochs", str(EPOCHS),
        "--batch-size", "12",
        "--alpha", str(alpha),
        "--eps", str(eps),
        "--weight-decay", str(wd),
        "--seed", str(SEED),
        "--run-name", run_name,
    ]
    if init_eta is not None:
        cmd += ["--init-eta", str(init_eta)]

    print(">>>", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout
    print(out)

    min_loss, max_accy = None, None
    for line in out.strip().splitlines():
        line = line.strip()
        if line.startswith("MIN_LOSS:"):
            try:
                min_loss = float(line.split("MIN_LOSS:")[1].strip())
            except Exception:
                pass
        elif line.startswith("MAX_ACCY:"):
            try:
                max_accy = float(line.split("MAX_ACCY:")[1].strip())
            except Exception:
                pass
    return run_name, min_loss, max_accy

# ------------------------
# Sweep loop
# ------------------------
def main():
    best = {"acc": -1.0, "row": None}
    rows = []
    combos = list(product(ALPHA_VALUES, EPS_VALUES, WD_VALUES, INITETA_VALUES))
    print(f"Total runs: {len(combos)}\n")

    for i, (alpha, eps, wd, init_eta) in enumerate(combos, 1):
        run_name, min_loss, max_accy = run_one(alpha, eps, wd, init_eta, i)
        row = {
            "idx": i,
            "run_name": run_name,
            "alpha": alpha,
            "eps": eps,
            "weight_decay": wd,
            "init_eta": init_eta if init_eta is not None else "",
            "epochs": EPOCHS,
            "batch_size": 12,
            "seed": SEED,
            "min_loss": min_loss,
            "max_accy": max_accy,
        }
        rows.append(row)

        # update best
        if isinstance(max_accy, float) and max_accy > best["acc"]:
            best["acc"] = max_accy
            best["row"] = row

        # append to CSV
        write_header = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)

    # ------------------------
    # Write best summary
    # ------------------------
    if best["row"]:
        with open(BEST_JSON, "w") as f:
            json.dump(best["row"], f, indent=2)
        print(f"\n[BEST RUN] #{best['row']['idx']} | "
              f"acc={best['acc']:.6f} | "
              f"alpha={best['row']['alpha']}, eps={best['row']['eps']}, "
              f"weight_decay={best['row']['weight_decay']}, init_eta={best['row']['init_eta']}")
        print(f"[Saved] {BEST_JSON}")
    else:
        print("No valid results parsed.")

if __name__ == "__main__":
    main()

