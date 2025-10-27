# experiments/sweep_dog.py
import csv
import os
import subprocess
import sys
from itertools import product

# -----------------------------
# Sweep config (edit ranges here)
# -----------------------------
DATA_DIR       = "./datasets/MNIST/raw"   # point this to your MNIST IDX folder (or where torchvision put raw)
EPOCHS         = 10
BATCH_SIZE     = 12
SEED           = 55
MODEL          = "cnn"
OPTIMIZER      = "dog"                     # coordinate-wise DoG (Algorithm 1)
DEVICE         = "cpu"                    # or "cpu"
RUN_PREFIX     = "sweep_dog_cnn_mnist"

# 7-point linspace/logspace style: use log-spaced values (common for step sizes)
ALPHA_VALUES = [1e-8, 3.16e-8, 1e-7, 3.16e-7, 1e-6, 3.16e-6, 1e-5]
EPS_VALUES   = [1e-10, 3.16e-10, 1e-9, 3.16e-9, 1e-8, 3.16e-8, 1e-7]

# Results
RESULTS_DIR = "experiments/results/sweep_results"
BEST_DIR    = "experiments/results/best_results"
CSV_PATH    = os.path.join(RESULTS_DIR, "sweep_log.csv")
BEST_JSON   = os.path.join(RESULTS_DIR, "best_result.json")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)

def run_one(alpha, eps, idx):
    run_name = f"{RUN_PREFIX}_a={alpha:.2e}_e={eps:.2e}_i={idx}"
    cmd = [
        sys.executable, "-m", "experiments.main",
        "--dataset", "mnist",
        "--model", MODEL,
        "--optimizer", OPTIMIZER,
        "--data-dir", DATA_DIR,
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--eps", f"{eps:.10g}",
        "--alpha", f"{alpha:.10g}",
        "--cnn-width", "16",
        "--dropout", "0.1",
        "--device", DEVICE,
        "--seed", str(SEED),
        "--run-name", run_name,
    ]
    print(">>>", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)

    # Parse final lines printed by make_run_artifacts: MIN_LOSS:..., MAX_ACCY:...
    min_loss, max_accy = None, None
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("MIN_LOSS:"):
            try:
                min_loss = float(line.split("MIN_LOSS:")[1].strip())
            except Exception:
                pass
        if line.startswith("MAX_ACCY:"):
            try:
                max_accy = float(line.split("MAX_ACCY:")[1].strip())
            except Exception:
                pass
    return run_name, min_loss, max_accy

def main():
    best = {"acc": -1.0, "row": None}
    rows = []

    combos = list(product(ALPHA_VALUES, EPS_VALUES))
    for i, (a, e) in enumerate(combos, 1):
        run_name, min_loss, max_accy = run_one(a, e, i)
        row = {
            "idx": i,
            "run_name": run_name,
            "alpha": f"{a:.10g}",
            "eps": f"{e:.10g}",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "min_loss": min_loss if min_loss is not None else "",
            "max_accy": max_accy if max_accy is not None else "",
        }
        rows.append(row)

        if isinstance(max_accy, float) and max_accy > best["acc"]:
            best["acc"] = max_accy
            best["row"] = row

        # Stream append to CSV
        write_header = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)

    # Save best result as JSON summary
    if best["row"] is not None:
        import json
        with open(BEST_JSON, "w") as f:
            json.dump(best["row"], f, indent=2)
        print(f"[BEST] max_accy={best['acc']:.6f} | run={best['row']['run_name']}")
        print(f"[BEST] logged at {BEST_JSON}")
    else:
        print("[BEST] No valid results parsed.")

if __name__ == "__main__":
    main()

