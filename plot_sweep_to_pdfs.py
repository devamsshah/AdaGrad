#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def to_numeric(series):
    # Convert strings like '1e-08' or '0.0' to numeric, coerce errors to NaN
    return pd.to_numeric(series, errors="coerce")


def maybe_log_scale(ax, xvalues):
    # If range spans several orders of magnitude and all positive, use log scale
    x_min = xvalues.min()
    x_max = xvalues.max()
    if pd.notna(x_min) and pd.notna(x_max) and x_min > 0 and x_max > 0:
        try:
            import math
            if x_max / x_min >= 100:  # ~2+ orders of magnitude
                ax.set_xscale("log")
        except Exception:
            pass


def plot_param_vs_min_loss(df, param, outdir):
    if param not in df.columns:
        print(f"[warn] Column '{param}' not found in CSV; skipping.")
        return None

    # Prepare numeric columns
    x = to_numeric(df[param])
    y = to_numeric(df.get("min_loss"))
    sub = pd.DataFrame({param: x, "min_loss": y}).dropna()

    if sub.empty:
        print(f"[warn] No numeric/non-NaN data for '{param}' vs min_loss; skipping.")
        return None

    # Aggregate by param value to reduce duplicate points; keep best (lowest) min_loss
    agg = sub.groupby(param, as_index=False)["min_loss"].min().sort_values(param)

    fig, ax = plt.subplots(figsize=(6, 4.25))
    ax.plot(agg[param].values, agg["min_loss"].values, marker="o")  # single chart

    ax.set_xlabel(param)
    ax.set_ylabel("min_loss")
    ax.set_title(f"{param} vs min_loss")

    maybe_log_scale(ax, agg[param].values)

    fig.tight_layout()
    outfile = Path(outdir) / f"{param}_vs_min_loss.pdf"
    fig.savefig(outfile, format="pdf")
    plt.close(fig)
    print(f"[ok] Wrote {outfile}")
    return str(outfile)


def main():
    p = argparse.ArgumentParser(description="Generate PDF plots: (alpha|eps|weight_decay) vs min_loss from sweep CSV.")
    p.add_argument("csv_path", type=str, help="Input CSV produced by the run parser.")
    p.add_argument("--outdir", type=str, default=".", help="Directory to write PDFs (default: current directory).")
    args = p.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    outputs = []
    for param in ["alpha", "eps", "weight_decay"]:
        out = plot_param_vs_min_loss(df, param, outdir)
        if out:
            outputs.append(out)

    if outputs:
        print("Generated files:")
        for o in outputs:
            print(" -", o)
    else:
        print("No plots were generated. Check that the CSV has columns: alpha, eps, weight_decay, min_loss.")


if __name__ == "__main__":
    main()
