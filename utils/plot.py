import os
import re
import json
import matplotlib.pyplot as plt

def _slug_float(x: float) -> str:
    return re.sub(r"[^0-9a-zA-Z\\-\\._]", "", f"{x:.1e}")

def _slug(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z\\-\\._]", "_", s)[:50]

def make_run_artifacts(args, history, metrics):
    pdf_name = (
        f"har_loss_{args.dataset}_{args.optimizer}"
        f"_a={_slug_float(getattr(args,'alpha',1.0))}"
        f"_e={_slug_float(getattr(args,'eps',1e-8))}"
        f"_b={_slug_float(getattr(args,'beta',0.9))}"
        f"_hid={args.hidden}"
        f"_do={_slug_float(args.dropout)}"
        f"_bs={args.batch_size}"
        f"_seed={args.seed}"
        + (f"_run={_slug(args.run_name)}" if args.run_name else "")
        + ".pdf"
    )

    epochs = range(1, len(history["train_losses"]) + 1)
    plt.figure()
    plt.plot(epochs, history["train_losses"], label="Train Loss")
    plt.plot(epochs, history["test_losses"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{args.dataset} — {args.optimizer} (Train/Test Loss)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pdf_name, format="pdf", bbox_inches="tight")
    plt.close()

    result = {
        **metrics,
        "alpha": getattr(args, "alpha", None),
        "eps": getattr(args, "eps", None),
        "beta": getattr(args, "beta", None),
        "hidden": args.hidden,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "pdf": pdf_name,
        "run_name": args.run_name,
        "dataset": args.dataset,
        "optimizer": args.optimizer,
        "model": args.model,
    }
    out_json = os.path.splitext(pdf_name)[0] + ".json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    # machine-parsable lines (compatible with your sweeper)
    print(f"MIN_LOSS:{metrics['min_test_loss']:.8f}")
    print(f"MAX_ACCY:{metrics['best_test_acc']:.8f}")

    return pdf_name, out_json


