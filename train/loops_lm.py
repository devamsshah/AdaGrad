# train/loops_lm.py
import time
import torch

def train_eval_lm(model, train_loader, val_loader, optimizer, args):
    """
    Language-model training loop.
    - Matches classification loop history keys so utils.plot.make_run_artifacts works:
        history["train_losses"], history["test_losses"]
    - Emits metrics fields expected by artifact code:
        metrics["min_test_loss"], metrics["best_test_acc"]
    - Prints MIN_LOSS / MAX_ACCY lines for sweep parsers.
    """
    device = torch.device(args.device)
    model.to(device)

    use_cuda = (args.device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    history = {"train_losses": [], "test_losses": []}
    best_state = None
    best_val = float("inf")

    for ep in range(1, args.epochs + 1):
        # ---------- train ----------
        model.train()
        running, n = 0.0, 0
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                _, loss = model(x, y)  # model returns (logits, loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
            n += x.size(0)
        tr_loss = running / max(1, n)
        history["train_losses"].append(tr_loss)

        # ---------- eval (use val loader but name 'test' to match plot util) ----------
        model.eval()
        running, n = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast("cuda", enabled=use_cuda):
                    _, loss = model(x, y)
                running += loss.item() * x.size(0)
                n += x.size(0)
        te_loss = running / max(1, n)
        history["test_losses"].append(te_loss)

        if te_loss < best_val:
            best_val = te_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if getattr(args, "verbose", False):
            print(f"Epoch {ep:03d}/{args.epochs} | train {tr_loss:.4f} | val {te_loss:.4f} | {time.time()-t0:.1f}s")

    # Metrics compatible with your artifact writer
    metrics = {
        "min_test_loss": float(best_val),
        "best_test_acc": 0.0,  # LM has no accuracy; keep key for uniformity
        "best_val_ppl": float(torch.exp(torch.tensor(best_val)).item()),
    }

    # Lines consumed by sweeps
    print(f"MIN_LOSS:{best_val:.8f}")
    print(f"MAX_ACCY:{0.0:.8f}")

    return history, metrics, best_state

