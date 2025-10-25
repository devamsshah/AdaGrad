import time
import torch
from torch import nn

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    running, n = 0.0, 0
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
    running_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        running_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        n += xb.size(0)
    return running_loss / n, correct / n

def train_eval(model, train_loader, test_loader, optimizer, args):
    device = torch.device(args.device)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    epochs = args.epochs
    train_losses, test_losses = [], []
    best_test_acc, best_state = 0.0, None
    min_test_loss = float("inf")

    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        te_loss, te_acc = eval_one_epoch(model, test_loader, loss_fn, device)
        train_losses.append(tr)
        test_losses.append(te_loss)
        if te_loss < min_test_loss:
            min_test_loss = te_loss
        if te_acc > best_test_acc:
            best_test_acc = te_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    t1 = time.time()

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "min_test_loss": float(min_test_loss),
        "best_test_acc": float(best_test_acc),
        "wall_time_sec": float(t1 - t0),
    }
    history = {"train_losses": train_losses, "test_losses": test_losses}
    return history, metrics, best_state


