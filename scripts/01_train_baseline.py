"""01 — Train the baseline CNN on CIFAR-10 and save the best checkpoint.

Usage:
  python scripts/01_train_baseline.py                          # baseline (seed 42)
  python scripts/01_train_baseline.py --seed 7 --tag member_1 --epochs 20 --cpu
"""
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src import config
from src.data import get_clean_datasets
from src.models import CIFAR10CNN


def get_transforms():
    from torchvision import transforms
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--tag", type=str, default="baseline_cnn")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--init", type=str, default=None,
                   help="checkpoint to warm-start from (adds member diversity "
                        "while guaranteeing fast convergence)")
    return p.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)
    device = config.get_device(args.cpu)
    out_path = os.path.join(config.CHECKPOINT_DIR, f"{args.tag}.pt")
    print(f"[train] tag={args.tag} seed={args.seed} device={device} "
          f"epochs={args.epochs}", flush=True)

    train_raw, cal_raw, test_raw = get_clean_datasets()
    train_ds = Subset(train_raw, list(range(len(train_raw))))

    # augmentation applied per-batch
    transform = get_transforms()

    def collate(batch):
        xs = torch.stack([transform(t) for t, _ in batch])
        ys = torch.tensor([lbl for _, lbl in batch])
        return xs, ys

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=True, collate_fn=collate,
    )
    cal_loader = DataLoader(cal_raw, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_raw, batch_size=512, shuffle=False)

    model = CIFAR10CNN(config.DROPOUT_P).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
        print(f"[train] warm-started from {args.init}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=config.LR,
                            weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "cal_acc": [], "test_acc": []}
    best_cal_acc, best_state = -1.0, None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_correct, n_seen = 0.0, 0, 0
        for xs, ys in train_loader:
            xs, ys = xs.to(device), ys.to(device)
            opt.zero_grad()
            out = model(xs)
            loss = loss_fn(out, ys)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(ys)
            n_correct += (out.argmax(1) == ys).sum().item()
            n_seen += len(ys)
        sched.step()

        cal_acc = eval_acc(model, cal_loader, device)
        test_acc = eval_acc(model, test_loader, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(total_loss / n_seen)
        history["train_acc"].append(n_correct / n_seen)
        history["cal_acc"].append(cal_acc)
        history["test_acc"].append(test_acc)

        if cal_acc > best_cal_acc:
            best_cal_acc = cal_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0:
            snap = os.path.join(config.CHECKPOINT_DIR, f"{args.tag}_ep{epoch:02d}.pt")
            torch.save(best_state, snap)
            print(f"[train] snapshot -> {snap}", flush=True)

        print(f"[train] epoch {epoch:02d}/{args.epochs}  "
              f"loss={history['train_loss'][-1]:.4f}  "
              f"train_acc={history['train_acc'][-1]:.4f}  "
              f"cal_acc={cal_acc:.4f}  test_acc={test_acc:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    torch.save(best_state, out_path)
    hist_path = os.path.join(config.RESULT_DIR, f"train_history_{args.tag}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    make_history_plot(history, args.tag)
    print(f"[train] done for '{args.tag}'. best cal_acc={best_cal_acc:.4f}, "
          f"best test_acc={max(history['test_acc']):.4f}", flush=True)
    print(f"[train] checkpoint -> {out_path}", flush=True)


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    n_correct = 0
    for xs, ys in loader:
        n_correct += (model(xs.to(device)).argmax(1).cpu() == ys).sum().item()
    return n_correct / len(loader.dataset)


def make_history_plot(history, tag="baseline_cnn"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(history["epoch"], history["train_loss"], label="train loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Training loss")
    axes[1].plot(history["epoch"], history["cal_acc"], label="calibration acc", marker="o")
    axes[1].plot(history["epoch"], history["test_acc"], label="test acc", marker="o")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].set_title("Accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIGURE_DIR, f"training_history_{tag}.png"), dpi=110)
    plt.close(fig)


def seed_all(seed=config.SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    main()