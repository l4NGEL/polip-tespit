"""U-Net egitimi (Kvasir-SEG train), model secimi yalnizca Kvasir val Dice ile.

Kullanim:  python scripts/train.py --epochs 60 --out checkpoints/unet_r34
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polip.data import PolypDataset, kvasir_subset  # noqa: E402
from polip.model import INPUT_SIZE, build_model  # noqa: E402


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + 1) / (denom + 1)).mean()


@torch.no_grad()
def val_dice(model, loader, device) -> float:
    model.eval()
    scores = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = (torch.sigmoid(model(x)) > 0.5).float()
        inter = (p * y).sum(dim=(1, 2, 3))
        denom = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
        scores.extend(((2 * inter) / denom.clamp(min=1e-7)).cpu().tolist())
    return float(np.mean(scores))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/unet_r34"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)

    train_ds = PolypDataset(kvasir_subset("train"), INPUT_SIZE, train=True)
    val_ds = PolypDataset(kvasir_subset("val"), INPUT_SIZE, train=False)
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                          pin_memory=True, persistent_workers=False)
    val_dl = DataLoader(val_ds, args.batch, shuffle=False, num_workers=args.workers)
    print(f"train {len(train_ds)}  val {len(val_ds)}")

    model = build_model(args.encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl),
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")

    best, log_rows = -1.0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            logits = logits.float()
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        vd = val_dice(model, val_dl, device)
        row = {"epoch": ep, "train_loss": float(np.mean(losses)), "val_dice": vd, "minutes": (time.time() - t0) / 60}
        log_rows.append(row)
        flag = ""
        if vd > best:
            best = vd
            torch.save({"encoder": args.encoder, "state_dict": model.state_dict(), "epoch": ep, "val_dice": vd},
                       args.out / "best.pt")
            flag = "  *"
        print(f"ep {ep:3d}  loss {row['train_loss']:.4f}  val_dice {vd:.4f}  {row['minutes']:.1f} dk{flag}", flush=True)

    with open(args.out / "train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0]))
        w.writeheader()
        w.writerows(log_rows)
    (args.out / "config.json").write_text(json.dumps({**vars(args), "out": str(args.out), "best_val_dice": best}, indent=1))
    print(f"en iyi val Dice {best:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
