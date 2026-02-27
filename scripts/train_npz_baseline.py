from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ProtonDoseDataset


class ResidualCNN3D(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_val = max(1, int(n * val_fraction)) if n > 1 else 0
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    if not train_idx:
        train_idx = val_idx
    return train_idx, val_idx


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    criterion = nn.L1Loss()

    loss_sum = 0.0
    n_batches = 0
    autocast_enabled = device.type == "cuda"

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
            y_hat = model(x)
            loss = criterion(y_hat, y)

        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        loss_sum += float(loss.item())
        n_batches += 1

    return loss_sum / max(1, n_batches)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento baseline 3D para NPZ de proton dose")
    parser.add_argument("--npz-dir", type=Path, required=True)
    parser.add_argument("--predict-residual", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/train_npz_baseline"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = ProtonDoseDataset(args.npz_dir, predict_residual=args.predict_residual)
    train_idx, val_idx = split_indices(len(dataset), args.val_fraction, args.seed)

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx) if val_idx else Subset(dataset, train_idx[:1])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualCNN3D(in_channels=3, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else None

    run_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    best_val = float("inf")
    with metrics_path.open("w", encoding="utf-8") as mf:
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, optimizer, device, scaler)
            with torch.no_grad():
                val_loss = run_epoch(model, val_loader, None, device, None)

            row = {"epoch": epoch, "train_l1": train_loss, "val_l1": val_loss}
            mf.write(json.dumps(row) + "\n")
            mf.flush()

            print(f"Epoch {epoch:03d} | train_l1={train_loss:.6f} | val_l1={val_loss:.6f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_val_l1": best_val,
                        "args": vars(args),
                    },
                    run_dir / "best.pt",
                )

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": args.epochs,
            "best_val_l1": best_val,
            "args": vars(args),
        },
        run_dir / "last.pt",
    )

    print(f"Run dir: {run_dir}")
    print(f"Best val L1: {best_val:.6f}")


if __name__ == "__main__":
    main()
