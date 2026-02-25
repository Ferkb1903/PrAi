import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import defaults
from src.data.dataset import ProtonDoseDataset
from src.losses.losses import total_loss
from src.models.model_factory import build_model


def set_seed(seed: int) -> None:
    """Fija semillas para mejorar reproducibilidad entre corridas."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    """Selecciona GPU si está disponible y fue pedida en config."""
    if defaults.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def train(dry_run: bool = False) -> None:
    """Entrena el modelo y guarda checkpoints por época.

    En modo dry-run ejecuta una sola época para validar pipeline.
    """

    set_seed(defaults.SEED)
    device = resolve_device()

    train_ds = ProtonDoseDataset(defaults.TRAIN_DIR, predict_residual=defaults.PREDICT_RESIDUAL)
    val_ds = ProtonDoseDataset(defaults.VAL_DIR, predict_residual=defaults.PREDICT_RESIDUAL)

    train_loader = DataLoader(
        train_ds,
        batch_size=defaults.BATCH_SIZE,
        shuffle=True,
        num_workers=defaults.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=defaults.NUM_WORKERS,
    )

    model = build_model().to(device)
    optimizer = AdamW(model.parameters(), lr=defaults.LEARNING_RATE, weight_decay=defaults.WEIGHT_DECAY)

    epochs = 1 if dry_run else defaults.EPOCHS

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        # Bucle de entrenamiento: forward -> loss -> backward -> step.
        for batch in tqdm(train_loader, desc=f"train epoch {epoch}"):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            beam_axis = batch["beam_axis"].to(device)

            optimizer.zero_grad()
            pred = model(x)
            losses = total_loss(pred, y, beam_axis)
            losses["total"].backward()
            optimizer.step()

            train_loss += float(losses["total"].item())

        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0

        # Validación sin gradiente para medir generalización por época.
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                beam_axis = batch["beam_axis"].to(device)

                pred = model(x)
                losses = total_loss(pred, y, beam_axis)
                val_loss += float(losses["total"].item())

        val_loss /= max(len(val_loader), 1)

        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if defaults.SAVE_EVERY_EPOCH:
            ckpt_path = defaults.CHECKPOINT_DIR / f"epoch_{epoch:03d}.pt"
            save_checkpoint(model, optimizer, epoch, ckpt_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Proton Dose Denoiser")
    parser.add_argument("--dry-run", action="store_true", help="Ejecuta 1 época para smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(dry_run=args.dry_run)
