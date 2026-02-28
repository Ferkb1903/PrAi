#!/usr/bin/env python
"""
Evaluate trained model on test/val set and report metrics.
Usage:
  python scripts/evaluate_model.py --checkpoint best_model_epoch4.pt --manifest data/training_npz/manifest_test.csv
"""

import argparse
import csv
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io_npz import load_case_npz
from src.data.preprocess import maybe_crop_bev
from src.model.resunet3d import ResidualUNet3D


class ManifestNPZDataset(Dataset):
    def __init__(self, manifest_csv: Path, use_bev_crop: bool = True, crop_size: tuple[int, int, int] = (96, 96, 96)) -> None:
        rows = list(csv.DictReader(manifest_csv.open(encoding="utf-8")))
        self.paths = [Path(r["npz_path"]) for r in rows]
        self.manifest_dir = manifest_csv.parent.resolve()
        self.use_bev_crop = use_bev_crop
        self.crop_size = crop_size
        self.bad_paths_reported: set[Path] = set()
        
        if not self.paths:
            raise ValueError(f"Manifest vacío: {manifest_csv}")
        
        self.paths = self._filter_invalid_paths(self.paths)
        if not self.paths:
            raise ValueError(f"No hay NPZ válidos tras filtrar: {manifest_csv}")

    def _filter_invalid_paths(self, paths: list[Path]) -> list[Path]:
        valid: list[Path] = []
        for path in paths:
            resolved = self._resolve_npz_path(path, raise_if_missing=False)
            if resolved is None or not zipfile.is_zipfile(resolved):
                continue
            valid.append(resolved)
        return valid

    def _resolve_npz_path(self, path: Path, raise_if_missing: bool = True) -> Path | None:
        if path.exists():
            return path

        candidates: list[Path] = [
            self.manifest_dir / path.name,
        ]
        if not path.is_absolute():
            candidates.append((self.manifest_dir / path).resolve())

        parts = list(path.parts)
        filtered_parts = [p for p in parts if not p.startswith("chunk_")]
        if filtered_parts != parts:
            try:
                candidates.append(Path(*filtered_parts))
            except TypeError:
                pass

        for candidate in candidates:
            if candidate.exists():
                return candidate

        if raise_if_missing:
            raise FileNotFoundError(f"NPZ not found: {path}")
        return None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        n = len(self.paths)
        last_error: Exception | None = None

        for attempt in range(3):
            real_idx = (idx + attempt) % n
            npz_path = self.paths[real_idx]
            try:
                case = load_case_npz(npz_path)
                break
            except Exception as exc:
                last_error = exc
                if npz_path not in self.bad_paths_reported:
                    print(f"[Dataset] Saltando NPZ: {npz_path}")
                    self.bad_paths_reported.add(npz_path)
                continue
        else:
            raise RuntimeError(f"No se pudo cargar muestra tras 3 intentos: {last_error}")

        d_low = case.d_low
        spr = case.spr
        d_high = case.d_high
        beam_mask = case.beam_mask if case.beam_mask is not None else np.ones_like(d_low, dtype=np.float32)

        if self.use_bev_crop:
            d_low, spr, d_high = maybe_crop_bev(d_low, spr, d_high, crop_size=self.crop_size, enabled=True)
            beam_mask = maybe_crop_bev(beam_mask, beam_mask, beam_mask, crop_size=self.crop_size, enabled=True)[0]

        e0_map = np.full_like(d_low, fill_value=float(case.e0_mev), dtype=np.float32)
        x = np.stack([d_low, spr, e0_map, beam_mask.astype(np.float32)], axis=0).astype(np.float32)

        return {
            "x": torch.from_numpy(x),
            "d_low": torch.from_numpy(d_low[None, ...].astype(np.float32)),
            "target": torch.from_numpy(d_high[None, ...].astype(np.float32)),
        }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, list[float]]:
    model.eval()
    criterion = nn.L1Loss()
    
    loss_sum = 0.0
    losses = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            x = batch["x"].to(device, non_blocking=True)
            d_low = batch["d_low"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            
            try:
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
                    delta = model(x)
                    pred = d_low + delta
                    loss = criterion(pred, target)
                
                loss_sum += float(loss.item())
                losses.append(float(loss.item()))
            except Exception as exc:
                print(f"[!] Error en batch: {exc}")
                continue
    
    mean_loss = loss_sum / max(1, len(losses))
    return mean_loss, losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model checkpoint on test/val set")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest CSV (test/val)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--crop-size", type=str, default="96,96,96")
    parser.add_argument("--no-bev-crop", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not args.manifest.exists():
        print(f"ERROR: Manifest not found: {args.manifest}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    crop_tokens = [int(x.strip()) for x in args.crop_size.split(",") if x.strip()]
    if len(crop_tokens) != 3:
        raise ValueError("--crop-size debe tener formato D,H,W")
    crop_size = (crop_tokens[0], crop_tokens[1], crop_tokens[2])

    # Load model
    model = ResidualUNet3D(in_channels=4, base_channels=args.base_channels, residual=True).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(f"[Checkpoint] Loaded: {args.checkpoint}")
    print(f"[Checkpoint] Trained until epoch {checkpoint.get('epoch', '?')}")
    print(f"[Checkpoint] best_val_l1: {checkpoint.get('best_val_l1', '?'):.6f}")

    # Load dataset
    dataset = ManifestNPZDataset(args.manifest, use_bev_crop=not args.no_bev_crop, crop_size=crop_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"[Dataset] {len(dataset)} samples from {args.manifest.name}")

    # Evaluate
    mean_l1, losses = evaluate(model, loader, device)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Mean L1 Loss: {mean_l1:.6f}")
    print(f"Min L1:       {min(losses):.6f}")
    print(f"Max L1:       {max(losses):.6f}")
    print(f"Std Dev:      {np.std(losses):.6f}")
    print(f"Samples:      {len(losses)}")
    print("="*60)


if __name__ == "__main__":
    main()
