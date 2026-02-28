#!/usr/bin/env python
"""
Generate predictions on test cases with visualizations.
Usage:
  python scripts/generate_predictions.py --checkpoint best.pt --manifest data/training_npz/spot_campaign_v2_parallel/manifest_test.csv --n-cases 5
"""

import argparse
import csv
import os
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
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


def ensure_safe_runtime_dirs() -> None:
    user = os.environ.get("USER", "user")
    base_tmp = Path(f"/tmp/miopen_cache_{user}")
    base_tmp.mkdir(parents=True, exist_ok=True)
    os.chmod(base_tmp, 0o700)

    tmpdir = os.environ.get("TMPDIR", "").strip()
    if not tmpdir or not Path(tmpdir).is_dir():
        os.environ["TMPDIR"] = str(base_tmp)

    miopen_db = base_tmp / "miopen_db"
    miopen_db.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MIOPEN_USER_DB_PATH", str(miopen_db))
    os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", str(base_tmp))


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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | Path]:
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
            "npz_path": str(npz_path),
        }


def visualize_case(
    case_idx: int,
    d_low: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    npz_name: str,
    out_dir: Path,
) -> None:
    """Save visualization of a single case: D_low, Target, Pred, Error."""
    
    # Select middle slice
    d = d_low.shape[0]
    mid_z = d // 2
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # D_low
    ax = axes[0, 0]
    im = ax.imshow(d_low[mid_z], cmap="viridis")
    ax.set_title("D_low (input)")
    plt.colorbar(im, ax=ax)
    
    # Ground truth
    ax = axes[0, 1]
    im = ax.imshow(target[mid_z], cmap="viridis")
    ax.set_title("Ground Truth (D_high)")
    plt.colorbar(im, ax=ax)
    
    # Prediction
    ax = axes[1, 0]
    im = ax.imshow(pred[mid_z], cmap="viridis")
    ax.set_title("Prediction")
    plt.colorbar(im, ax=ax)
    
    # Error
    err = np.abs(pred - target)
    ax = axes[1, 1]
    im = ax.imshow(err[mid_z], cmap="hot")
    ax.set_title(f"Absolute Error (mean={err.mean():.6f})")
    plt.colorbar(im, ax=ax)
    
    plt.suptitle(f"Case {case_idx}: {npz_name}")
    save_path = out_dir / f"pred_{case_idx:03d}_{Path(npz_name).stem}.png"
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {save_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions with visualizations")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest CSV (test/val)")
    parser.add_argument("--n-cases", type=int, default=5, help="Number of cases to predict")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--crop-size", type=str, default="96,96,96")
    parser.add_argument("--no-bev-crop", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("predictions"))
    args = parser.parse_args()

    ensure_safe_runtime_dirs()
    print(f"Runtime TMPDIR: {os.environ.get('TMPDIR')}")

    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not args.manifest.exists():
        print(f"ERROR: Manifest not found: {args.manifest}")
        sys.exit(1)

    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    crop_tokens = [int(x.strip()) for x in args.crop_size.split(",") if x.strip()]
    if len(crop_tokens) != 3:
        raise ValueError("--crop-size debe tener formato D,H,W")
    crop_size = (crop_tokens[0], crop_tokens[1], crop_tokens[2])

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if "args" in checkpoint and "base_channels" in checkpoint["args"]:
        base_channels = checkpoint["args"]["base_channels"]
    else:
        base_channels = args.base_channels
    
    model = ResidualUNet3D(in_channels=4, base_channels=base_channels, residual=True).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"[Checkpoint] Loaded: {args.checkpoint} (epoch {checkpoint.get('epoch', '?')})")

    # Load dataset
    dataset = ManifestNPZDataset(args.manifest, use_bev_crop=not args.no_bev_crop, crop_size=crop_size)
    n_cases = min(args.n_cases, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"[Dataset] {len(dataset)} total samples; predicting on {n_cases}")

    # Generate predictions
    case_count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            if case_count >= n_cases:
                break

            x = batch["x"].to(device, non_blocking=True)
            d_low = batch["d_low"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            npz_path = batch["npz_path"][0]

            try:
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
                    delta = model(x)
                    pred = d_low + delta

                # Move to CPU & numpy
                d_low_np = d_low[0, 0].cpu().numpy()
                target_np = target[0, 0].cpu().numpy()
                pred_np = pred[0, 0].cpu().numpy()

                # Visualize
                npz_name = Path(npz_path).name
                visualize_case(case_count, d_low_np, target_np, pred_np, npz_name, args.out_dir)
                
                case_count += 1
            except Exception as exc:
                print(f"[!] Error on case {case_count}: {exc}")
                continue

    print(f"\n[Done] Predictions saved to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
