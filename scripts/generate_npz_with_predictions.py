#!/usr/bin/env python3
"""
Generate NPZ file with predictions for analysis.
Creates a new NPZ with structure: d_low, spr, d_high (ground truth), d_pred (model prediction).
Usage:
  python scripts/generate_npz_with_predictions.py --checkpoint best.pt --npz input.npz --out pred_output.npz
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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


def generate_prediction_npz(checkpoint_path: str, npz_path: str, out_path: str, crop_size: tuple = (96, 96, 96)):
    """Generate NPZ with predictions."""
    
    ensure_safe_runtime_dirs()
    
    print(f"Loading NPZ: {npz_path}")
    case = load_case_npz(npz_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    
    # Load original data
    d_low = case.d_low
    d_high = case.d_high
    spr = case.spr
    beam_mask = case.beam_mask if case.beam_mask is not None else np.ones_like(d_low, dtype=np.float32)
    
    print(f"Original shape: {d_low.shape}")
    
    # Crop data for inference
    d_low_crop, spr_crop, d_high_crop = maybe_crop_bev(d_low, spr, d_high, crop_size=crop_size, enabled=True)
    beam_mask_crop = maybe_crop_bev(beam_mask, beam_mask, beam_mask, crop_size=crop_size, enabled=True)[0]
    
    print(f"Cropped shape: {d_low_crop.shape}")
    
    # Prepare input for model
    e0_map = np.full_like(d_low_crop, fill_value=float(case.e0_mev), dtype=np.float32)
    x = np.stack([d_low_crop, spr_crop, e0_map, beam_mask_crop.astype(np.float32)], axis=0).astype(np.float32)
    x_tensor = torch.from_numpy(x[None, ...]).to(device)
    d_low_tensor = torch.from_numpy(d_low_crop[None, None, ...].astype(np.float32)).to(device)
    
    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "args" in checkpoint and "base_channels" in checkpoint["args"]:
        base_channels = checkpoint["args"]["base_channels"]
    else:
        base_channels = 24
    
    model = ResidualUNet3D(in_channels=4, base_channels=base_channels, residual=True).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"Checkpoint: epoch {checkpoint.get('epoch', '?')}, best_val_l1={checkpoint.get('best_val_l1', '?'):.6f}")
    
    # Generate prediction
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
            delta = model(x_tensor)
            pred_tensor = d_low_tensor + delta
    
    pred = pred_tensor[0, 0].cpu().numpy()
    
    print(f"Prediction shape: {pred.shape}")
    print(f"Prediction range: [{pred.min():.6f}, {pred.max():.6f}]")
    print(f"GT (cropped) range: [{d_high_crop.min():.6f}, {d_high_crop.max():.6f}]")
    
    # Create output NPZ
    output_dict = {
        "d_low": d_low_crop.astype(np.float32),
        "spr": spr_crop.astype(np.float32),
        "d_high": d_high_crop.astype(np.float32),  # Ground truth
        "d_pred": pred.astype(np.float32),  # Model prediction
        "beam_mask": beam_mask_crop.astype(np.float32),
        "e0_mev": np.float32(case.e0_mev),
        "case_id": case.case_id,
    }
    
    # Save
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **output_dict)
    
    print(f"\n✓ Saved: {out_path}")
    
    # Print stats
    diff = pred - d_high_crop
    mask = d_high_crop > 0.01
    if np.sum(mask) > 0:
        print(f"\nStatistics:")
        print(f"  Mean absolute error: {np.mean(np.abs(diff[mask])):.6f}")
        print(f"  Max absolute error:  {np.max(np.abs(diff[mask])):.6f}")
        print(f"  Mean error (signed): {np.mean(diff[mask]):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NPZ with model predictions")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--npz", type=Path, required=True, help="Path to input NPZ file")
    parser.add_argument("--out", type=Path, required=True, help="Path to output NPZ file")
    parser.add_argument("--crop-size", type=str, default="96,96,96", help="Crop size D,H,W")
    args = parser.parse_args()
    
    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    if not args.npz.exists():
        print(f"ERROR: NPZ not found: {args.npz}")
        sys.exit(1)
    
    crop_tokens = [int(x.strip()) for x in args.crop_size.split(",") if x.strip()]
    if len(crop_tokens) != 3:
        raise ValueError("--crop-size debe tener formato D,H,W")
    crop_size = (crop_tokens[0], crop_tokens[1], crop_tokens[2])
    
    generate_prediction_npz(str(args.checkpoint), str(args.npz), str(args.out), crop_size)


if __name__ == "__main__":
    main()
