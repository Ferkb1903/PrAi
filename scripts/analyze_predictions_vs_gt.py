#!/usr/bin/env python3
"""
Analyze predictions vs ground truth using same format as analyze_beam.py
Compares model predictions against ground truth dose (D_high).
Usage: python analyze_predictions_vs_gt.py --checkpoint best.pt --npz colorectal_xxx.npz
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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


def analyze_predictions_vs_ground_truth(checkpoint_path: str, npz_path: str, crop_size: tuple = (96, 96, 96)):
    """Analyze predictions vs ground truth using analyze_beam.py format."""
    
    ensure_safe_runtime_dirs()
    
    print(f"Loading: {npz_path}")
    case = load_case_npz(npz_path)
    
    print(f"\n{'='*70}")
    print(f"PREDICTION VS GROUND TRUTH ANALYSIS: {case.case_id}")
    print(f"{'='*70}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    
    # Load original data
    d_low = case.d_low
    d_high = case.d_high
    spr = case.spr
    beam_mask = case.beam_mask if case.beam_mask is not None else np.ones_like(d_low, dtype=np.float32)
    beam_axis = case.beam_axis  # 0=x, 1=y, 2=z (depth)
    
    print(f"\nBeam Configuration:")
    print(f"  Beam axis: {beam_axis} (0=width, 1=height, 2=depth)")
    print(f"  Volume shape: {d_low.shape} (D x H x W)")
    
    # Crop data for inference
    d_low_crop, spr_crop, d_high_crop = maybe_crop_bev(d_low, spr, d_high, crop_size=crop_size, enabled=True)
    beam_mask_crop = maybe_crop_bev(beam_mask, beam_mask, beam_mask, crop_size=crop_size, enabled=True)[0]
    
    # Prepare input for model
    e0_map = np.full_like(d_low_crop, fill_value=float(case.e0_mev), dtype=np.float32)
    x = np.stack([d_low_crop, spr_crop, e0_map, beam_mask_crop.astype(np.float32)], axis=0).astype(np.float32)
    x_tensor = torch.from_numpy(x[None, ...]).to(device)
    d_low_tensor = torch.from_numpy(d_low_crop[None, None, ...].astype(np.float32)).to(device)
    
    # Load model
    print(f"\n[Loading checkpoint] {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "args" in checkpoint and "base_channels" in checkpoint["args"]:
        base_channels = checkpoint["args"]["base_channels"]
    else:
        base_channels = 24
    
    model = ResidualUNet3D(in_channels=4, base_channels=base_channels, residual=True).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"[Checkpoint] Epoch {checkpoint.get('epoch', '?')}, best_val_l1={checkpoint.get('best_val_l1', '?'):.6f}")
    
    # Generate prediction
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
            delta = model(x_tensor)
            pred_tensor = d_low_tensor + delta
    
    # Convert to numpy
    pred = pred_tensor[0, 0].cpu().numpy()
    d_high_crop = d_high_crop  # Ground truth (cropped)
    
    print(f"\nVolume shape (cropped): {pred.shape}")
    
    # Compute projections
    print(f"\n{'─'*70}")
    print(f"DOSE PROJECTIONS (sum across axes)")
    print(f"{'─'*70}")
    
    # Along depth axis (z)
    proj_z_pred = pred.sum(axis=0)
    proj_z_gt = d_high_crop.sum(axis=0)
    proj_z_diff = proj_z_pred - proj_z_gt
    
    # Along height axis (y)
    proj_y_pred = pred.sum(axis=1)
    proj_y_gt = d_high_crop.sum(axis=1)
    
    # Along width axis (x)
    proj_x_pred = pred.sum(axis=2)
    proj_x_gt = d_high_crop.sum(axis=2)
    
    print(f"\nDepth (Z) projection:")
    print(f"  Prediction - max at ({np.unravel_index(proj_z_pred.argmax(), proj_z_pred.shape)})")
    print(f"  GT (D_high) - max at ({np.unravel_index(proj_z_gt.argmax(), proj_z_gt.shape)})")
    
    # Find Bragg peak
    print(f"\n{'─'*70}")
    print(f"BRAGG PEAK LOCATION")
    print(f"{'─'*70}")
    
    idx_pred = np.unravel_index(pred.argmax(), pred.shape)
    idx_gt = np.unravel_index(d_high_crop.argmax(), d_high_crop.shape)
    
    print(f"\nPrediction Bragg peak:")
    print(f"  Location: Depth={idx_pred[0]:3d}, Height={idx_pred[1]:3d}, Width={idx_pred[2]:3d}")
    print(f"  Value: {pred[idx_pred]:.6f}")
    print(f"  SPR at peak: {spr_crop[idx_pred]:.6f}")
    
    print(f"\nGround Truth Bragg peak:")
    print(f"  Location: Depth={idx_gt[0]:3d}, Height={idx_gt[1]:3d}, Width={idx_gt[2]:3d}")
    print(f"  Value: {d_high_crop[idx_gt]:.6f}")
    print(f"  SPR at peak: {spr_crop[idx_gt]:.6f}")
    
    # Depth profiles
    print(f"\n{'─'*70}")
    print(f"DEPTH PROFILES (along beam axis)")
    print(f"{'─'*70}")
    
    h_pred, w_pred = idx_pred[1], idx_pred[2]
    h_gt, w_gt = idx_gt[1], idx_gt[2]
    
    profile_pred = pred[:, h_pred, w_pred]
    profile_gt = d_high_crop[:, h_gt, w_gt]
    profile_spr = spr_crop[:, h_gt, w_gt]
    
    d_threshold = 0.01 * profile_gt.max()
    entrance_idx_pred = np.where(profile_pred > d_threshold)[0][0] if np.any(profile_pred > d_threshold) else 0
    entrance_idx_gt = np.where(profile_gt > d_threshold)[0][0] if np.any(profile_gt > d_threshold) else 0
    
    print(f"\nPrediction profile (at peak location: H={h_pred}, W={w_pred}):")
    print(f"  Entrance depth: ~{entrance_idx_pred}")
    print(f"  Peak depth: {idx_pred[0]}")
    print(f"  Peak value: {profile_pred.max():.6f}")
    
    print(f"\nGround Truth profile (at peak location: H={h_gt}, W={w_gt}):")
    print(f"  Entrance depth: ~{entrance_idx_gt}")
    print(f"  Peak depth: {idx_gt[0]}")
    print(f"  Peak value: {profile_gt.max():.6f}")
    
    # Error analysis
    print(f"\n{'─'*70}")
    print(f"ERROR ANALYSIS")
    print(f"{'─'*70}")
    
    diff = pred - d_high_crop
    abs_error = np.abs(diff)
    
    mask = d_high_crop > 0.01
    
    if np.sum(mask) > 0:
        mean_error = np.mean(abs_error[mask])
        max_error = np.max(abs_error[mask])
        print(f"\nAbsolute error (inside beam):")
        print(f"  Mean: {mean_error:.6f}")
        print(f"  Max:  {max_error:.6f}")
        
        correlation = np.corrcoef(spr_crop.ravel()[mask], diff.ravel()[mask])[0, 1]
        print(f"\nCorrelation between SPR and prediction error: {correlation:.4f}")
    
    # Visualize (same format as analyze_beam.py)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Profile visualization
    ax = axes[0, 0]
    ax.plot(profile_pred, label='Prediction', linewidth=2, color='blue')
    ax.plot(profile_gt, label='Ground Truth', linewidth=2, color='red')
    ax.axvline(idx_pred[0], color='blue', linestyle='--', alpha=0.5, label=f'Peak Pred ({idx_pred[0]})')
    ax.axvline(idx_gt[0], color='red', linestyle='--', alpha=0.5, label=f'Peak GT ({idx_gt[0]})')
    ax.set_xlabel('Depth Index')
    ax.set_ylabel('Dose')
    ax.set_title('Depth Profile at Peak Location')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Dose difference profile
    ax = axes[0, 1]
    diff_profile = profile_pred - profile_gt
    ax.plot(diff_profile, linewidth=2, color='purple')
    ax.fill_between(range(len(diff_profile)), diff_profile, alpha=0.3, color='purple')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Depth Index')
    ax.set_ylabel('Dose Difference (Pred - GT)')
    ax.set_title('Dose Difference Along Depth')
    ax.grid(True, alpha=0.3)
    
    # SPR profile
    ax = axes[0, 2]
    ax.plot(profile_spr, linewidth=2, color='brown')
    ax.set_xlabel('Depth Index')
    ax.set_ylabel('SPR')
    ax.set_title('SPR Along Depth')
    ax.grid(True, alpha=0.3)
    
    # XY projection of Ground Truth
    ax = axes[1, 0]
    proj = d_high_crop.max(axis=0)
    im = ax.imshow(proj, cmap='hot')
    ax.plot(idx_gt[2], idx_gt[1], 'b+', markersize=15, markeredgewidth=2)
    ax.set_title('Ground Truth Max Intensity Projection (XY)')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    plt.colorbar(im, ax=ax)
    
    # DZ projection of Ground Truth
    ax = axes[1, 1]
    proj = d_high_crop.max(axis=2)
    im = ax.imshow(proj, cmap='hot', aspect='auto')
    ax.plot(idx_gt[2], idx_gt[0], 'b+', markersize=15, markeredgewidth=2)
    ax.set_title('Ground Truth Max Intensity Projection (Depth-Height)')
    ax.set_xlabel('Width')
    ax.set_ylabel('Depth')
    plt.colorbar(im, ax=ax)
    
    # SPR vs Prediction error scatter
    ax = axes[1, 2]
    scatter_mask = mask & (np.abs(diff.ravel()) < np.std(diff[mask]) * 3)
    if np.sum(scatter_mask) > 0:
        ax.scatter(spr_crop.ravel()[scatter_mask], diff.ravel()[scatter_mask], alpha=0.1, s=1)
    ax.set_xlabel('SPR')
    ax.set_ylabel('Prediction Error (Pred - GT)')
    ax.set_title(f'SPR vs Error (r={correlation:.3f})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(npz_path).with_stem(Path(npz_path).stem + "_pred_analysis").with_suffix('.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Analysis saved: {output_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze predictions vs ground truth (like analyze_beam.py)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--npz", type=Path, required=True, help="Path to NPZ file")
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
    
    analyze_predictions_vs_ground_truth(str(args.checkpoint), str(args.npz), crop_size)


if __name__ == "__main__":
    main()
