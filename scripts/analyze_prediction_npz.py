#!/usr/bin/env python3
"""
Analyze prediction NPZ locally (same format as analyze_beam.py).
loads d_low, d_high (GT), d_pred, spr, etc. and compares GT vs Prediction.
Usage:
  python scripts/analyze_prediction_npz.py pred_output.npz
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def analyze_prediction_npz(npz_path: str):
    """Analyze prediction NPZ file (GT vs Pred)."""
    
    npz_file = Path(npz_path)
    if not npz_file.exists():
        print(f"ERROR: {npz_path} not found")
        sys.exit(1)
    
    print(f"Loading: {npz_path}")
    data = np.load(npz_path)
    
    d_low = data["d_low"]
    d_high = data["d_high"]  # Ground truth
    d_pred = data["d_pred"]  # Model prediction
    spr = data["spr"]
    beam_mask = data["beam_mask"]
    e0_mev = float(data["e0_mev"])
    case_id = str(data["case_id"]) if "case_id" in data else "unknown"
    
    # Ensure spr is at least 3D
    if spr.ndim == 0:
        spr = np.full_like(d_low, float(spr), dtype=np.float32)
    elif spr.ndim == 1 or spr.size == 1:
        spr = np.full_like(d_low, np.mean(spr), dtype=np.float32)
    
    print(f"\n{'='*70}")
    print(f"PREDICTION NPZ ANALYSIS: {case_id}")
    print(f"{'='*70}")
    
    print(f"\nVolume Configuration:")
    print(f"  Shape: {d_low.shape} (D x H x W)")
    print(f"  E0: {e0_mev:.0f} MeV")
    
    # Compute projections
    print(f"\n{'─'*70}")
    print(f"DOSE PROJECTIONS (sum across axes)")
    print(f"{'─'*70}")
    
    proj_z_pred = d_pred.sum(axis=0)
    proj_z_gt = d_high.sum(axis=0)
    
    print(f"\nDepth (Z) projection:")
    print(f"  Prediction - max at ({np.unravel_index(proj_z_pred.argmax(), proj_z_pred.shape)})")
    print(f"  GT (D_high) - max at ({np.unravel_index(proj_z_gt.argmax(), proj_z_gt.shape)})")
    
    # Find Bragg peak
    print(f"\n{'─'*70}")
    print(f"BRAGG PEAK LOCATION")
    print(f"{'─'*70}")
    
    idx_pred = np.unravel_index(d_pred.argmax(), d_pred.shape)
    idx_gt = np.unravel_index(d_high.argmax(), d_high.shape)
    
    print(f"\nPrediction Bragg peak:")
    print(f"  Location: Depth={idx_pred[0]:3d}, Height={idx_pred[1]:3d}, Width={idx_pred[2]:3d}")
    print(f"  Value: {d_pred[idx_pred]:.6f}")
    print(f"  SPR at peak: {spr[idx_pred]:.6f}")
    
    print(f"\nGround Truth Bragg peak:")
    print(f"  Location: Depth={idx_gt[0]:3d}, Height={idx_gt[1]:3d}, Width={idx_gt[2]:3d}")
    print(f"  Value: {d_high[idx_gt]:.6f}")
    print(f"  SPR at peak: {spr[idx_gt]:.6f}")
    
    # Depth profiles
    print(f"\n{'─'*70}")
    print(f"DEPTH PROFILES (along beam axis)")
    print(f"{'─'*70}")
    
    h_pred, w_pred = idx_pred[1], idx_pred[2]
    h_gt, w_gt = idx_gt[1], idx_gt[2]
    
    profile_pred = d_pred[:, h_pred, w_pred]
    profile_gt = d_high[:, h_gt, w_gt]
    profile_spr = spr[:, h_gt, w_gt]
    
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
    
    diff = d_pred - d_high
    abs_error = np.abs(diff)
    
    mask = d_high > 0.01
    correlation = np.nan
    
    if np.sum(mask) > 0:
        mean_error = np.mean(abs_error[mask])
        max_error = np.max(abs_error[mask])
        print(f"\nAbsolute error (inside beam):")
        print(f"  Mean: {mean_error:.6f}")
        print(f"  Max:  {max_error:.6f}")
        
        spr_masked = spr.ravel()[mask.ravel()]
        diff_masked = diff.ravel()[mask.ravel()]
        
        if len(spr_masked) > 1 and np.std(spr_masked) > 0 and np.std(diff_masked) > 0:
            corr_matrix = np.corrcoef(spr_masked, diff_masked)
            correlation = corr_matrix[0, 1]
            print(f"\nCorrelation between SPR and prediction error: {correlation:.4f}")
        else:
            print(f"\nCorrelation between SPR and prediction error: N/A (insufficient variance)")
    
    # Visualize (same format as analyze_beam.py)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Profile visualization
    ax = axes[0, 0]
    ax.plot(profile_pred, label='Prediction', linewidth=2.5, color='blue')
    ax.plot(profile_gt, label='Ground Truth (D_high)', linewidth=2.5, color='red')
    profile_low = d_low[:, h_gt, w_gt]
    ax.plot(profile_low, label='D_low (input)', linewidth=2, color='gray', alpha=0.7)
    ax.axvline(idx_pred[0], color='blue', linestyle='--', alpha=0.5, label=f'Peak Pred ({idx_pred[0]})')
    ax.axvline(idx_gt[0], color='red', linestyle='--', alpha=0.5, label=f'Peak GT ({idx_gt[0]})')
    ax.set_xlabel('Depth Index')
    ax.set_ylabel('Dose')
    ax.set_title('Depth Profile at Peak Location')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Dose difference profile (show residuals)
    ax = axes[0, 1]
    pred_residual = profile_pred - profile_low  # What model added
    gt_residual = profile_gt - profile_low      # What should be added
    ax.plot(gt_residual, linewidth=2.5, color='red', label='GT Residual (D_high - D_low)', alpha=0.8)
    ax.plot(pred_residual, linewidth=2.5, color='blue', label='Pred Residual (Prediction - D_low)', alpha=0.8)
    ax.fill_between(range(len(gt_residual)), gt_residual, alpha=0.2, color='red')
    ax.fill_between(range(len(pred_residual)), pred_residual, alpha=0.2, color='blue')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Depth Index')
    ax.set_ylabel('Added Dose (Residual)')
    ax.set_title('Dose Residual Comparison')
    ax.legend(fontsize=9)
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
    proj = d_high.max(axis=0)
    im = ax.imshow(proj, cmap='hot')
    ax.plot(idx_gt[2], idx_gt[1], 'b+', markersize=15, markeredgewidth=2)
    ax.set_title('Ground Truth Max Intensity Projection (XY)')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    plt.colorbar(im, ax=ax)
    
    # DZ projection of Ground Truth
    ax = axes[1, 1]
    proj = d_high.max(axis=2)
    im = ax.imshow(proj, cmap='hot', aspect='auto')
    ax.plot(idx_gt[2], idx_gt[0], 'b+', markersize=15, markeredgewidth=2)
    ax.set_title('Ground Truth Max Intensity Projection (Depth-Height)')
    ax.set_xlabel('Width')
    ax.set_ylabel('Depth')
    plt.colorbar(im, ax=ax)
    
    # SPR vs Prediction error scatter
    ax = axes[1, 2]
    scatter_mask = mask & (np.abs(diff.ravel().reshape(mask.shape)) < np.std(diff[mask]) * 3) if np.sum(mask) > 0 else np.zeros_like(mask)
    if np.sum(scatter_mask) > 0:
        ax.scatter(spr.ravel()[scatter_mask.ravel()], diff.ravel()[scatter_mask.ravel()], alpha=0.1, s=1)
        title_str = f'SPR vs Error (r={correlation:.3f})' if not np.isnan(correlation) else 'SPR vs Error'
    else:
        title_str = 'SPR vs Error (no data)'
    ax.set_xlabel('SPR')
    ax.set_ylabel('Prediction Error (Pred - GT)')
    ax.set_title(title_str)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = npz_file.with_stem(npz_file.stem + "_analysis").with_suffix('.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Analysis saved: {output_path}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_prediction_npz.py <npz_file_path>")
        sys.exit(1)
    
    npz_file = sys.argv[1]
    analyze_prediction_npz(npz_file)
