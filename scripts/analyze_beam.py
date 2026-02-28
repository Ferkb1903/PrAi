#!/usr/bin/env python3
"""
Analyze beam direction and Bragg peak location.
Identifies where the proton beam enters, its peak, and how it attenuates.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io_npz import load_case_npz


def analyze_beam_direction(npz_path: str):
    """Analyze beam direction and Bragg peak location."""
    print(f"Loading: {npz_path}")
    case = load_case_npz(npz_path)
    
    print(f"\n{'='*70}")
    print(f"BEAM ANALYSIS: {case.case_id}")
    print(f"{'='*70}")
    
    d_low = case.d_low
    d_high = case.d_high
    spr = case.spr
    beam_axis = case.beam_axis  # 0=x, 1=y, 2=z (depth)
    
    print(f"\nBeam Configuration:")
    print(f"  Beam axis: {beam_axis} (0=width, 1=height, 2=depth)")
    print(f"  Volume shape: {d_low.shape} (D x H x W)")
    
    # Compute dose projections along different axes
    print(f"\n{'─'*70}")
    print(f"DOSE PROJECTIONS (sum across axes)")
    print(f"{'─'*70}")
    
    # Along depth axis (z)
    proj_z_low = d_low.sum(axis=0)  # Shape: (H, W)
    proj_z_high = d_high.sum(axis=0)
    proj_z_delta = proj_z_low - proj_z_high  # This should be negative (high > low)
    
    # Along height axis (y)
    proj_y_low = d_low.sum(axis=1)  # Shape: (D, W)
    proj_y_high = d_high.sum(axis=1)
    
    # Along width axis (x)
    proj_x_low = d_low.sum(axis=2)  # Shape: (D, H)
    proj_x_high = d_high.sum(axis=2)
    
    print(f"\nDepth (Z) projection:")
    print(f"  D_low  - max at ({np.unravel_index(proj_z_low.argmax(), proj_z_low.shape)})")
    print(f"  D_high - max at ({np.unravel_index(proj_z_high.argmax(), proj_z_high.shape)})")
    print(f"  Residual (D_high - D_low) - max at ({np.unravel_index((-proj_z_delta).argmax(), proj_z_delta.shape)})")
    
    print(f"\nWidth (X) projection:")
    print(f"  D_low  - max at depth {np.unravel_index(proj_x_low.argmax(), proj_x_low.shape)[0]}")
    print(f"  D_high - max at depth {np.unravel_index(proj_x_high.argmax(), proj_x_high.shape)[0]}")
    
    print(f"\nHeight (Y) projection:")
    print(f"  D_low  - max at depth {np.unravel_index(proj_y_low.argmax(), proj_y_low.shape)[0]}")
    print(f"  D_high - max at depth {np.unravel_index(proj_y_high.argmax(), proj_y_high.shape)[0]}")
    
    # Find Bragg peak (maximum dose point)
    print(f"\n{'─'*70}")
    print(f"BRAGG PEAK LOCATION")
    print(f"{'─'*70}")
    
    # Find peak of D_low and D_high
    idx_low = np.unravel_index(d_low.argmax(), d_low.shape)
    idx_high = np.unravel_index(d_high.argmax(), d_high.shape)
    
    print(f"\nD_low Bragg peak:")
    print(f"  Location: Depth={idx_low[0]:3d}, Height={idx_low[1]:3d}, Width={idx_low[2]:3d}")
    print(f"  Value: {d_low[idx_low]:.6f}")
    print(f"  SPR at peak: {spr[idx_low]:.6f}")
    
    print(f"\nD_high Bragg peak:")
    print(f"  Location: Depth={idx_high[0]:3d}, Height={idx_high[1]:3d}, Width={idx_high[2]:3d}")
    print(f"  Value: {d_high[idx_high]:.6f}")
    print(f"  SPR at peak: {spr[idx_high]:.6f}")
    
    # Depth profiles
    print(f"\n{'─'*70}")
    print(f"DEPTH PROFILES (along beam axis)")
    print(f"{'─'*70}")
    
    # Get profiles at the location of max dose
    h_low, w_low = idx_low[1], idx_low[2]
    h_high, w_high = idx_high[1], idx_high[2]
    
    profile_low = d_low[:, h_low, w_low]
    profile_high = d_high[:, h_high, w_high]
    profile_spr = spr[:, h_high, w_high]
    
    # Find where dose starts increasing significantly
    d_threshold = 0.01 * profile_high.max()
    entrance_idx_low = np.where(profile_low > d_threshold)[0][0] if np.any(profile_low > d_threshold) else 0
    entrance_idx_high = np.where(profile_high > d_threshold)[0][0] if np.any(profile_high > d_threshold) else 0
    
    print(f"\nD_low profile (at peak location: H={h_low}, W={w_low}):")
    print(f"  Entrance depth: ~{entrance_idx_low}")
    print(f"  Peak depth: {idx_low[0]}")
    print(f"  Peak value: {profile_low.max():.6f}")
    print(f"  Range (100% to 50%): {np.where(profile_low > 0.5*profile_low.max())[0][-1] - np.where(profile_low > 0.5*profile_low.max())[0][0]}")
    
    print(f"\nD_high profile (at peak location: H={h_high}, W={w_high}):")
    print(f"  Entrance depth: ~{entrance_idx_high}")
    print(f"  Peak depth: {idx_high[0]}")
    print(f"  Peak value: {profile_high.max():.6f}")
    print(f"  Range (100% to 50%): {np.where(profile_high > 0.5*profile_high.max())[0][-1] - np.where(profile_high > 0.5*profile_high.max())[0][0]}")
    print(f"  SPR at peak: {profile_spr[idx_high[0]]:.6f}")
    
    # SPR effect analysis
    print(f"\n{'─'*70}")
    print(f"SPR EFFECT ON DOSE DIFFERENCE")
    print(f"{'─'*70}")
    
    residual = d_high - d_low
    
    # Correlation between SPR change and dose difference
    spr_flat = spr.ravel()
    residual_flat = residual.ravel()
    
    # Only consider voxels with meaningful data
    mask = (d_low.ravel() > 0.01) & (d_high.ravel() > 0.01)
    
    if np.sum(mask) > 0:
        correlation = np.corrcoef(spr_flat[mask], residual_flat[mask])[0, 1]
        print(f"\nCorrelation between SPR and residual dose: {correlation:.4f}")
        print(f"  (Higher SPR → Higher residual dose, expected: positive)")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Profile visualization
    ax = axes[0, 0]
    ax.plot(profile_low, label='D_low', linewidth=2, color='blue')
    ax.plot(profile_high, label='D_high', linewidth=2, color='red')
    ax.axvline(idx_low[0], color='blue', linestyle='--', alpha=0.5, label=f'Peak D_low ({idx_low[0]})')
    ax.axvline(idx_high[0], color='red', linestyle='--', alpha=0.5, label=f'Peak D_high ({idx_high[0]})')
    ax.set_xlabel('Depth Index')
    ax.set_ylabel('Dose')
    ax.set_title('Depth Profile at Peak Location')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Residual profile
    ax = axes[0, 1]
    residual_profile = profile_high - profile_low
    ax.plot(residual_profile, linewidth=2, color='green')
    ax.fill_between(range(len(residual_profile)), residual_profile, alpha=0.3, color='green')
    ax.set_xlabel('Depth Index')
    ax.set_ylabel('Delta Dose (D_high - D_low)')
    ax.set_title('Dose Difference Along Depth')
    ax.grid(True, alpha=0.3)
    
    # SPR profile
    ax = axes[0, 2]
    ax.plot(profile_spr, linewidth=2, color='purple')
    ax.set_xlabel('Depth Index')
    ax.set_ylabel('SPR')
    ax.set_title('SPR Along Depth')
    ax.grid(True, alpha=0.3)
    
    # XY projection of D_high
    ax = axes[1, 0]
    proj = d_high.max(axis=0)  # Max across depth
    im = ax.imshow(proj, cmap='hot')
    ax.plot(idx_high[2], idx_high[1], 'b+', markersize=15, markeredgewidth=2)
    ax.set_title('D_high Max Intensity Projection (XY)')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    plt.colorbar(im, ax=ax)
    
    # DZ projection of D_high
    ax = axes[1, 1]
    proj = d_high.max(axis=2)  # Max across width
    im = ax.imshow(proj, cmap='hot', aspect='auto')
    ax.plot(idx_high[2], idx_high[0], 'b+', markersize=15, markeredgewidth=2)
    ax.set_title('D_high Max Intensity Projection (Depth-Height)')
    ax.set_xlabel('Width')
    ax.set_ylabel('Depth')
    plt.colorbar(im, ax=ax)
    
    # SPR vs Residual scatter
    ax = axes[1, 2]
    scatter_mask = mask & (np.abs(residual_flat) < residual_flat[mask].std() * 3)  # Remove outliers
    ax.scatter(spr_flat[scatter_mask], residual_flat[scatter_mask], alpha=0.1, s=1)
    ax.set_xlabel('SPR')
    ax.set_ylabel('Residual Dose (D_high - D_low)')
    ax.set_title(f'SPR vs Residual (r={correlation:.3f})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(npz_path).with_stem(Path(npz_path).stem + "_beam_analysis").with_suffix('.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Analysis saved: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_beam.py <npz_file_path>")
        sys.exit(1)
    
    npz_file = sys.argv[1]
    if not Path(npz_file).exists():
        print(f"Error: {npz_file} not found")
        sys.exit(1)
    
    analyze_beam_direction(npz_file)
