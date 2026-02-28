#!/usr/bin/env python3
"""
Visualize NPZ training data locally.
Displays dose distributions, SPR, energy, and beam mask.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io_npz import load_case_npz


def visualize_case(npz_path: str):
    """Load and visualize a single NPZ case."""
    print(f"Loading: {npz_path}")
    case = load_case_npz(npz_path)
    
    print(f"\n{'='*60}")
    print(f"CASE DATA: {case.case_id}")
    print(f"{'='*60}")
    
    # Meta information
    print(f"\nMetadata:")
    print(f"  Energy: {case.e0_mev:.2f} MeV")
    print(f"  Spacing: {case.spacing_mm} mm")
    print(f"  Beam axis: {case.beam_axis}")
    
    # Array shapes and ranges
    print(f"\nArray Shapes:")
    print(f"  d_low:  {case.d_low.shape} | range: [{case.d_low.min():.6f}, {case.d_low.max():.6f}]")
    print(f"  d_high: {case.d_high.shape} | range: [{case.d_high.min():.6f}, {case.d_high.max():.6f}]")
    print(f"  spr:    {case.spr.shape} | range: [{case.spr.min():.6f}, {case.spr.max():.6f}]")
    if case.beam_mask is not None:
        print(f"  beam_mask: {case.beam_mask.shape} | voxels active: {(case.beam_mask > 0).sum()}")
    
    # Compute residual
    residual = case.d_high - case.d_low
    print(f"\nResidual (Delta = D_high - D_low):")
    print(f"  delta:  {residual.shape} | range: [{residual.min():.6f}, {residual.max():.6f}]")
    print(f"  mean abs delta: {np.abs(residual).mean():.6f}")
    
    # Statistics
    print(f"\nBasic Statistics:")
    print(f"  D_low  - mean: {case.d_low.mean():.6f}, std: {case.d_low.std():.6f}")
    print(f"  D_high - mean: {case.d_high.mean():.6f}, std: {case.d_high.std():.6f}")
    print(f"  SPR    - mean: {case.spr.mean():.6f}, std: {case.spr.std():.6f}")
    
    # 3D Visualization
    fig = plt.figure(figsize=(16, 12))
    
    # Get center slices for visualization
    d, h, w = case.d_low.shape
    mid_d, mid_h, mid_w = d//2, h//2, w//2
    
    # 1. D_low axial slice
    ax1 = fig.add_subplot(3, 3, 1)
    im1 = ax1.imshow(case.d_low[mid_d, :, :], cmap='viridis')
    ax1.set_title(f'D_low (Depth {mid_d})')
    ax1.set_ylabel('Height')
    ax1.set_xlabel('Width')
    plt.colorbar(im1, ax=ax1)
    
    # 2. D_high axial slice
    ax2 = fig.add_subplot(3, 3, 2)
    im2 = ax2.imshow(case.d_high[mid_d, :, :], cmap='viridis')
    ax2.set_title(f'D_high (Depth {mid_d})')
    ax2.set_ylabel('Height')
    ax2.set_xlabel('Width')
    plt.colorbar(im2, ax=ax2)
    
    # 3. Residual/Delta
    ax3 = fig.add_subplot(3, 3, 3)
    im3 = ax3.imshow(residual[mid_d, :, :], cmap='RdBu_r')
    ax3.set_title(f'Delta = D_high - D_low (Depth {mid_d})')
    ax3.set_ylabel('Height')
    ax3.set_xlabel('Width')
    plt.colorbar(im3, ax=ax3)
    
    # 4. SPR axial slice
    ax4 = fig.add_subplot(3, 3, 4)
    im4 = ax4.imshow(case.spr[mid_d, :, :], cmap='plasma')
    ax4.set_title(f'SPR (Depth {mid_d})')
    ax4.set_ylabel('Height')
    ax4.set_xlabel('Width')
    plt.colorbar(im4, ax=ax4)
    
    # 5. Beam mask
    if case.beam_mask is not None:
        ax5 = fig.add_subplot(3, 3, 5)
        im5 = ax5.imshow(case.beam_mask[mid_d, :, :], cmap='gray')
        ax5.set_title(f'Beam Mask (Depth {mid_d})')
        ax5.set_ylabel('Height')
        ax5.set_xlabel('Width')
        plt.colorbar(im5, ax=ax5)
    
    # 6. Sagittal slice (D_low)
    ax6 = fig.add_subplot(3, 3, 6)
    im6 = ax6.imshow(case.d_low[:, mid_h, :], cmap='viridis', aspect='auto')
    ax6.set_title(f'D_low Sagittal (Height {mid_h})')
    ax6.set_ylabel('Depth')
    ax6.set_xlabel('Width')
    plt.colorbar(im6, ax=ax6)
    
    # 7. Coronal slice (D_low)
    ax7 = fig.add_subplot(3, 3, 7)
    im7 = ax7.imshow(case.d_low[:, :, mid_w], cmap='viridis', aspect='auto')
    ax7.set_title(f'D_low Coronal (Width {mid_w})')
    ax7.set_ylabel('Depth')
    ax7.set_xlabel('Height')
    plt.colorbar(im7, ax=ax7)
    
    # 8. Histogram
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.hist(case.d_low.ravel(), bins=50, alpha=0.5, label='D_low', density=True)
    ax8.hist(case.d_high.ravel(), bins=50, alpha=0.5, label='D_high', density=True)
    ax8.set_xlabel('Dose Value')
    ax8.set_ylabel('Frequency (normalized)')
    ax8.set_title('Dose Distribution')
    ax8.legend()
    ax8.set_yscale('log')
    
    # 9. Profile (along depth axis at center)
    ax9 = fig.add_subplot(3, 3, 9)
    profile_low = case.d_low[:, mid_h, mid_w]
    profile_high = case.d_high[:, mid_h, mid_w]
    ax9.plot(profile_low, label='D_low', linewidth=2)
    ax9.plot(profile_high, label='D_high', linewidth=2)
    ax9.set_xlabel('Depth Index')
    ax9.set_ylabel('Dose')
    ax9.set_title('Depth Profile (center)')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(npz_path).with_suffix('.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_npz.py <npz_file_path>")
        sys.exit(1)
    
    npz_file = sys.argv[1]
    if not Path(npz_file).exists():
        print(f"Error: {npz_file} not found")
        sys.exit(1)
    
    visualize_case(npz_file)
