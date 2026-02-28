#!/usr/bin/env python3
"""
EMERGENCY FIX: Retrospectively normalize already-saved NPZ files.
This will reload raw NPZ files and re-save them with proper normalization.
Use when normalize_doses_global() isn't being applied during initial generation.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io_npz import load_case_npz, save_case_npz


def retrospectively_normalize(npz_path: Path, dose_norm_const: float, dry_run: bool = False) -> bool:
    """
    Load an NPZ file, normalize doses, and save it back.
    
    Returns True if file was modified, False if already normalized or error.
    """
    try:
        # Load original
        case = load_case_npz(npz_path)
        d_low_orig = float(np.max(case.d_low))
        d_high_orig = float(np.max(case.d_high))
        
        # Check if already normalized (very small values)
        if d_high_orig < 200:
            # Already normalized or very small dataset
            return False
        
        # Normalize
        eps = 1e-8
        case.d_low = (case.d_low / (dose_norm_const + eps)).astype(np.float32)
        case.d_high = (case.d_high / (dose_norm_const + eps)).astype(np.float32)
        
        d_high_new = float(np.max(case.d_high))
        
        if not dry_run:
            save_case_npz(npz_path, case)
            print(f"✓ {npz_path.name:60s} | {d_high_orig:>8.1f} → {d_high_new:>8.1f} Gy")
            return True
        else:
            print(f"[DRY RUN] {npz_path.name:60s} | {d_high_orig:>8.1f} → {d_high_new:>8.1f} Gy")
            return True
            
    except Exception as e:
        print(f"❌ {npz_path.name:60s} | ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Retrospectively normalize NPZ files that weren't properly normalized during generation"
    )
    parser.add_argument("npz_dir", type=Path, help="Directory containing NPZ files")
    parser.add_argument("--dose-norm-const", type=float, default=100.0, help="Normalization constant")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without modifying files")
    parser.add_argument("--pattern", type=str, default="*.npz", help="Glob pattern for NPZ files")
    args = parser.parse_args()
    
    npz_dir = Path(args.npz_dir)
    if not npz_dir.is_dir():
        print(f"Error: {npz_dir} is not a directory")
        sys.exit(1)
    
    # Find all NPZ files
    npz_files = sorted(npz_dir.glob(args.pattern))
    if not npz_files:
        print(f"No NPZ files found matching {args.pattern} in {npz_dir}")
        sys.exit(1)
    
    print(f"Found {len(npz_files)} NPZ files")
    print(f"Normalization constant: {args.dose_norm_const}")
    if args.dry_run:
        print("DRY RUN MODE - Files will NOT be modified")
    print(f"\n{'File':60s} | {'Before':>8s} | {'After':>8s}")
    print("─" * 85)
    
    modified = 0
    skipped = 0
    errors = 0
    
    for npz_path in npz_files:
        if retrospectively_normalize(npz_path, args.dose_norm_const, args.dry_run):
            modified += 1
        else:
            skipped += 1
    
    print(f"\n{'─' * 85}")
    print(f"Results: {modified} modified, {skipped} skipped, {errors} errors out of {len(npz_files)} files")
    
    if not args.dry_run and modified > 0:
        print(f"\n✅ Retrospective normalization complete!")
        print(f"   Re-do any training runs with the normalized data")


if __name__ == "__main__":
    main()
