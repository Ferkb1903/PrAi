#!/usr/bin/env python3
"""
Diagnostic script to verify if NPZ files are actually normalized.
Compares dose ranges against expected values.
"""

import sys
from pathlib import Path
import numpy as np
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io_npz import load_case_npz


def diagnose_normalization(npz_path: Path, expected_norm_const: float = 100.0) -> None:
    """Check if NPZ is normalized correctly."""
    
    print(f"\n{'='*70}")
    print(f"NORMALIZATION DIAGNOSTIC: {npz_path.name}")
    print(f"{'='*70}")
    
    if not npz_path.exists():
        print(f"❌ File not found: {npz_path}")
        return
    
    case = load_case_npz(npz_path)
    d_low = case.d_low
    d_high = case.d_high
    
    # Get ranges
    d_low_max = float(np.max(d_low))
    d_high_max = float(np.max(d_high))
    
    print(f"\nActual dose ranges:")
    print(f"  d_low  max: {d_low_max:>10.2f} Gy")
    print(f"  d_high max: {d_high_max:>10.2f} Gy")
    
    print(f"\nExpected ranges (if normalized by {expected_norm_const}):")
    print("  Typical d_high max often falls in ~50-300 Gy after /100")
    print(f"  (equivalent raw max estimate ≈ {expected_norm_const*d_high_max:.1f} Gy)")
    
    # Diagnostic checks
    print(f"\n{'─'*70}")
    print("CHECKS:")
    print(f"{'─'*70}")
    
    # Check 1: Is d_high normalized?
    # Heuristic bands for dose-norm-const=100:
    # - <= 600: usually already normalized
    # - 600..2000: ambiguous, inspect sample manually
    # - > 2000: likely still raw (pre-normalization)
    if d_high_max > 2000:
        print(f"❌ d_high_max = {d_high_max:.1f} >> LIKELY RAW (not normalized)")
        print(f"   If divided by {expected_norm_const}, it would be ≈ {d_high_max/expected_norm_const:.1f} Gy")
    elif d_high_max > 600:
        print(f"⚠️  d_high_max = {d_high_max:.1f} in ambiguous zone (600-2000)")
        print("   Review a few more files or verify generation logs")
    elif d_high_max < 0.5:
        print(f"⚠️  d_high_max = {d_high_max:.6f} << Very small (possible issue)")
    else:
        print(f"✅ d_high_max = {d_high_max:.2f} (appears normalized)")
    
    # Check 2: Dose ratio
    ratio = d_high_max / max(d_low_max, 1e-8)
    print(f"\nDose correction ratio (d_high/d_low): {ratio:.1f}x")
    if ratio < 2:
        print(f"   ⚠️  Low correction ratio - may indicate data issue")
    else:
        print(f"   ✅ Reasonable correction ratio")
    
    # Check 3: Histogram
    print(f"\nd_high distribution:")
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(d_high, p)
        print(f"  p{p:2d}: {val:>10.4f}")
    
    # Check 4: Metadata
    print(f"\nMetadata:")
    print(f"  Energy: {case.e0_mev:.4f} (normalized)")
    print(f"  Case ID: {case.case_id}")
    print(f"  d_low shape: {case.d_low.shape}")
    print(f"  d_high shape: {case.d_high.shape}")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose NPZ normalization")
    parser.add_argument("npz_files", nargs="+", type=Path, help="NPZ files to check")
    parser.add_argument("--norm-const", type=float, default=100.0, help="Expected normalization constant")
    args = parser.parse_args()
    
    for npz_path in args.npz_files:
        diagnose_normalization(npz_path, args.norm_const)
