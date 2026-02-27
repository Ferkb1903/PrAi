#!/usr/bin/env python3
"""
Diagnose NaN in training data.
Checks for NaN/Inf in NPZ files loaded from manifest.
"""

import sys
from pathlib import Path

import numpy as np
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io_npz import load_case_npz


def diagnose_nan_in_data(manifest_path: Path):
    """Check first 100 samples for NaN/Inf."""
    print(f"Checking {manifest_path.name}...")
    
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    
    nan_count = 0
    inf_count = 0
    valid_count = 0
    
    for i, row in enumerate(rows[:100]):  # Check first 100
        npz_path = Path(row.get("npz_path", ""))
        
        try:
            case = load_case_npz(npz_path)
            
            # Check each array
            arrays = [
                ("d_low", case.d_low),
                ("d_high", case.d_high),
                ("spr", case.spr),
            ]
            if case.beam_mask is not None:
                arrays.append(("beam_mask", case.beam_mask))
            
            has_issue = False
            for name, arr in arrays:
                if arr is None:
                    continue
                
                nan_in_arr = np.isnan(arr).sum()
                inf_in_arr = np.isinf(arr).sum()
                
                if nan_in_arr > 0:
                    print(f"  [{i:3d}] {npz_path.name}: {name} has {nan_in_arr} NaN values")
                    nan_count += nan_in_arr
                    has_issue = True
                
                if inf_in_arr > 0:
                    print(f"  [{i:3d}] {npz_path.name}: {name} has {inf_in_arr} Inf values")
                    inf_count += inf_in_arr
                    has_issue = True
            
            if not has_issue:
                valid_count += 1
        
        except Exception as e:
            print(f"  [{i:3d}] {npz_path.name}: Error loading - {e}")
    
    print(f"\n=== DATA HEALTH CHECK ===")
    print(f"Checked: {min(100, len(rows))} samples")
    print(f"Valid: {valid_count}")
    print(f"Total NaN values: {nan_count}")
    print(f"Total Inf values: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        print(f"\n[!] WARNING: Found NaN/Inf in training data!")
        print(f"    This will cause loss=nan during training")
        print(f"    Need to regenerate NPZ with better QC")
    else:
        print(f"\n[✓] Data looks good - no NaN/Inf detected")
    
    return nan_count, inf_count


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_nan_in_data.py <manifest.csv>")
        sys.exit(1)
    
    manifest = Path(sys.argv[1])
    if not manifest.exists():
        print(f"Error: {manifest} not found")
        sys.exit(1)
    
    nan_count, inf_count = diagnose_nan_in_data(manifest)
    
    if nan_count > 0 or inf_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)
