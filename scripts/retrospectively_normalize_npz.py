#!/usr/bin/env python3
"""
EMERGENCY FIX: Retrospectively normalize already-saved NPZ files.
This will reload raw NPZ files and re-save them with proper normalization.
Use when normalize_doses_global() isn't being applied during initial generation.
"""

import argparse
import sys
from pathlib import Path
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


def resolve_npz_files(npz_dir: Path | None, pattern: str, file_list: Path | None) -> list[Path]:
    if file_list is not None:
        lines = [line.strip() for line in file_list.read_text(encoding="utf-8").splitlines()]
        files = [Path(line) for line in lines if line and not line.startswith("#")]
        return sorted(files)

    if npz_dir is None:
        raise ValueError("Debes indicar npz_dir o --file-list")

    if not npz_dir.is_dir():
        raise ValueError(f"{npz_dir} no es un directorio")

    return sorted(npz_dir.glob(pattern))


def main():
    parser = argparse.ArgumentParser(
        description="Retrospectively normalize NPZ files that weren't properly normalized during generation"
    )
    parser.add_argument("npz_dir", type=Path, nargs="?", help="Directory containing NPZ files")
    parser.add_argument("--dose-norm-const", type=float, default=100.0, help="Normalization constant")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without modifying files")
    parser.add_argument("--pattern", type=str, default="*.npz", help="Glob pattern for NPZ files")
    parser.add_argument("--file-list", type=Path, default=None, help="Optional text file with one NPZ path per line")
    args = parser.parse_args()

    try:
        npz_files = resolve_npz_files(args.npz_dir, args.pattern, args.file_list)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if not npz_files:
        source_desc = f"file-list {args.file_list}" if args.file_list else f"{args.npz_dir} ({args.pattern})"
        print(f"No NPZ files found in {source_desc}")
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
