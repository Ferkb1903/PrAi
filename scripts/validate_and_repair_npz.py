#!/usr/bin/env python3
"""
Validate all NPZ files and remove corrupted ones.
Helps identify which files need to be regenerated.
"""

import os
import sys
import zipfile
from pathlib import Path
import numpy as np

def validate_npz_file(filepath: str) -> tuple[bool, str]:
    """
    Check if NPZ file is valid.
    Returns (is_valid, error_message)
    """
    try:
        # Check file exists and has size
        if not os.path.exists(filepath):
            return False, "File does not exist"
        
        file_size = os.path.getsize(filepath)
        if file_size < 100:  # NPZ should be > 100 bytes
            return False, f"File too small ({file_size} bytes)"
        
        # Try to open as zip (NPZ is a zip file)
        with zipfile.ZipFile(filepath, 'r') as zf:
            zf.testzip()  # This returns None if all OK
        
        # Try to load with numpy
        with np.load(filepath, allow_pickle=True) as npz:
            keys = list(npz.keys())
            if not keys:
                return False, "No arrays in NPZ"
            # Check that required keys exist
            required = {'d_low', 'd_high', 'spr'}
            loaded_keys = set(keys)
            if not required.issubset(loaded_keys):
                return False, f"Missing keys: {required - loaded_keys}"
        
        return True, "OK"
    
    except zipfile.BadZipFile as e:
        return False, f"BadZipFile: {str(e)}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_and_repair_npz.py <npz_directory>")
        sys.exit(1)
    
    npz_dir = Path(sys.argv[1])
    if not npz_dir.exists():
        print(f"Error: Directory {npz_dir} does not exist")
        sys.exit(1)
    
    # Find all NPZ files
    npz_files = sorted(npz_dir.glob("*.npz"))
    print(f"Found {len(npz_files)} NPZ files in {npz_dir}")
    
    valid_count = 0
    corrupted_files = []
    
    for i, npz_file in enumerate(npz_files, 1):
        is_valid, msg = validate_npz_file(str(npz_file))
        
        if is_valid:
            valid_count += 1
            if i % 100 == 0:
                print(f"  [{i:5d}/{len(npz_files)}] ✓ {npz_file.name}")
        else:
            corrupted_files.append((npz_file.name, msg))
            print(f"  [{i:5d}/{len(npz_files)}] ✗ {npz_file.name} - {msg}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total:     {len(npz_files)}")
    print(f"Valid:     {valid_count}")
    print(f"Corrupted: {len(corrupted_files)}")
    
    if corrupted_files:
        print(f"\n=== CORRUPTED FILES ({len(corrupted_files)}) ===")
        for name, msg in corrupted_files:
            print(f"  {name}: {msg}")
        
        # Remove corrupted files
        print(f"\nRemoving corrupted files...")
        for name, _ in corrupted_files:
            filepath = npz_dir / name
            try:
                os.remove(filepath)
                print(f"  Removed: {name}")
            except Exception as e:
                print(f"  Failed to remove {name}: {e}")
    
    print(f"\nTo regenerate missing NPZ files, extract pair_index entries for:")
    print(f"  Files with indices: {[int(name.split('_')[-1].replace('.npz', '')) for name, _ in corrupted_files[:5]]}")
    print(f"\nThen run: bash scripts/run_prepare_and_train_mi210.sh")


if __name__ == "__main__":
    main()
