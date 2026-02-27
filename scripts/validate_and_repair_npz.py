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
            
            # Check for CORRECT keys
            required_new = {'d_low', 'd_high', 'spr'}
            # Also check for OLD key names (migration case)
            required_old = {'low_dose', 'high_dose', 'spr'}
            loaded_keys = set(keys)
            
            if required_new.issubset(loaded_keys):
                return True, "OK"
            elif required_old.issubset(loaded_keys):
                return False, "Old key names (needs migration)"
            else:
                missing = required_new - loaded_keys
                return False, f"Missing keys: {missing}"
        
        return True, "OK"
    
    except zipfile.BadZipFile as e:
        return False, f"BadZipFile: {str(e)}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"


def migrate_npz_keys(filepath: str) -> bool:
    """
    Convert old NPZ key names to new format.
    low_dose -> d_low
    high_dose -> d_high
    Overwrites the file with corrected keys.
    """
    try:
        filepath = Path(filepath)
        
        # Load old format
        with np.load(filepath, allow_pickle=True) as npz:
            old_data = {key: npz[key] for key in npz.files}
        
        # Check if needs migration
        if 'low_dose' not in old_data or 'high_dose' not in old_data:
            return False  # Already correct or invalid
        
        # Create new format
        new_data = {}
        for key, value in old_data.items():
            if key == 'low_dose':
                new_data['d_low'] = value
            elif key == 'high_dose':
                new_data['d_high'] = value
            else:
                new_data[key] = value
        
        # Save with new keys
        backup_path = filepath.with_suffix('.npz.bak')
        if filepath.exists():
            filepath.rename(backup_path)
        
        np.savez_compressed(filepath, **new_data)
        if backup_path.exists():
            backup_path.unlink()  # Remove backup after success
        
        return True
    
    except Exception as e:
        print(f"  Error migrating {filepath}: {e}")
        return False


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
        print(f"\n=== CORRUPTED/MIGRATION FILES ({len(corrupted_files)}) ===")
        
        migration_files = []
        true_corrupted = []
        
        for name, msg in corrupted_files:
            if "migration" in msg.lower():
                migration_files.append((name, msg))
            else:
                true_corrupted.append((name, msg))
        
        # Try to migrate old key format files
        if migration_files:
            print(f"\nMigrating {len(migration_files)} files with old key names...")
            migrated = 0
            for name, _ in migration_files:
                filepath = npz_dir / name
                if migrate_npz_keys(str(filepath)):
                    migrated += 1
                    if migrated % 50 == 0:
                        print(f"  Migrated: {migrated}/{len(migration_files)}")
            print(f"  Successfully migrated: {migrated}/{len(migration_files)}")
        
        # Remove truly corrupted files
        if true_corrupted:
            print(f"\nRemoving {len(true_corrupted)} truly corrupted files...")
            for name, msg in true_corrupted:
                filepath = npz_dir / name
                try:
                    os.remove(filepath)
                    print(f"  Removed: {name} ({msg})")
                except Exception as e:
                    print(f"  Failed to remove {name}: {e}")
    
    
    print(f"\nTo regenerate missing NPZ files, extract pair_index entries for:")
    print(f"  Files with indices: {[int(name.split('_')[-1].replace('.npz', '')) for name, _ in corrupted_files[:5]]}")
    print(f"\nThen run: bash scripts/run_prepare_and_train_mi210.sh")


if __name__ == "__main__":
    main()
