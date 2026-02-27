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
    print(f"Found {len(npz_files)} NPZ files in {npz_dir}\n")
    
    valid_count = 0
    corrupted_files = []
    
    for i, npz_file in enumerate(npz_files, 1):
        is_valid, msg = validate_npz_file(str(npz_file))
        
        if is_valid:
            valid_count += 1
        else:
            corrupted_files.append((npz_file.name, msg, npz_file))
        
        if i % 100 == 0:
            status = f"✓ {valid_count} valid" if is_valid else f"✗ {len(corrupted_files)} corrupted"
            print(f"  [{i:5d}/{len(npz_files)}] {status}", flush=True)
    
    print(f"\n{'='*60}")
    print(f"VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total:     {len(npz_files)}")
    print(f"Valid:     {valid_count}")
    print(f"Corrupted: {len(corrupted_files)}")
    
    if corrupted_files:
        print(f"\n{'='*60}")
        print(f"PROCESSING CORRUPTED FILES ({len(corrupted_files)})")
        print(f"{'='*60}")
        
        migration_files = []
        true_corrupted = []
        
        for name, msg, path in corrupted_files:
            if "migration" in msg.lower():
                migration_files.append((name, msg, path))
            else:
                true_corrupted.append((name, msg, path))
        
        # Try to migrate old key format files
        if migration_files:
            print(f"\nMigrating {len(migration_files)} files with old key names...")
            migrated = 0
            for name, _, filepath in migration_files:
                if migrate_npz_keys(str(filepath)):
                    migrated += 1
            print(f"  ✓ Successfully migrated: {migrated}/{len(migration_files)}")
        
        # Remove truly corrupted files
        removed_files = []
        if true_corrupted:
            print(f"\nRemoving {len(true_corrupted)} truly corrupted files...")
            for name, msg, filepath in true_corrupted:
                try:
                    os.remove(filepath)
                    removed_files.append(name)
                    print(f"  ✗ Removed: {name}")
                except Exception as e:
                    print(f"  ✗ Failed to remove {name}: {e}")
        
        if removed_files:
            print(f"\n{'='*60}")
            print(f"REGENERATION NEEDED ({len(removed_files)} files)")
            print(f"{'='*60}")
            print(f"\nRemoved files that need to be regenerated:")
            for fname in removed_files[:10]:
                print(f"  - {fname}")
            if len(removed_files) > 10:
                print(f"  ... and {len(removed_files) - 10} more")
            print(f"\nTo regenerate all missing NPZ files, run:")
            print(f"  bash scripts/run_prepare_and_train_mi210.sh")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
