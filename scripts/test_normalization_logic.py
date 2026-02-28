#!/usr/bin/env python3
"""
Test script to verify normalization logic works correctly.
Creates a small synthetic dose pair and tests the normalization.
"""

import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_training_tensors import normalize_doses_global


def test_normalization():
    """Test that dose normalization works correctly."""
    
    print("="*70)
    print("NORMALIZATION LOGIC TEST")
    print("="*70)
    
    # Create synthetic doses
    low = np.array([100, 200, 300, 400, 500], dtype=np.float32)
    high = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.float32)
    
    print(f"\nInput doses (raw):")
    print(f"  low:  {low}")
    print(f"  high: {high}")
    
    # Test 1: With dose_scale = 1.0 (NO normalization)
    low_n1, high_n1 = normalize_doses_global(low, high, scale=1.0)
    print(f"\nWith scale=1.0 (NO normalization):")
    print(f"  low:  {low_n1}")
    print(f"  high: {high_n1}")
    print(f"  high max: {high_n1.max():.2f}")
    
    # Test 2: With dose_scale = 100.0 (EXPECTED normalization)
    low_n2, high_n2 = normalize_doses_global(low, high, scale=100.0)
    print(f"\nWith scale=100.0 (NORMALIZED):")
    print(f"  low:  {low_n2}")
    print(f"  high: {high_n2}")
    print(f"  high max: {high_n2.max():.2f}")
    
    # Verify
    print(f"\n{'─'*70}")
    print("VERIFICATION:")
    print(f"{'─'*70}")
    
    if high_n1.max() > 4000:
        print(f"✅ scale=1.0 works: high_max = {high_n1.max():.2f} (raw/unnormalized)")
    else:
        print(f"❌ scale=1.0 failed")
    
    if 45 < high_n2.max() < 55:
        print(f"✅ scale=100.0 works: high_max = {high_n2.max():.2f} (normalized)")
    else:
        print(f"❌ scale=100.0 failed: got {high_n2.max():.2f}, expected ~50")
    
    print(f"\n{'─'*70}")


if __name__ == "__main__":
    test_normalization()
