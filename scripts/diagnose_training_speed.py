#!/usr/bin/env python3
"""
Diagnose training speed and GPU usage.
Checks:
1. GPU availability and memory
2. Data loading speed
3. Forward/backward pass timing
4. Identifies bottlenecks
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io_npz import load_case_npz
from src.data.preprocess import maybe_crop_bev
from src.model.resunet3d import ResidualUNet3D


def diagnose():
    print("=" * 70)
    print("TRAINING SPEED DIAGNOSIS")
    print("=" * 70)
    
    # 1. GPU Check
    print("\n1. GPU AVAILABILITY")
    print("-" * 70)
    print(f"  PyTorch version:       {torch.__version__}")
    print(f"  CUDA available:        {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  CUDA device count:     {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  Device {i}:              {props.name}")
            print(f"    - Compute Capability: {props.major}.{props.minor}")
            print(f"    - Total Memory:       {props.total_memory / 1e9:.1f} GB")
        
        print(f"\n  Current Device:        {torch.cuda.current_device()}")
        print(f"  Memory Allocated:      {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Memory Reserved:       {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print(f"  WARNING: CUDA not available! Training on CPU will be ~100x slower")
    
    # 2. Model speed test
    print("\n2. MODEL INFERENCE SPEED")
    print("-" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualUNet3D(in_channels=4, base_channels=24, residual=True).to(device)
    model.eval()
    
    batch_size = 2
    input_shape = (batch_size, 4, 96, 96, 96)
    x = torch.randn(*input_shape, device=device, dtype=torch.float32)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    
    # Time inference
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.time()
    
    inference_time = (t1 - t0) / 10
    print(f"  Batch size:            {batch_size}")
    print(f"  Input shape:           {input_shape}")
    print(f"  Inference time:        {inference_time*1000:.2f} ms/batch")
    print(f"  Expected speed:        {inference_time:.2f} s/iter (inference only)")
    
    # 3. Forward + Backward speed
    print("\n3. FORWARD + BACKWARD PASS SPEED")
    print("-" * 70)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    criterion = nn.L1Loss()
    
    # Dummy target
    target = torch.randn(batch_size, 1, 96, 96, 96, device=device, dtype=torch.float32)
    d_low = torch.randn(batch_size, 1, 96, 96, 96, device=device, dtype=torch.float32)
    
    # Warmup
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        delta = model(x)
        pred = d_low + delta
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
    
    # Time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        delta = model(x)
        pred = d_low + delta
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.time()
    
    fwd_bwd_time = (t1 - t0) / 5
    print(f"  Forward + Backward:    {fwd_bwd_time*1000:.2f} ms/batch")
    print(f"  Expected speed:        {fwd_bwd_time:.2f} s/iter (compute only)")
    
    # 4. Data loading speed test (if manifest exists)
    print("\n4. DATA LOADING SPEED")
    print("-" * 70)
    
    manifest_paths = [
        Path("/lustre/home/acastaneda/Fernando/PrAi/data/training_npz/manifest_train.csv"),
        Path("data/training_npz/manifest_train.csv"),
    ]
    
    manifest_found = None
    for mp in manifest_paths:
        if mp.exists():
            manifest_found = mp
            break
    
    if manifest_found:
        print(f"  Loading from:          {manifest_found}")
        
        import csv
        rows = list(csv.DictReader(manifest_found.open(encoding="utf-8")))
        npz_paths = [Path(r["npz_path"]) for r in rows[:32]]
        
        # Test loading speed
        print(f"  Testing {len(npz_paths)} files...")
        t0 = time.time()
        valid = 0
        for npz_path in npz_paths:
            try:
                case = load_case_npz(npz_path)
                valid += 1
            except Exception as e:
                print(f"    Error loading {npz_path.name}: {e}")
        t1 = time.time()
        
        load_time_per_file = (t1 - t0) / len(npz_paths)
        print(f"  Valid files:           {valid}/{len(npz_paths)}")
        print(f"  Load time per file:    {load_time_per_file*1000:.2f} ms")
        print(f"  Load time per batch:   {load_time_per_file * batch_size * 1000:.2f} ms")
    else:
        print(f"  Manifest not found. Skipping data loading test.")
    
    # 5. Expected vs Actual
    print("\n5. SPEED ESTIMATE")
    print("-" * 70)
    compute_time = fwd_bwd_time
    data_time = (t1 - t0) / len(npz_paths) * batch_size if manifest_found else 0.1
    
    expected_total = compute_time + data_time
    print(f"  Compute time:          {compute_time*1000:.2f} ms")
    print(f"  Data loading time:     {data_time*1000:.2f} ms")
    print(f"  Expected total:        {expected_total:.2f} s/iter")
    print(f"  Actual reported:       ~5.00 s/iter")
    print(f"  Slowdown factor:       ~{5.0 / expected_total:.1f}x")
    
    # 6. Recommendations
    print("\n6. RECOMMENDATIONS")
    print("-" * 70)
    
    if not torch.cuda.is_available():
        print("  ⚠️  CRITICAL: GPU not available! Check:")
        print("      - CUDA installation: python -c 'import torch; print(torch.cuda.is_available())'")
        print("      - GPU driver: nvidia-smi or rocm-smi")
        print("      - Torch installation: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    elif expected_total < 2.0:
        print("  ✓ Compute speed is good (~" + f"{compute_time:.2f} s/iter)")
        print("  ⚠️  Data loading may be the bottleneck")
        print("  → Try increasing num_workers or check disk I/O")
    elif expected_total < 5.0:
        print("  ✓ Expected speed is good (~" + f"{expected_total:.2f} s/iter)")
        print("  ⚠️  Actual is 5.0 s/iter - something is wrong")
        print("  → Check if num_workers is set correctly")
        print("  → Monitor GPU utilization: nvidia-smi -l 1")
    else:
        print("  ✗ Speed is significantly slower than expected")
        print("  → GPU may not be in use")
        print("  → Check torch device placement in training script")
        print("  → Verify .to(device) is called on model, inputs, targets")
    
    print("\n" + "=" * 70)
    print("HOW TO MONITOR DURING TRAINING:")
    print("=" * 70)
    print("  GPU utilization (NVIDIA):  nvidia-smi -l 1")
    print("  GPU utilization (AMD):     rocm-smi --watch")
    print("  Power usage:               nvidia-smi --query-gpu=power.draw --format=csv")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    diagnose()
