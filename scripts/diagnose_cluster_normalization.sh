#!/bin/bash
# Run this ON THE CLUSTER to diagnose normalization status
# Usage: bash scripts/diagnose_cluster_normalization.sh

PROJECT_ROOT="/lustre/home/acastaneda/Fernando/PrAi"
cd "$PROJECT_ROOT"

source .venv/bin/activate

echo "========================================================================"
echo "CLUSTER NORMALIZATION DIAGNOSTIC"
echo "========================================================================"

# Check if parallel output directory exists
echo ""
echo "1. Checking parallel output directory:"
parallel_dir="data/training_npz/spot_campaign_v2_parallel"
if [ -d "$parallel_dir" ]; then
    echo "✓ Found: $parallel_dir"
    echo "  Files:"
    ls -lh "$parallel_dir"/*.npz 2>/dev/null | wc -l
    echo "  npz files"
    
    # Get first file and its timestamp
    first_file=$(ls -1 "$parallel_dir"/*.npz 2>/dev/null | head -1)
    if [ -n "$first_file" ]; then
        echo ""
        echo "  Most recent file:"
        ls -lh "$first_file"
        echo ""
        echo "  File timestamp:"
        stat "$first_file" | grep Modify
    fi
else
    echo "✗ Not found: $parallel_dir (parallel jobs may not have completed)"
fi

# Check original output directory
echo ""
echo "2. Checking original output directory:"
orig_dir="data/training_npz/spot_campaign_v2"
if [ -d "$orig_dir" ]; then
    echo "✓ Found: $orig_dir"
    num_files=$(ls -1 "$orig_dir"/*.npz 2>/dev/null | wc -l)
    echo "  Total .npz files: $num_files"
    
    first_file=$(ls -1 "$orig_dir"/*.npz 2>/dev/null | head -1)
    if [ -n "$first_file" ]; then
        echo "  First file: $(basename $first_file)"
        echo "  Timestamp:"
        stat "$first_file" | grep Modify
    fi
else
    echo "✗ Not found: $orig_dir"
fi

# Check job status
echo ""
echo "3. Job status:"
squeue -u $USER | grep -E "prepare_npz|prepare_parallel" || echo "No active prepare jobs"

# Look for logs
echo ""
echo "4. Log files:"
if [ -d "logs" ]; then
    echo "  Recent logs:"
    ls -lht logs/prepare_chunk_*.log 2>/dev/null | head -5 || echo "  No logs found"
else
    echo "  logs/ directory not found"
fi

# Diagnostics: Check a sample file if it exists
echo ""
echo "5. Sample file analysis:"
sample_file=$(ls -1 "$parallel_dir"/*.npz 2>/dev/null | head -1)
if [ -z "$sample_file" ]; then
    sample_file=$(ls -1 "$orig_dir"/*.npz 2>/dev/null | head -1)
fi

if [ -n "$sample_file" ]; then
    echo "  Testing: $(basename $sample_file)"
    python scripts/diagnose_npz_normalization.py "$sample_file" --norm-const 100.0 2>&1 | head -40
else
    echo "  No NPZ files found to test"
fi

echo ""
echo "========================================================================"
echo "NEXT STEPS:"
echo "========================================================================"
echo "If parallel jobs still running:"
echo "  1. Wait for completion: watch squeue -u \$USER"
echo "  2. Then run: bash scripts/merge_npz_chunks.sh"
echo "  3. Then run this diagnostic again"
echo ""
echo "If jobs completed but files not normalized:"
echo "  1. Check logs: cat logs/prepare_chunk_*.log | grep -i 'norm\|dose'"
echo "  2. Re-run with explicit dose normalization:"
echo "     python scripts/prepare_training_tensors.py \\"
echo "         --pair-index-csv cluster_jobs/spot_campaign_3060/pair_index.csv \\"
echo "         --out-dir data/training_npz/spot_campaign_v2_norm_fix \\"
echo "         --dose-norm-const 100.0 \\"
echo "         --limit 10  # Test with first 10 pairs only"
