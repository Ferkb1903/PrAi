#!/bin/bash
# Check all NPZ directories and their normalization status

PROJECT_ROOT="/lustre/home/acastaneda/Fernando/PrAi"
cd "$PROJECT_ROOT"

echo "========================================================================"
echo "CHECKING ALL NPZ DIRECTORIES"
echo "========================================================================"

# Directory 1: Original sequential
echo ""
echo "1. ORIGINAL (spot_campaign_v2) - Sequential run:"
if [ -d "data/training_npz/spot_campaign_v2" ]; then
    count=$(ls -1 data/training_npz/spot_campaign_v2/*.npz 2>/dev/null | wc -l)
    echo "   Files: $count"
    if [ -f "data/training_npz/split_summary.json" ]; then
        dose_norm=$(cat data/training_npz/split_summary.json | grep dose_norm_const | head -1)
        echo "   Dose config: $dose_norm"
    fi
    # Show timestamp of first file
    first=$(ls -1 data/training_npz/spot_campaign_v2/*.npz 2>/dev/null | head -1)
    if [ -n "$first" ]; then
        echo "   First file timestamp:"
        stat "$first" 2>/dev/null | grep Modify || ls -lh "$first"
    fi
else
    echo "   NOT FOUND"
fi

# Directory 2: Parallel output
echo ""
echo "2. PARALLEL (spot_campaign_v2_parallel) - Job array run:"
if [ -d "data/training_npz/spot_campaign_v2_parallel" ]; then
    count=$(find data/training_npz/spot_campaign_v2_parallel -name "*.npz" 2>/dev/null | wc -l)
    echo "   Total files (including chunks): $count"
    
    # Check if merged (files at root)
    root_count=$(ls -1 data/training_npz/spot_campaign_v2_parallel/*.npz 2>/dev/null | wc -l)
    chunk_count=$(ls -1d data/training_npz/spot_campaign_v2_parallel/chunk_* 2>/dev/null | wc -l)
    echo "   Files at root: $root_count"
    echo "   Chunk directories: $chunk_count"
    
    # Check an example chunk
    if [ $chunk_count -gt 0 ]; then
        first_chunk=$(ls -1d data/training_npz/spot_campaign_v2_parallel/chunk_* | head -1)
        if [ -d "$first_chunk" ]; then
            echo "   Example chunk: $first_chunk"
            if [ -f "$first_chunk/split_summary.json" ]; then
                dose_norm=$(cat "$first_chunk/split_summary.json" | grep dose_norm_const | head -1)
                echo "     Dose config: $dose_norm"
            fi
            chunk_files=$(ls -1 "$first_chunk"/*.npz 2>/dev/null | wc -l)
            echo "     NPZ files in chunk: $chunk_files"
        fi
    fi
    
    # Show newest file timestamp
    newest=$(find data/training_npz/spot_campaign_v2_parallel -name "*.npz" -type f 2>/dev/null | xargs ls -t | head -1)
    if [ -n "$newest" ]; then
        echo "   Newest file timestamp:"
        stat "$newest" 2>/dev/null | grep Modify || ls -lh "$newest"
    fi
else
    echo "   NOT FOUND (parallel jobs may not have completed)"
fi

# Directory 3: Test output
echo ""
echo "3. TEST FIX (spot_campaign_v2_norm_fix) - Manual single test:"
if [ -d "data/training_npz/spot_campaign_v2_norm_fix" ]; then
    count=$(ls -1 data/training_npz/spot_campaign_v2_norm_fix/*.npz 2>/dev/null | wc -l)
    echo "   Files: $count"
    if [ -f "data/training_npz/spot_campaign_v2_norm_fix/split_summary.json" ]; then
        dose_norm=$(cat data/training_npz/spot_campaign_v2_norm_fix/split_summary.json | grep dose_norm_const)
        echo "   Dose config: $dose_norm"
    fi
else
    echo "   NOT FOUND"
fi

echo ""
echo "========================================================================"
echo "TO DOWNLOAD AND TEST:"
echo "========================================================================"
echo ""
echo "# From your local machine:"
echo "scp user@cluster:'/lustre/home/acastaneda/Fernando/PrAi/data/training_npz/spot_campaign_v2_parallel/colorectal_*.npz' ./"
echo "python scripts/diagnose_npz_normalization.py colorectal_*.npz"
echo ""
