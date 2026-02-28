#!/bin/bash
# Investigate what parameters are being used for dose normalization

PROJECT_ROOT="/lustre/home/acastaneda/Fernando/PrAi"
cd "$PROJECT_ROOT"

echo "========================================================================"
echo "INVESTIGATION: What dose_norm_const is being used?"
echo "========================================================================"

# Check 1: Look at actual parameters in split_summary.json files
echo ""
echo "1. ACTUAL SAVED CONFIGURATION (split_summary.json):"
echo "───────────────────────────────────────────────────"

if [ -f "data/training_npz/split_summary.json" ]; then
    echo "Original directory:"
    cat data/training_npz/split_summary.json | grep -A1 dose_norm_const
else
    echo "data/training_npz/split_summary.json not found"
fi

echo ""
if [ -d "data/training_npz/spot_campaign_v2_parallel" ]; then
    echo "Parallel chunks:"
    for chunk_dir in data/training_npz/spot_campaign_v2_parallel/chunk_*/; do
        name=$(basename "$chunk_dir")
        if [ -f "$chunk_dir/split_summary.json" ]; then
            val=$(cat "$chunk_dir/split_summary.json" | grep -o '"dose_norm_const": [0-9.]*' | head -1)
            echo "  $name: $val"
        fi
    done | head -10
fi

# Check 2: Look at actual values in any NPZ file
echo ""
echo "2. ACTUAL DATA IN NPZ FILES:"
echo "───────────────────────────"
source .venv/bin/activate

# Try original directory
if [ -f "data/training_npz/spot_campaign_v2/colorectal_*.npz" ]; then
    sample=$(ls -1 data/training_npz/spot_campaign_v2/colorectal_*.npz 2>/dev/null | head -1)
    if [ -n "$sample" ]; then
        echo "Sample: $(basename $sample)"
        python scripts/diagnose_npz_normalization.py "$sample" --norm-const 100.0 2>&1 | grep "d_high_max\|dose_scale\|dose_norm_const"
    fi
fi

# Check 3: Look at recent log files
echo ""
echo "3. RECENT LOG OUTPUT:"
echo "───────────────────────────"
if [ -d "logs" ]; then
    most_recent=$(ls -t logs/prepare_chunk_*.log 2>/dev/null | head -1)
    if [ -n "$most_recent" ]; then
        echo "Most recent: $(basename $most_recent)"
        echo ""
        echo "Searching for normalization messages in log:"
        grep -i "dose\|normaliz" "$most_recent" | head -20
    fi
else
    echo "No logs directory found"
fi

# Check 4: Show the EXACT command that would be sent to SLURM
echo ""
echo "4. SLURM JOB ARRAY SCRIPT (if exists):"
echo "───────────────────────────────────────"
if [ -f "scripts/submit_prepare_parallel.sh" ]; then
    echo "Checking DOSE_NORM_CONST variable in SLURM script:"
    grep "DOSE_NORM_CONST" scripts/submit_prepare_parallel.sh
    echo ""
    echo "Full python command in SLURM script:"
    grep "python scripts/prepare_training_tensors.py" scripts/submit_prepare_parallel.sh -A 10 | head -20
fi

echo ""
echo "========================================================================"
echo ""
echo "HYPOTHESIS TO TEST:"
echo "  If dose_norm_const in split_summary.json is 1.0 → Parameter not passed"
echo "  If dose_norm_const in split_summary.json is 100.0 → Parameter passed but not applied"
echo ""
