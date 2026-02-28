#!/bin/bash
# Example: Run prepare_training_tensors in parallel on MI210 cluster using SLURM
# This splits the work into 10 parallel jobs for ~10x speedup

set -e

PROJECT_ROOT="/lustre/home/acastaneda/Fernando/PrAi"
cd "$PROJECT_ROOT"

# Activate venv
source .venv/bin/activate

echo "===== STEP 1: Splitting CSV and creating SLURM job array ====="
python scripts/split_and_prepare_parallel.py \
    --pair-index-csv cluster_jobs/spot_campaign_3060/pair_index.csv \
    --out-dir data/training_npz/spot_campaign_v2_parallel \
    --dose-norm-const 100.0 \
    --num-chunks 10 \
    --qc-report data/training_npz/qc_spot_campaign.csv \
    --manifest-all data/training_npz/manifest_all.csv \
    --manifest-train data/training_npz/manifest_train.csv \
    --manifest-val data/training_npz/manifest_val.csv \
    --manifest-test data/training_npz/manifest_test.csv

echo ""
echo "===== STEP 2: Submitting parallel jobs ====="
sbatch scripts/submit_prepare_parallel.sh

echo ""
echo "===== WAITING FOR JOBS (YOU CAN ALSO Ctrl+C AND USE: squeue -u \$USER) ====="
echo "Monitor jobs with: squeue -u \$USER"
echo "View logs with: tail -f logs/prepare_chunk_*.log"
echo ""
echo "After jobs complete, run:"
echo "  bash scripts/merge_npz_chunks.sh"
