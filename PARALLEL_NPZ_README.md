# Parallel NPZ Preparation with SLURM

## Overview

Instead of processing 2460 NPZ files sequentially (~30-40 min), use SLURM job arrays to process multiple chunks in parallel.

**Expected speedup**: ~10x with 10 parallel jobs (3-4 minutes total)

## Files

- `split_and_prepare_parallel.py` - Splits CSV and creates job array scripts
- `run_prepare_parallel.sh` - One-liner to run everything
- `submit_prepare_parallel.sh` - SLURM job array script (auto-generated)
- `merge_npz_chunks.sh` - Merges results from all chunks (auto-generated)

## Quick Start

### On the cluster:

```bash
cd /lustre/home/acastaneda/Fernando/PrAi

# Activate venv
source .venv/bin/activate

# Option A: Run everything automatically
bash scripts/run_prepare_parallel.sh

# Wait for jobs to complete, then merge:
bash scripts/merge_npz_chunks.sh
```

### Or step-by-step:

```bash
# Step 1: Create 10 chunks and SLURM scripts
python scripts/split_and_prepare_parallel.py \
    --pair-index-csv cluster_jobs/spot_campaign_3060/pair_index.csv \
    --out-dir data/training_npz/spot_campaign_v2_parallel \
    --dose-norm-const 100.0 \
    --num-chunks 10

# Step 2: Submit job array (10 parallel jobs)
sbatch scripts/submit_prepare_parallel.sh

# Step 3: Monitor progress
squeue -u $USER
tail -f logs/prepare_chunk_*.log

# Step 4: After all jobs complete, merge results
bash scripts/merge_npz_chunks.sh
```

## How It Works

### Split Phase
- `split_and_prepare_parallel.py` divides `pair_index.csv` into 10 equal chunks
- Creates `cluster_jobs/spot_campaign_3060/chunks/chunk_00.csv`, `chunk_01.csv`, etc.

### Parallel Phase
- SLURM submits 10 independent jobs (one per chunk)
- Each job processes its chunk independently
- Each outputs NPZ files + local manifest CSVs to `chunk_XX/` subdirectories

### Merge Phase
- `merge_npz_chunks.sh` combines:
  - All `.npz` files → `data/training_npz/spot_campaign_v2_parallel/`
  - All QC reports → single `qc_spot_campaign.csv`
  - All manifests → `manifest_all.csv`, `manifest_train.csv`, etc.

## Customization

Change number of chunks:
```bash
python scripts/split_and_prepare_parallel.py \
    --pair-index-csv cluster_jobs/spot_campaign_3060/pair_index.csv \
    --out-dir data/training_npz/spot_campaign_v2_parallel \
    --num-chunks 20  # Use 20 chunks instead of 10
```

## Example Output

```
Chunk 00: 246 rows → cluster_jobs/spot_campaign_3060/chunks/chunk_00.csv
Chunk 01: 246 rows → cluster_jobs/spot_campaign_3060/chunks/chunk_01.csv
...
Chunk 09: 246 rows → cluster_jobs/spot_campaign_3060/chunks/chunk_09.csv

✓ SLURM script: scripts/submit_prepare_parallel.sh
✓ Merge script: scripts/merge_npz_chunks.sh

=======================================================================
NEXT STEPS:
=======================================================================
1. Submit parallel jobs:
   sbatch scripts/submit_prepare_parallel.sh

2. After jobs complete, merge results:
   bash scripts/merge_npz_chunks.sh

3. Monitor job status:
   squeue -u $USER
   tail -f logs/prepare_chunk_*.log

4. Expected speedup: ~10x (with 10 parallel chunks)
```

## Performance

| Configuration | Time | Speedup |
|---|---|---|
| Sequential (original) | 30-40 min | 1x |
| 10 parallel chunks | 3-4 min | ~10x |
| 20 parallel chunks | 1.5-2 min | ~20x |

## Troubleshooting

**Job fails?** Check logs:
```bash
cat logs/prepare_chunk_00.log
```

**Missing manifests after merge?** 
- Some chunks may have no training/val/test splits
- Check: `wc -l manifest_train.csv`

**Want to rerun?** Clean up first:
```bash
rm -rf data/training_npz/spot_campaign_v2_parallel/
rm -rf cluster_jobs/spot_campaign_3060/chunks/
```
