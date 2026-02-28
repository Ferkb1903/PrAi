#!/usr/bin/env python3
"""
Split pair_index.csv into chunks and prepare a SLURM job array script
for parallel NPZ generation.
"""

import argparse
import csv
from pathlib import Path
import sys


def split_csv(input_csv: Path, output_dir: Path, num_chunks: int) -> list[Path]:
    """Split CSV into chunks, return list of chunk file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read all rows
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    
    total = len(rows)
    chunk_size = (total + num_chunks - 1) // num_chunks  # ceil division
    
    chunk_files = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        
        if start >= total:
            break
        
        chunk_rows = rows[start:end]
        chunk_path = output_dir / f"chunk_{i:02d}.csv"
        
        with open(chunk_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(chunk_rows)
        
        chunk_files.append(chunk_path)
        print(f"Chunk {i:02d}: {len(chunk_rows)} rows → {chunk_path}")
    
    return chunk_files


def create_slurm_script(output_file: Path, num_chunks: int, project_root: Path, 
                       pair_index_csv: Path, out_dir: Path, dose_norm_const: float,
                       qc_report: Path, manifest_all: Path, manifest_train: Path,
                       manifest_val: Path, manifest_test: Path) -> None:
    """Create SLURM job array script."""
    
    script = f"""#!/bin/bash
#SBATCH --job-name=prepare_npz_parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-{num_chunks-1}
#SBATCH --gres=gpu:0
#SBATCH --output=logs/prepare_chunk_%a.log

set -e

PROJECT_ROOT="{project_root}"
CHUNKS_DIR="{pair_index_csv.parent / 'chunks'}"
OUT_DIR="{out_dir}"
DOSE_NORM_CONST={dose_norm_const}

# Activate environment
cd "$PROJECT_ROOT"
source .venv/bin/activate

# Export Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Create output directory
mkdir -p "$OUT_DIR"
mkdir -p logs

CHUNK_CSV="$CHUNKS_DIR/chunk_${{SLURM_ARRAY_TASK_ID:02d}}.csv"
OUT_CHUNK_DIR="$OUT_DIR/chunk_${{SLURM_ARRAY_TASK_ID:02d}}"

echo "Processing chunk ${{SLURM_ARRAY_TASK_ID}} from $CHUNK_CSV"
echo "Output: $OUT_CHUNK_DIR"

mkdir -p "$OUT_CHUNK_DIR"

# Run prepare_training_tensors for this chunk
python scripts/prepare_training_tensors.py \\
    --pair-index-csv "$CHUNK_CSV" \\
    --out-dir "$OUT_CHUNK_DIR" \\
    --dose-norm-const $DOSE_NORM_CONST \\
    --qc-report "$OUT_CHUNK_DIR/qc_report_chunk.csv" \\
    --manifest-all "$OUT_CHUNK_DIR/manifest_all.csv" \\
    --manifest-train "$OUT_CHUNK_DIR/manifest_train.csv" \\
    --manifest-val "$OUT_CHUNK_DIR/manifest_val.csv" \\
    --manifest-test "$OUT_CHUNK_DIR/manifest_test.csv"

echo "Chunk ${{SLURM_ARRAY_TASK_ID}} complete"
"""
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(script)
    output_file.chmod(0o755)
    print(f"\n✓ SLURM script: {output_file}")


def create_merge_script(output_file: Path, num_chunks: int, project_root: Path,
                       out_dir: Path, qc_report: Path, manifest_all: Path,
                       manifest_train: Path, manifest_val: Path, manifest_test: Path) -> None:
    """Create post-processing script to merge results."""
    
    script = f"""#!/bin/bash
# Merge NPZ and manifests from all chunks

PROJECT_ROOT="{project_root}"
OUT_DIR="{out_dir}"
NUM_CHUNKS={num_chunks}

echo "Merging results from $NUM_CHUNKS chunks..."

# Merge NPZ files: move from chunk subdirs to main out_dir
for chunk_dir in "$OUT_DIR"/chunk_*; do
    if [ -d "$chunk_dir" ]; then
        echo "Processing: $chunk_dir"
        find "$chunk_dir" -name "*.npz" -type f -exec mv {{}} "$OUT_DIR/" \\;
    fi
done

# Merge QC reports
echo "Merging QC reports..."
> "{qc_report}"
first=1
for chunk_dir in "$OUT_DIR"/chunk_*; do
    if [ -f "$chunk_dir/qc_report_chunk.csv" ]; then
        if [ $first -eq 1 ]; then
            cat "$chunk_dir/qc_report_chunk.csv" >> "{qc_report}"
            first=0
        else
            tail -n +2 "$chunk_dir/qc_report_chunk.csv" >> "{qc_report}"
        fi
    fi
done

# Merge manifests
echo "Merging manifests..."
for manifest in manifest_all manifest_train manifest_val manifest_test; do
    > "{out_dir}/${{manifest}}.csv"
    first=1
    for chunk_dir in "$OUT_DIR"/chunk_*; do
        chunk_manifest="$chunk_dir/${{manifest}}.csv"
        if [ -f "$chunk_manifest" ]; then
            if [ $first -eq 1 ]; then
                cat "$chunk_manifest" >> "{out_dir}/${{manifest}}.csv"
                first=0
            else
                tail -n +2 "$chunk_manifest" >> "{out_dir}/${{manifest}}.csv"
            fi
        fi
    done
done

# Clean up chunk directories
echo "Cleaning up chunk directories..."
rm -rf "$OUT_DIR"/chunk_*

echo "✓ Merge complete!"
echo "Results in: $OUT_DIR"
ls -lh "$OUT_DIR" | head -20
"""
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(script)
    output_file.chmod(0o755)
    print(f"✓ Merge script: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split CSV and create SLURM job array")
    parser.add_argument("--pair-index-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dose-norm-const", type=float, default=100.0)
    parser.add_argument("--num-chunks", type=int, default=10, help="Number of parallel chunks")
    parser.add_argument("--qc-report", type=Path, default=Path("data/training_npz/qc_spot_campaign.csv"))
    parser.add_argument("--manifest-all", type=Path, default=Path("data/training_npz/manifest_all.csv"))
    parser.add_argument("--manifest-train", type=Path, default=Path("data/training_npz/manifest_train.csv"))
    parser.add_argument("--manifest-val", type=Path, default=Path("data/training_npz/manifest_val.csv"))
    parser.add_argument("--manifest-test", type=Path, default=Path("data/training_npz/manifest_test.csv"))
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent
    
    # Step 1: Split CSV into chunks
    print(f"Splitting {args.pair_index_csv} into {args.num_chunks} chunks...")
    chunks_dir = args.pair_index_csv.parent / "chunks"
    chunk_files = split_csv(args.pair_index_csv, chunks_dir, args.num_chunks)
    
    # Step 2: Create SLURM job array script
    slurm_script = project_root / "scripts" / "submit_prepare_parallel.sh"
    create_slurm_script(
        slurm_script, 
        len(chunk_files),
        project_root,
        args.pair_index_csv,
        args.out_dir,
        args.dose_norm_const,
        args.qc_report,
        args.manifest_all,
        args.manifest_train,
        args.manifest_val,
        args.manifest_test
    )
    
    # Step 3: Create merge script
    merge_script = project_root / "scripts" / "merge_npz_chunks.sh"
    create_merge_script(
        merge_script,
        len(chunk_files),
        project_root,
        args.out_dir,
        args.qc_report,
        args.manifest_all,
        args.manifest_train,
        args.manifest_val,
        args.manifest_test
    )
    
    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print(f"{'='*70}")
    print(f"1. Submit parallel jobs:")
    print(f"   sbatch {slurm_script}")
    print(f"\n2. After jobs complete, merge results:")
    print(f"   bash {merge_script}")
    print(f"\n3. Monitor job status:")
    print(f"   squeue -u $USER")
    print(f"   tail -f logs/prepare_chunk_*.log")
    print(f"\n4. Expected speedup: ~{args.num_chunks}x (with {args.num_chunks} parallel chunks)")
