#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def chunk_items(items: list[Path], num_chunks: int) -> list[list[Path]]:
    if num_chunks <= 0:
        raise ValueError("num_chunks debe ser > 0")
    chunk_size = (len(items) + num_chunks - 1) // num_chunks
    chunks: list[list[Path]] = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(items))
        if start >= len(items):
            break
        chunks.append(items[start:end])
    return chunks


def write_chunks(npz_dir: Path, pattern: str, chunks_dir: Path, num_chunks: int) -> list[Path]:
    files = sorted(npz_dir.glob(pattern))
    if not files:
        raise RuntimeError(f"No hay archivos {pattern} en {npz_dir}")

    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: list[Path] = []
    for idx, group in enumerate(chunk_items(files, num_chunks)):
        chunk_path = chunks_dir / f"npz_chunk_{idx:03d}.txt"
        chunk_path.write_text("\n".join(str(p.resolve()) for p in group) + "\n", encoding="utf-8")
        chunk_paths.append(chunk_path)
        print(f"Chunk {idx:03d}: {len(group)} archivos -> {chunk_path}")

    return chunk_paths


def write_slurm_script(
    script_path: Path,
    project_root: Path,
    chunks_dir: Path,
    array_max: int,
    dose_norm_const: float,
    dry_run: bool,
) -> None:
    dry_run_flag = "--dry-run" if dry_run else ""
    content = f"""#!/bin/bash
#SBATCH --job-name=retro_norm_npz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-{array_max}
#SBATCH --output=logs/retro_norm_%a.log

set -euo pipefail

PROJECT_ROOT=\"{project_root}\"
CHUNKS_DIR=\"{chunks_dir}\"
DOSE_NORM_CONST={dose_norm_const}

cd \"$PROJECT_ROOT\"
source .venv/bin/activate
mkdir -p logs

CHUNK_ID=$(printf \"%03d\" ${{SLURM_ARRAY_TASK_ID}})
CHUNK_FILE=\"$CHUNKS_DIR/npz_chunk_${{CHUNK_ID}}.txt\"

if [[ ! -f \"$CHUNK_FILE\" ]]; then
  echo \"No existe chunk: $CHUNK_FILE\"
  exit 1
fi

echo \"Procesando chunk $CHUNK_ID -> $CHUNK_FILE\"
python scripts/retrospectively_normalize_npz.py \\
  --file-list \"$CHUNK_FILE\" \\
  --dose-norm-const \"$DOSE_NORM_CONST\" \\
  {dry_run_flag}
"""
    script_path.write_text(content, encoding="utf-8")
    script_path.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crear job-array SLURM para normalización retrospectiva NPZ")
    parser.add_argument("--npz-dir", type=Path, required=True)
    parser.add_argument("--pattern", type=str, default="*.npz")
    parser.add_argument("--num-chunks", type=int, default=20)
    parser.add_argument("--dose-norm-const", type=float, default=100.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chunks-dir", type=Path, default=Path("cluster_jobs/retro_norm_chunks"))
    parser.add_argument("--slurm-script", type=Path, default=Path("scripts/submit_retro_norm_parallel.sh"))
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    npz_dir = args.npz_dir.resolve()

    chunk_files = write_chunks(npz_dir, args.pattern, args.chunks_dir.resolve(), args.num_chunks)
    write_slurm_script(
        script_path=args.slurm_script.resolve(),
        project_root=project_root.resolve(),
        chunks_dir=args.chunks_dir.resolve(),
        array_max=len(chunk_files) - 1,
        dose_norm_const=float(args.dose_norm_const),
        dry_run=bool(args.dry_run),
    )

    print("\nListo.")
    print(f"Chunks: {len(chunk_files)}")
    print(f"SLURM: {args.slurm_script}")
    print(f"Submit: sbatch {args.slurm_script}")


if __name__ == "__main__":
    main()
