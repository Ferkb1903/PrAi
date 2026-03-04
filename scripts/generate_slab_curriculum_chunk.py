from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripts.generate_slab_curriculum_dataset import (
    generate_case,
    load_config,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a chunk of slab curriculum cases for Slurm array")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--stage-name", type=str, default="stage2_medium")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--cases-per-task", type=int, default=3)
    parser.add_argument("--total-cases", type=int, default=2500)
    args = parser.parse_args()

    if args.task_id < 0:
        raise ValueError("--task-id must be >= 0")
    if args.cases_per_task <= 0:
        raise ValueError("--cases-per-task must be > 0")

    seed, shape, spacing_mm, beam_axis, lateral_sigma_mm, beam_radius_mm, stages = load_config(args.config)
    stage = next((s for s in stages if s.name == args.stage_name), None)
    if stage is None:
        raise ValueError(f"Stage not found in config: {args.stage_name}")

    start_idx = args.task_id * args.cases_per_task
    end_idx = min(start_idx + args.cases_per_task, args.total_cases)

    if start_idx >= args.total_cases:
        print(f"task={args.task_id}: no work (start={start_idx} >= total={args.total_cases})")
        return

    stage_seed = seed + args.task_id * 10_000
    rng = np.random.default_rng(stage_seed)

    chunk_dir = args.out_root / args.stage_name / "chunks" / f"chunk_{args.task_id:04d}"
    npz_dir = chunk_dir / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []

    for global_idx in range(start_idx, end_idx):
        case_id = f"{args.stage_name}_case_{global_idx:05d}"
        payload, _meta = generate_case(
            case_id=case_id,
            stage=stage,
            shape=shape,
            spacing_mm=spacing_mm,
            beam_axis=beam_axis,
            lateral_sigma_mm=lateral_sigma_mm,
            beam_radius_mm=beam_radius_mm,
            rng=rng,
        )
        npz_path = npz_dir / f"{case_id}.npz"
        np.savez_compressed(npz_path, **payload)
        rows.append(
            {
                "npz_path": str(npz_path.resolve()),
                "stage": args.stage_name,
                "case_id": case_id,
            }
        )

    write_manifest(rows, chunk_dir / "manifest_all.csv")
    print(f"task={args.task_id}: generated {len(rows)} cases ({start_idx}..{end_idx-1})")


if __name__ == "__main__":
    main()
