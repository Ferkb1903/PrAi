from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scripts.generate_slab_curriculum_dataset import split_rows, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge chunk manifests and build train/val/test splits")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--stage-name", type=str, default="stage2_medium")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    args = parser.parse_args()

    stage_dir = args.out_root / args.stage_name
    chunks_dir = stage_dir / "chunks"
    manifests = sorted(chunks_dir.glob("chunk_*/manifest_all.csv"))

    if not manifests:
        raise FileNotFoundError(f"No chunk manifests found under: {chunks_dir}")

    rows: list[dict[str, str]] = []
    for mf in manifests:
        with mf.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({"npz_path": r["npz_path"], "stage": r.get("stage", args.stage_name), "case_id": r["case_id"]})

    rows = sorted(rows, key=lambda r: r["case_id"])
    train_rows, val_rows, test_rows = split_rows(rows, seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac)

    write_manifest(rows, stage_dir / "manifest_all.csv")
    write_manifest(train_rows, stage_dir / "manifest_train.csv")
    write_manifest(val_rows, stage_dir / "manifest_val.csv")
    write_manifest(test_rows, stage_dir / "manifest_test.csv")

    print(f"Merged cases: {len(rows)}")
    print(f"Train/Val/Test: {len(train_rows)}/{len(val_rows)}/{len(test_rows)}")


if __name__ == "__main__":
    main()
