#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def repair_manifest(manifest_path: Path, npz_root: Path, in_place: bool = True) -> tuple[int, int]:
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    if not rows:
        return 0, 0

    fixed = 0
    missing = 0
    output_rows: list[dict[str, str]] = []

    for row in rows:
        p = Path(row["npz_path"])
        if p.exists():
            output_rows.append(row)
            continue

        candidate = npz_root / p.name
        if candidate.exists():
            row["npz_path"] = str(candidate)
            fixed += 1
        else:
            missing += 1
        output_rows.append(row)

    if in_place:
        backup = manifest_path.with_suffix(manifest_path.suffix + ".bak")
        manifest_path.replace(backup)
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
            w.writeheader()
            w.writerows(output_rows)

    return fixed, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair manifest NPZ paths after merging chunk outputs")
    parser.add_argument("--npz-root", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    total_fixed = 0
    total_missing = 0
    for manifest in args.manifests:
        fixed, missing = repair_manifest(manifest, args.npz_root, in_place=True)
        total_fixed += fixed
        total_missing += missing
        print(f"{manifest}: fixed={fixed}, missing={missing}")

    print(f"TOTAL fixed={total_fixed}, missing={total_missing}")


if __name__ == "__main__":
    main()
