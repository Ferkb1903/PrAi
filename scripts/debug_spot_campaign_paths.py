from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable


def normalize_rel_path(value: str) -> str:
    return value.strip().strip('"').replace("\r", "")


def escaped(path: Path) -> str:
    return str(path).encode("unicode_escape", errors="backslashreplace").decode("ascii", errors="ignore")


def list_samples(paths: Iterable[Path], limit: int = 10) -> list[Path]:
    out = []
    for index, item in enumerate(paths):
        if index >= limit:
            break
        out.append(item)
    return out


def has_dose_files(folder: Path) -> bool:
    for ext in (".raw", ".mhd"):
        target = folder / f"dose_voxelized_ct_edep{ext}"
        if target.is_file() and target.stat().st_size > 0:
            return True
    return False


def file_listing(folder: Path) -> list[str]:
    if not folder.exists() or not folder.is_dir():
        return []
    output: list[str] = []
    for item in sorted(folder.iterdir()):
        if item.is_file():
            output.append(f"{item.name} size={item.stat().st_size}")
        elif item.is_dir():
            output.append(f"{item.name}/")
    return output


def audit_single_case(spot_root: Path, case_id: str) -> None:
    case_dir = spot_root / case_id
    print("\n=== SINGLE CASE AUDIT ===")
    print(f"case_dir={escaped(case_dir)} exists={case_dir.exists()}")
    if not case_dir.exists():
        return

    energy_dirs = sorted([p for p in case_dir.iterdir() if p.is_dir()])
    print(f"energies_found={len(energy_dirs)}")

    total_spots = 0
    total_low_ok = 0
    total_high_ok = 0
    total_high_missing_dir = 0

    for energy_dir in energy_dirs:
        spot_dirs = sorted([p for p in energy_dir.iterdir() if p.is_dir() and p.name.startswith("spot_")])
        print(f"\nenergy={energy_dir.name} spot_count={len(spot_dirs)}")

        for spot_dir in spot_dirs:
            total_spots += 1
            low_dir = spot_dir / "low"
            high_dir = spot_dir / "high"

            low_exists = low_dir.exists()
            high_exists = high_dir.exists()
            low_ok = has_dose_files(low_dir)
            high_ok = has_dose_files(high_dir)

            total_low_ok += int(low_ok)
            total_high_ok += int(high_ok)
            if not high_exists:
                total_high_missing_dir += 1

            print(
                f"spot={spot_dir.name} "
                f"low_exists={int(low_exists)} low_dose_ok={int(low_ok)} "
                f"high_exists={int(high_exists)} high_dose_ok={int(high_ok)}"
            )
            print(f"  checked_low_path={escaped(low_dir)}")
            if low_exists:
                low_files = file_listing(low_dir)
                print(f"  low_files_count={len(low_files)}")
                for row in low_files:
                    print(f"    - {row}")

            print(f"  checked_high_path={escaped(high_dir)}")
            if high_exists:
                high_files = file_listing(high_dir)
                print(f"  high_files_count={len(high_files)}")
                for row in high_files:
                    print(f"    - {row}")

    print("\n=== SINGLE CASE SUMMARY ===")
    print(f"case_id={case_id}")
    print(f"total_spots={total_spots}")
    print(f"low_dose_ok={total_low_ok}")
    print(f"high_dose_ok={total_high_ok}")
    print(f"high_missing_dir={total_high_missing_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug path resolution for spot campaign low/high outputs")
    parser.add_argument("--root", type=Path, required=True, help="Project root (the folder that contains outputs/spot_campaign)")
    parser.add_argument("--pair-csv", type=Path, default=None, help="Optional pair_index.csv for row-by-row resolution checks")
    parser.add_argument("--case-id", type=str, default="", help="Optional case_id for a focused check")
    parser.add_argument("--energy", type=str, default="", help="Optional energy token like E160")
    parser.add_argument("--spot", type=str, default="", help="Optional spot token like spot_003")
    parser.add_argument("--rows", type=int, default=30, help="Rows to sample from pair csv")
    parser.add_argument("--single-case", type=str, default="", help="If set, prints full audit for this case_id")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    spot_root = root / "outputs" / "spot_campaign"

    print(f"ROOT: {escaped(root)}")
    print(f"SPOT_ROOT: {escaped(spot_root)} exists={spot_root.exists()}")

    if not spot_root.exists():
        print("ERROR: outputs/spot_campaign does not exist under provided root")
        return

    cases = sorted([p for p in spot_root.iterdir() if p.is_dir()])
    print(f"case_dirs={len(cases)}")
    for sample in list_samples(cases, limit=5):
        print(f"  case_sample: {escaped(sample)}")

    strict_low = sorted(spot_root.glob("*/E*/spot_*/low"))
    strict_high = sorted(spot_root.glob("*/E*/spot_*/high"))
    deep_low = sorted([p for p in spot_root.rglob("low") if p.is_dir() and p.parent.name.startswith("spot_")])
    deep_high = sorted([p for p in spot_root.rglob("high") if p.is_dir() and p.parent.name.startswith("spot_")])

    print(f"strict_low={len(strict_low)} strict_high={len(strict_high)}")
    print(f"deep_low={len(deep_low)} deep_high={len(deep_high)}")

    if strict_high:
        print("strict_high sample:")
        for sample in list_samples(strict_high, limit=5):
            print(f"  {escaped(sample)} dose_ok={has_dose_files(sample)}")
    else:
        print("strict_high sample: <none>")

    if args.case_id and args.energy and args.spot:
        base = spot_root / args.case_id / args.energy / args.spot
        low_dir = base / "low"
        high_dir = base / "high"
        print("\nFocused folder check")
        print(f"  low_dir={escaped(low_dir)} exists={low_dir.exists()} dose_ok={has_dose_files(low_dir) if low_dir.exists() else False}")
        if low_dir.exists():
            print(f"  low_files={[p.name for p in sorted(low_dir.iterdir())[:10]]}")
        print(f"  high_dir={escaped(high_dir)} exists={high_dir.exists()} dose_ok={has_dose_files(high_dir) if high_dir.exists() else False}")
        if high_dir.exists():
            print(f"  high_files={[p.name for p in sorted(high_dir.iterdir())[:10]]}")

    if args.pair_csv:
        pair_csv = args.pair_csv.expanduser().resolve()
        print(f"\nPAIR_CSV: {escaped(pair_csv)} exists={pair_csv.exists()}")
        if not pair_csv.exists():
            print("ERROR: pair csv path does not exist")
            return

        with pair_csv.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        print(f"rows_total={len(rows)} sampling_first={min(args.rows, len(rows))}")
        for index, row in enumerate(rows[: args.rows]):
            low_out = normalize_rel_path(row.get("low_out", ""))
            high_out = normalize_rel_path(row.get("high_out", ""))

            low_dir = (root / low_out).resolve()
            high_dir = (root / high_out).resolve()

            low_ok = has_dose_files(low_dir)
            high_ok = has_dose_files(high_dir)

            print(
                f"row={index} low_ok={int(low_ok)} high_ok={int(high_ok)} "
                f"low_dir={escaped(low_dir)} high_dir={escaped(high_dir)}"
            )

            if not high_ok:
                parent_exists = high_dir.parent.exists()
                print(
                    f"  high_missing_debug: parent_exists={int(parent_exists)} "
                    f"high_dir_exists={int(high_dir.exists())} high_out_raw={row.get('high_out', '')!r}"
                )

    if args.single_case:
        audit_single_case(spot_root, args.single_case)


if __name__ == "__main__":
    main()
