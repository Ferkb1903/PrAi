from __future__ import annotations

import argparse
import csv
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug path resolution for spot campaign low/high outputs")
    parser.add_argument("--root", type=Path, required=True, help="Project root (the folder that contains outputs/spot_campaign)")
    parser.add_argument("--pair-csv", type=Path, default=None, help="Optional pair_index.csv for row-by-row resolution checks")
    parser.add_argument("--case-id", type=str, default="", help="Optional case_id for a focused check")
    parser.add_argument("--energy", type=str, default="", help="Optional energy token like E160")
    parser.add_argument("--spot", type=str, default="", help="Optional spot token like spot_003")
    parser.add_argument("--rows", type=int, default=30, help="Rows to sample from pair csv")
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


if __name__ == "__main__":
    main()
