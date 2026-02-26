from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_energy_list(txt: str) -> list[int]:
    return [int(x.strip()) for x in txt.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build training-ready manifest including only complete cases"
    )
    parser.add_argument(
        "--sim-ready-manifest",
        type=Path,
        default=Path("data/ct_cases_by_case/manifest_sim_ready.csv"),
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs/cluster_runs"),
    )
    parser.add_argument(
        "--energies",
        type=str,
        default="90,120,150,180,210",
    )
    parser.add_argument(
        "--dose-file",
        type=str,
        default="dose_voxelized_ct_edep.mhd",
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=Path("data/ct_cases_by_case/manifest_training_ready.csv"),
    )
    parser.add_argument(
        "--out-missing",
        type=Path,
        default=Path("data/ct_cases_by_case/missing_for_training.csv"),
    )
    args = parser.parse_args()

    if not args.sim_ready_manifest.exists():
        raise FileNotFoundError(f"Missing input manifest: {args.sim_ready_manifest}")

    energies = parse_energy_list(args.energies)
    rows = list(csv.DictReader(args.sim_ready_manifest.open(encoding="utf-8")))
    ready_rows = [r for r in rows if str(r.get("ready_for_sim", "0")) == "1"]

    missing_records: list[dict[str, str]] = []
    complete_rows: list[dict[str, str]] = []

    for row in ready_rows:
        case_id = row["case_id"]
        case_missing = False

        for mode in ("low", "high"):
            for e in energies:
                p = args.outputs_root / case_id / mode / f"E{e}" / args.dose_file
                if not p.exists():
                    case_missing = True
                    missing_records.append(
                        {
                            "case_id": case_id,
                            "mode": mode,
                            "energy": str(e),
                            "missing_path": str(p),
                        }
                    )

        if not case_missing:
            complete_rows.append(row)

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_missing.parent.mkdir(parents=True, exist_ok=True)

    with args.out_manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(complete_rows)

    with args.out_missing.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["case_id", "mode", "energy", "missing_path"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(missing_records)

    total_ready = len(ready_rows)
    total_complete = len(complete_rows)
    total_missing_cases = len({m["case_id"] for m in missing_records})

    print(f"Ready cases in sim manifest: {total_ready}")
    print(f"Training-ready complete cases: {total_complete}")
    print(f"Excluded cases with missing outputs: {total_missing_cases}")
    print(f"Saved training manifest: {args.out_manifest}")
    print(f"Saved missing detail: {args.out_missing}")


if __name__ == "__main__":
    main()
