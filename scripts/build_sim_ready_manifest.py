from __future__ import annotations

import argparse
import csv
from pathlib import Path


def pick_deepest_dicom_series(root: Path) -> Path | None:
    if not root.exists():
        return None
    # Prefer deepest directory containing .dcm files
    candidates = []
    for p in root.rglob("*.dcm"):
        candidates.append(p.parent)
    if not candidates:
        return None
    # unique and choose deepest path
    uniq = sorted(set(candidates), key=lambda p: len(p.parts), reverse=True)
    return uniq[0]


def infer_case_meta(case_folder: str) -> tuple[str, str]:
    if case_folder.startswith("lung_"):
        return "lung", "NSCLC"
    if case_folder.startswith("hn_"):
        return "head_neck", "HNSC/Head-Neck"
    if case_folder.startswith("colorectal_"):
        return "colorectal", "Stage II colorectal"
    if case_folder.startswith("prostate_"):
        return "prostate", "prostate"
    return "unknown", "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build simulation-ready manifest from unified CT case folders")
    parser.add_argument("--cases-root", type=Path, default=Path("data/ct_cases_by_case"))
    parser.add_argument("--out-csv", type=Path, default=Path("data/ct_cases_by_case/manifest_sim_ready.csv"))
    args = parser.parse_args()

    cases_root = args.cases_root
    if not cases_root.exists():
        raise FileNotFoundError(f"Cases root not found: {cases_root}")

    rows = []
    for case_dir in sorted(p for p in cases_root.iterdir() if p.is_dir()):
        case_id = case_dir.name
        site, cancer = infer_case_meta(case_id)

        mhd_path = None
        for cand in [case_dir / "CT_IMAGE.mhd", case_dir / "ct.mhd"]:
            if cand.exists():
                mhd_path = cand
                break

        dicom_series = None
        dicom_root = case_dir / "dicom_root"
        if dicom_root.exists():
            try:
                dicom_series = pick_deepest_dicom_series(dicom_root.resolve())
            except Exception:
                dicom_series = None

        if mhd_path is not None:
            input_type = "mhd"
            input_path = str(mhd_path)
            ready = 1
            next_action = "simulate_direct"
        elif dicom_series is not None:
            input_type = "dicom_series"
            input_path = str(dicom_series)
            ready = 1
            next_action = "convert_then_simulate"
        else:
            input_type = "missing"
            input_path = ""
            ready = 0
            next_action = "fix_case"

        rows.append(
            {
                "case_id": case_id,
                "site": site,
                "cancer": cancer,
                "input_type": input_type,
                "input_path": input_path,
                "ready_for_sim": ready,
                "next_action": next_action,
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "site",
                "cancer",
                "input_type",
                "input_path",
                "ready_for_sim",
                "next_action",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    n_total = len(rows)
    n_ready = sum(int(r["ready_for_sim"]) for r in rows)
    n_mhd = sum(1 for r in rows if r["input_type"] == "mhd")
    n_dcm = sum(1 for r in rows if r["input_type"] == "dicom_series")

    print(f"Saved: {args.out_csv}")
    print(f"Total cases: {n_total}")
    print(f"Ready: {n_ready} | MHD direct: {n_mhd} | DICOM series: {n_dcm}")


if __name__ == "__main__":
    main()
