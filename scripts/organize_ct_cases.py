from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src)


def find_lung_cases(repo_root: Path) -> list[tuple[str, Path]]:
    base = repo_root / "CT_Lung" / "NSCLC-Radiomics-Genomics"
    out: list[tuple[str, Path]] = []
    if not base.exists():
        return out
    for case_dir in sorted(base.glob("LUNG3-*")):
        if case_dir.is_dir():
            out.append((case_dir.name, case_dir))
    return out


def find_colorectal_cases(repo_root: Path) -> list[tuple[str, Path]]:
    base = repo_root / "CT_colorectal" / "StageII-Colorectal-CT"
    out: list[tuple[str, Path]] = []
    if not base.exists():
        return out
    for case_dir in sorted(base.glob("StageII-Colorectal-CT-*")):
        if case_dir.is_dir():
            out.append((case_dir.name, case_dir))
    return out


def find_hn_nrrd_cases(repo_root: Path) -> dict[str, Path]:
    root = repo_root / "data" / "raw" / "tcia_hn_subset" / "nrrds"
    result: dict[str, Path] = {}
    if not root.exists():
        return result
    for ct in root.rglob("CT_IMAGE.nrrd"):
        case = ct.parent.name
        if case not in result:
            result[case] = ct
    return result


def find_hn_mhd_cases(repo_root: Path) -> dict[str, Path]:
    root = repo_root / "data" / "raw" / "tcia_hn_subset" / "mhd"
    result: dict[str, Path] = {}
    if not root.exists():
        return result
    for mhd in root.rglob("CT_IMAGE.mhd"):
        case = mhd.parent.name
        if case not in result:
            result[case] = mhd
    return result


def find_prostate_case(repo_root: Path) -> Path | None:
    p = repo_root / "data" / "raw" / "ct_prostate_0002"
    return p if p.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize all CT sources into one folder by case")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-root", type=Path, default=Path("data/ct_cases_by_case"))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_root = (repo_root / args.out_root).resolve() if not args.out_root.is_absolute() else args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []

    # Lung NSCLC
    for case_id, case_path in find_lung_cases(repo_root):
        folder = out_root / f"lung_{case_id}"
        folder.mkdir(parents=True, exist_ok=True)
        safe_symlink(case_path, folder / "dicom_root")
        rows.append(
            {
                "case_folder": folder.name,
                "site": "lung",
                "cancer": "NSCLC",
                "source_type": "DICOM series folders",
                "path": str(case_path),
            }
        )

    # Colorectal Stage II
    for case_id, case_path in find_colorectal_cases(repo_root):
        folder = out_root / f"colorectal_{case_id}"
        folder.mkdir(parents=True, exist_ok=True)
        safe_symlink(case_path, folder / "dicom_root")

        meta_csv = repo_root / "CT_colorectal" / "metadata.csv"
        if meta_csv.exists():
            safe_symlink(meta_csv, folder / "metadata.csv")

        rows.append(
            {
                "case_folder": folder.name,
                "site": "colorectal",
                "cancer": "stage II colorectal",
                "source_type": "DICOM series folders",
                "path": str(case_path),
            }
        )

    # HN TCIA subset
    hn_nrrd = find_hn_nrrd_cases(repo_root)
    hn_mhd = find_hn_mhd_cases(repo_root)
    for case_id in sorted(set(hn_nrrd.keys()) | set(hn_mhd.keys())):
        folder = out_root / f"hn_{case_id}"
        folder.mkdir(parents=True, exist_ok=True)
        if case_id in hn_nrrd:
            safe_symlink(hn_nrrd[case_id], folder / "CT_IMAGE.nrrd")
        if case_id in hn_mhd:
            safe_symlink(hn_mhd[case_id], folder / "CT_IMAGE.mhd")
            raw = hn_mhd[case_id].with_suffix(".raw")
            if raw.exists():
                safe_symlink(raw, folder / "CT_IMAGE.raw")

        rows.append(
            {
                "case_folder": folder.name,
                "site": "head_neck",
                "cancer": "HNSC / Head-Neck Cetuximab cohort",
                "source_type": "NRRD (+ MHD converted when available)",
                "path": str(hn_nrrd.get(case_id, hn_mhd.get(case_id))),
            }
        )

    # Prostate
    prostate = find_prostate_case(repo_root)
    if prostate is not None:
        folder = out_root / "prostate_ct_prostate_0002"
        folder.mkdir(parents=True, exist_ok=True)
        for name in ["ct.mhd", "ct.raw"]:
            p = prostate / name
            if p.exists():
                safe_symlink(p, folder / name)
        rows.append(
            {
                "case_folder": folder.name,
                "site": "prostate",
                "cancer": "prostate (local case)",
                "source_type": "MHD/RAW",
                "path": str(prostate),
            }
        )

    manifest = out_root / "manifest_ct_cases.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_folder", "site", "cancer", "source_type", "path"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["case_folder"]))

    print(f"Created root: {out_root}")
    print(f"Cases organized: {len(rows)}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
