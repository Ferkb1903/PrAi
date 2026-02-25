from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk


DEFAULT_INCLUDE = r"(ct|image|img)"
DEFAULT_EXCLUDE = r"(mask|label|seg|contour|oar|gtv|ptv|rtstruct)"


def infer_split(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    if "test" in parts:
        return "test"
    if "validation" in parts or "val" in parts:
        return "validation"
    if "train" in parts:
        return "train"
    return "unknown"


def infer_case_id(rel_path: Path) -> str:
    # Uses parent folder + filename stem for stable IDs
    parent = rel_path.parent.name.replace(" ", "_")
    stem = rel_path.stem.replace(" ", "_")
    return f"{parent}__{stem}"


def volume_stats(img: sitk.Image) -> dict:
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    spacing = img.GetSpacing()  # x,y,z
    size = img.GetSize()  # x,y,z
    return {
        "size_xyz": f"{size[0]}x{size[1]}x{size[2]}",
        "spacing_xyz_mm": f"{spacing[0]:.4f},{spacing[1]:.4f},{spacing[2]:.4f}",
        "vmin": float(np.min(arr)),
        "vmax": float(np.max(arr)),
        "vmean": float(np.mean(arr)),
    }


def looks_like_ct_hu(vmin: float, vmax: float) -> bool:
    return (vmin <= -700.0) and (vmax >= 800.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest TCIA NRRD CT dataset into MHD format")
    parser.add_argument("--input-root", type=Path, required=True, help="Root with NRRD files")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root for converted MHD files")
    parser.add_argument("--manifest-csv", type=Path, required=True, help="Output CSV manifest path")
    parser.add_argument("--include-regex", type=str, default=DEFAULT_INCLUDE, help="Regex to include file paths")
    parser.add_argument("--exclude-regex", type=str, default=DEFAULT_EXCLUDE, help="Regex to exclude file paths")
    parser.add_argument("--dry-run", action="store_true", help="Scan and validate without writing MHD")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")

    include_re = re.compile(args.include_regex, flags=re.IGNORECASE)
    exclude_re = re.compile(args.exclude_regex, flags=re.IGNORECASE)

    nrrd_files = sorted(args.input_root.rglob("*.nrrd"))
    if not nrrd_files:
        raise RuntimeError(f"No NRRD files found in {args.input_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    converted = 0
    skipped = 0

    for src in nrrd_files:
        rel = src.relative_to(args.input_root)
        rel_text = str(rel).lower()

        status = ""
        reason = ""

        if not include_re.search(rel_text):
            skipped += 1
            status = "skipped"
            reason = "include_regex_no_match"
            rows.append(
                {
                    "case_id": infer_case_id(rel),
                    "split": infer_split(rel),
                    "source_nrrd": str(src),
                    "output_mhd": "",
                    "size_xyz": "",
                    "spacing_xyz_mm": "",
                    "vmin": "",
                    "vmax": "",
                    "vmean": "",
                    "hu_like": "",
                    "status": status,
                    "reason": reason,
                }
            )
            continue

        if exclude_re.search(rel_text):
            skipped += 1
            status = "skipped"
            reason = "exclude_regex_match"
            rows.append(
                {
                    "case_id": infer_case_id(rel),
                    "split": infer_split(rel),
                    "source_nrrd": str(src),
                    "output_mhd": "",
                    "size_xyz": "",
                    "spacing_xyz_mm": "",
                    "vmin": "",
                    "vmax": "",
                    "vmean": "",
                    "hu_like": "",
                    "status": status,
                    "reason": reason,
                }
            )
            continue

        try:
            img = sitk.ReadImage(str(src))
            if img.GetDimension() != 3:
                skipped += 1
                status = "skipped"
                reason = f"not_3d_dim={img.GetDimension()}"
                rows.append(
                    {
                        "case_id": infer_case_id(rel),
                        "split": infer_split(rel),
                        "source_nrrd": str(src),
                        "output_mhd": "",
                        "size_xyz": "",
                        "spacing_xyz_mm": "",
                        "vmin": "",
                        "vmax": "",
                        "vmean": "",
                        "hu_like": "",
                        "status": status,
                        "reason": reason,
                    }
                )
                continue

            stats = volume_stats(img)
            hu_like = looks_like_ct_hu(stats["vmin"], stats["vmax"])

            out_mhd = args.output_root / rel.with_suffix(".mhd")
            out_mhd.parent.mkdir(parents=True, exist_ok=True)

            if not args.dry_run:
                sitk.WriteImage(img, str(out_mhd), useCompression=False)

            converted += 1
            rows.append(
                {
                    "case_id": infer_case_id(rel),
                    "split": infer_split(rel),
                    "source_nrrd": str(src),
                    "output_mhd": str(out_mhd),
                    "size_xyz": stats["size_xyz"],
                    "spacing_xyz_mm": stats["spacing_xyz_mm"],
                    "vmin": f"{stats['vmin']:.3f}",
                    "vmax": f"{stats['vmax']:.3f}",
                    "vmean": f"{stats['vmean']:.3f}",
                    "hu_like": int(hu_like),
                    "status": "converted" if not args.dry_run else "dry_run",
                    "reason": "",
                }
            )

        except Exception as exc:  # noqa: BLE001
            skipped += 1
            rows.append(
                {
                    "case_id": infer_case_id(rel),
                    "split": infer_split(rel),
                    "source_nrrd": str(src),
                    "output_mhd": "",
                    "size_xyz": "",
                    "spacing_xyz_mm": "",
                    "vmin": "",
                    "vmax": "",
                    "vmean": "",
                    "hu_like": "",
                    "status": "error",
                    "reason": str(exc).replace("\n", " ")[:300],
                }
            )

    with open(args.manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "split",
                "source_nrrd",
                "output_mhd",
                "size_xyz",
                "spacing_xyz_mm",
                "vmin",
                "vmax",
                "vmean",
                "hu_like",
                "status",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"NRRD found: {len(nrrd_files)}")
    print(f"Converted: {converted}")
    print(f"Skipped/Error: {skipped}")
    print(f"Manifest: {args.manifest_csv}")


if __name__ == "__main__":
    main()
