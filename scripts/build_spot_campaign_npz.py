from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io_npz import save_case_npz
from src.data.schema import CaseData


@dataclass
class PairEntry:
    case_id: str
    energy_mev: float
    energy_token: str
    spot_token: str
    low_mhd: Path
    high_mhd: Path


def _parse_energy_mev(energy_token: str) -> float:
    if not energy_token.startswith("E"):
        raise ValueError(f"Energy folder inválida: {energy_token}")
    return float(energy_token[1:])


def _resolve_high_dir(spot_dir: Path) -> Path | None:
    candidates = [spot_dir / "high", Path(str(spot_dir / "high") + "\r")]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _resolve_dose_mhd(folder: Path, dose_stem: str) -> Path | None:
    mhd = folder / f"{dose_stem}.mhd"
    if mhd.exists():
        return mhd
    return None


def discover_pairs(root: Path, dose_stem: str) -> list[PairEntry]:
    pair_entries: list[PairEntry] = []
    for spot_dir in sorted(root.glob("*/E*/spot_*")):
        if not spot_dir.is_dir():
            continue
        case_id = spot_dir.parent.parent.name
        energy_token = spot_dir.parent.name
        spot_token = spot_dir.name

        low_dir = spot_dir / "low"
        high_dir = _resolve_high_dir(spot_dir)
        if not low_dir.exists() or high_dir is None:
            continue

        low_mhd = _resolve_dose_mhd(low_dir, dose_stem)
        high_mhd = _resolve_dose_mhd(high_dir, dose_stem)
        if low_mhd is None or high_mhd is None:
            continue

        pair_entries.append(
            PairEntry(
                case_id=case_id,
                energy_mev=_parse_energy_mev(energy_token),
                energy_token=energy_token,
                spot_token=spot_token,
                low_mhd=low_mhd,
                high_mhd=high_mhd,
            )
        )
    return pair_entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Convierte outputs spot_campaign a dataset NPZ para entrenamiento")
    parser.add_argument("--spot-root", type=Path, default=Path("outputs/spot_campaign"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/training_npz/spot_campaign"))
    parser.add_argument("--dose-stem", type=str, default="dose_voxelized_ct_edep")
    parser.add_argument("--beam-axis", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--limit", type=int, default=0, help="0 = procesar todo")
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("data/training_npz/spot_campaign_manifest.csv"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(args.spot_root, args.dose_stem)
    if args.limit > 0:
        pairs = pairs[: args.limit]

    if not pairs:
        raise RuntimeError(f"No se encontraron pares válidos en {args.spot_root}")

    kept_rows: list[dict[str, str]] = []
    skipped = 0

    for i, pair in enumerate(pairs, start=1):
        low_img = sitk.ReadImage(str(pair.low_mhd))
        high_img = sitk.ReadImage(str(pair.high_mhd))

        low_arr = sitk.GetArrayFromImage(low_img).astype(np.float32)
        high_arr = sitk.GetArrayFromImage(high_img).astype(np.float32)

        if low_arr.shape != high_arr.shape:
            skipped += 1
            continue

        if not np.isfinite(low_arr).all() or not np.isfinite(high_arr).all():
            skipped += 1
            continue

        spr = np.ones_like(low_arr, dtype=np.float32)
        spacing_mm = np.asarray(low_img.GetSpacing()[::-1], dtype=np.float32)
        case = CaseData(
            d_low=low_arr,
            spr=spr,
            d_high=high_arr,
            e0_mev=pair.energy_mev,
            spacing_mm=spacing_mm,
            beam_axis=args.beam_axis,
            case_id=pair.case_id,
        )

        file_name = f"{pair.case_id}_{pair.energy_token}_{pair.spot_token}.npz"
        out_path = args.out_dir / file_name
        save_case_npz(out_path, case)

        kept_rows.append(
            {
                "npz_path": str(out_path),
                "case_id": pair.case_id,
                "energy_token": pair.energy_token,
                "spot_token": pair.spot_token,
            }
        )

        if i % 200 == 0:
            print(f"Procesados {i}/{len(pairs)} pares...")

    with args.manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["npz_path", "case_id", "energy_token", "spot_token"])
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"Pares detectados: {len(pairs)}")
    print(f"NPZ generados: {len(kept_rows)}")
    print(f"Pares omitidos: {skipped}")
    print(f"Dataset NPZ: {args.out_dir}")
    print(f"Manifest CSV: {args.manifest_csv}")


if __name__ == "__main__":
    main()
