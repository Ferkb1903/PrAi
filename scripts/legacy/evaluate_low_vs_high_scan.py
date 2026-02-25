from pathlib import Path
import argparse
import csv
import re
import sys

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.gamma import gamma_report
from src.metrics.distal_range import distal_error_mm


def parse_energy_from_dir(name: str) -> float:
    m = re.match(r"E([0-9]+(?:\.[0-9]+)?)", name)
    if not m:
        raise ValueError(f"Nombre inválido de carpeta energía: {name}")
    return float(m.group(1))


def load_dose(path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img).astype(np.float32)  # z,y,x


def evaluate_pair(low_path: Path, high_path: Path, spacing_xyz: np.ndarray) -> dict:
    low = load_dose(low_path)
    high = load_dose(high_path)

    if low.shape != high.shape:
        raise ValueError(f"Shape mismatch low={low.shape} high={high.shape}")

    mae = float(np.mean(np.abs(high - low)))
    rmse = float(np.sqrt(np.mean((high - low) ** 2)))

    g = gamma_report(high, low, criteria=((3.0, 3.0), (2.0, 2.0)))

    # arrays son z,y,x -> beam axis de simulación actual es +z => axis=0 en z,y,x
    d_err = float(distal_error_mm(high, low, spacing_mm=np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], dtype=np.float32), beam_axis=0))

    return {
        "mae": mae,
        "rmse": rmse,
        "gamma_3pct": float(g["gamma_3pct"]),
        "gamma_2pct": float(g["gamma_2pct"]),
        "distal_error_mm": d_err,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa low vs high por energía")
    parser.add_argument("--low-root", type=Path, required=True, help="Carpeta raíz low (ej. 200k)")
    parser.add_argument("--high-root", type=Path, required=True, help="Carpeta raíz high (ej. 500k)")
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for d in sorted(args.high_root.iterdir()):
        if not d.is_dir() or not d.name.startswith("E"):
            continue

        energy = parse_energy_from_dir(d.name)
        high_dose = d / "dose_voxelized_ct_edep.mhd"
        low_dose = args.low_root / d.name / "dose_voxelized_ct_edep.mhd"

        if not high_dose.exists() or not low_dose.exists():
            continue

        high_img = sitk.ReadImage(str(high_dose))
        spacing_xyz = np.array(high_img.GetSpacing(), dtype=np.float32)

        metrics = evaluate_pair(low_dose, high_dose, spacing_xyz=spacing_xyz)
        row = {"energy_mev": energy, **metrics}
        rows.append(row)

    if not rows:
        raise RuntimeError("No se encontraron pares low/high comparables")

    rows.sort(key=lambda r: r["energy_mev"])

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["energy_mev", "mae", "rmse", "gamma_3pct", "gamma_2pct", "distal_error_mm"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV guardado en: {args.out_csv}")
    print("\nResumen por energía (low vs high):")
    for r in rows:
        print(
            f"E={r['energy_mev']:.0f} MeV | "
            f"MAE={r['mae']:.3f} RMSE={r['rmse']:.3f} | "
            f"G3={r['gamma_3pct']:.2f}% G2={r['gamma_2pct']:.2f}% | "
            f"dR={r['distal_error_mm']:.2f} mm"
        )

    # promedio global
    mean = {k: float(np.mean([r[k] for r in rows])) for k in ["mae", "rmse", "gamma_3pct", "gamma_2pct", "distal_error_mm"]}
    print("\nPromedio global:")
    print(
        f"MAE={mean['mae']:.3f}, RMSE={mean['rmse']:.3f}, "
        f"G3={mean['gamma_3pct']:.2f}%, G2={mean['gamma_2pct']:.2f}%, dR={mean['distal_error_mm']:.2f} mm"
    )


if __name__ == "__main__":
    main()
