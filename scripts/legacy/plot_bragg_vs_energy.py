from pathlib import Path
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def parse_energy_from_folder(name: str) -> float:
    m = re.match(r"E([0-9]+(?:\.[0-9]+)?)", name)
    if not m:
        raise ValueError(f"No puedo extraer energía de carpeta: {name}")
    return float(m.group(1))


def depth_profile_integral(dose_mhd: Path):
    img = sitk.ReadImage(str(dose_mhd))
    arr = sitk.GetArrayFromImage(img).astype(np.float64)  # z,y,x
    spacing = img.GetSpacing()  # x,y,z
    z_mm = np.arange(arr.shape[0], dtype=np.float64) * float(spacing[2])
    prof = arr.sum(axis=(1, 2))
    return z_mm, prof


def distal_r80_mm(z_mm: np.ndarray, prof: np.ndarray) -> float:
    pmax = float(np.max(prof))
    if pmax <= 0:
        return float("nan")
    thr = 0.8 * pmax
    peak_idx = int(np.argmax(prof))
    tail = prof[peak_idx:]
    idx_rel = np.where(tail <= thr)[0]
    if idx_rel.size == 0:
        return float(z_mm[-1])
    return float(z_mm[peak_idx + int(idx_rel[0])])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Bragg peak shift vs energy")
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    entries = []
    for d in sorted(args.scan_root.iterdir()):
        if not d.is_dir():
            continue
        try:
            energy = parse_energy_from_folder(d.name)
        except ValueError:
            continue
        dose = d / "dose_voxelized_ct_edep.mhd"
        if dose.exists():
            entries.append((energy, dose))

    if not entries:
        raise RuntimeError(f"No encontré dosis en {args.scan_root}")

    entries.sort(key=lambda x: x[0])

    energies = []
    z_peak = []
    z_r80 = []

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    for energy, dose_path in entries:
        z_mm, prof = depth_profile_integral(dose_path)
        prof_n = prof / (np.max(prof) + 1e-12)

        ip = int(np.argmax(prof))
        peak_mm = float(z_mm[ip])
        r80_mm = distal_r80_mm(z_mm, prof)

        energies.append(energy)
        z_peak.append(peak_mm)
        z_r80.append(r80_mm)

        axes[0].plot(z_mm, prof_n, linewidth=1.8, label=f"{energy:.0f} MeV")

    axes[0].set_title("Depth-dose normalizado por energía")
    axes[0].set_xlabel("z (mm)")
    axes[0].set_ylabel("Dosis integrada normalizada")
    axes[0].grid(alpha=0.3)
    axes[0].legend(ncol=2)

    axes[1].plot(energies, z_peak, marker="o", linewidth=2, label="z pico")
    axes[1].plot(energies, z_r80, marker="s", linewidth=2, label="z R80 distal")
    axes[1].set_title("Desplazamiento del pico de Bragg vs energía")
    axes[1].set_xlabel("Energía (MeV)")
    axes[1].set_ylabel("Profundidad (mm, en grilla dosis)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    plt.close(fig)

    print(f"Figura guardada en: {args.out}")
    for e, p, r in zip(energies, z_peak, z_r80):
        print(f"E={e:.0f} MeV -> z_peak={p:.2f} mm, z_R80={r:.2f} mm")


if __name__ == "__main__":
    main()
