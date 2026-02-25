from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def main() -> None:
    parser = argparse.ArgumentParser(description="Residual 10k vs 1M")
    parser.add_argument("--dose-10k", type=Path, required=True)
    parser.add_argument("--dose-1m", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=100.0, help="Factor para escalar 10k hacia 1M")
    args = parser.parse_args()

    d10_img = sitk.ReadImage(str(args.dose_10k))
    d1m_img = sitk.ReadImage(str(args.dose_1m))

    d10 = sitk.GetArrayFromImage(d10_img).astype(np.float64)
    d1m = sitk.GetArrayFromImage(d1m_img).astype(np.float64)

    if d10.shape != d1m.shape:
        raise ValueError(f"Shapes distintas: {d10.shape} vs {d1m.shape}")

    d10s = d10 * float(args.scale)
    eps = 1e-12

    # Residuo relativo voxel-wise respecto a 1M
    rel = (d10s - d1m) / (d1m + eps)

    # Perfil depth-dose integrado
    p10 = d10s.sum(axis=(1, 2))
    p1m = d1m.sum(axis=(1, 2))
    rel_prof = (p10 - p1m) / (p1m + eps)

    spacing = d1m_img.GetSpacing()  # x,y,z
    z_mm = np.arange(d1m.shape[0], dtype=np.float64) * float(spacing[2])

    # Máscara en región relevante de profundidad para evitar explosión en cola casi cero
    m = p1m > 0.05 * np.max(p1m)
    rel_prof_valid = rel_prof[m]

    z_mid = d1m.shape[0] // 2
    rel_slice = rel[z_mid]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    axes[0, 0].plot(z_mm, rel_prof * 100.0, linewidth=2)
    axes[0, 0].axhline(0.0, color="k", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Residuo relativo depth-dose (%)")
    axes[0, 0].set_xlabel("z (mm)")
    axes[0, 0].set_ylabel("(10k*100 - 1M) / 1M [%]")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].hist((rel_prof_valid * 100.0), bins=50, color="tab:blue", alpha=0.8)
    axes[0, 1].set_title("Hist residuo depth-dose (%)\n(z con p1m > 5% pico)")
    axes[0, 1].set_xlabel("residuo %")
    axes[0, 1].set_ylabel("bins")
    axes[0, 1].grid(alpha=0.3)

    vmax = np.percentile(np.abs(rel_slice[np.isfinite(rel_slice)]), 99)
    im = axes[1, 0].imshow(rel_slice * 100.0, cmap="bwr", vmin=-vmax * 100.0, vmax=vmax * 100.0)
    axes[1, 0].set_title(f"Residuo relativo axial z={z_mid} (%)")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Mapa de magnitud absoluta de diferencia relativa
    im2 = axes[1, 1].imshow(np.abs(rel_slice) * 100.0, cmap="magma")
    axes[1, 1].set_title(f"|Residuo relativo| axial z={z_mid} (%)")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("y")
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)

    print(f"Figura guardada en: {args.out}")
    if rel_prof_valid.size > 0:
        print("Depth-dose residual % (valid region):")
        print(f"  mean={np.mean(rel_prof_valid)*100:.3f}")
        print(f"  median={np.median(rel_prof_valid)*100:.3f}")
        print(f"  p95={np.percentile(np.abs(rel_prof_valid),95)*100:.3f}")


if __name__ == "__main__":
    main()
