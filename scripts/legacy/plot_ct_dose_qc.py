from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    amin = float(arr.min())
    amax = float(arr.max())
    if amax - amin < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - amin) / (amax - amin)


def save_qc_figure(ct_path: Path, dose_path: Path, output_png: Path) -> None:
    ct_img = sitk.ReadImage(str(ct_path))
    dose_img = sitk.ReadImage(str(dose_path))

    ct = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    dose = sitk.GetArrayFromImage(dose_img).astype(np.float32)

    if ct.shape != dose.shape:
        raise ValueError(f"CT y dosis deben tener misma shape. CT={ct.shape}, Dose={dose.shape}")

    z, y, x = ct.shape
    z_mid = z // 2
    y_mid = y // 2
    x_mid = x // 2

    ct_slice = ct[z_mid, :, :]
    dose_slice = dose[z_mid, :, :]

    ct_norm = normalize(ct_slice)
    dose_norm = normalize(dose_slice)

    eps = 1e-12
    dose_log = np.log10(dose_slice + eps)

    # Perfil central (single-line) para referencia.
    profile_z_center = dose[:, y_mid, x_mid]
    profile_z_center_norm = normalize(profile_z_center)

    # Perfil depth-dose paralelo al haz usando ROI cilíndrica en plano transversal.
    yy, xx = np.ogrid[:y, :x]
    r_px = max(6, int(min(x, y) * 0.04))
    roi_cyl = (yy - y_mid) ** 2 + (xx - x_mid) ** 2 <= r_px**2
    profile_z_roi = dose[:, roi_cyl].mean(axis=1)
    profile_z_roi_norm = normalize(profile_z_roi)

    # Perfil integrado en toda la sección transversal (x,y) para cada z.
    profile_z_integral = dose.sum(axis=(1, 2))
    profile_z_integral_norm = normalize(profile_z_integral)

    spacing = np.array(ct_img.GetSpacing(), dtype=np.float32)
    z_mm = np.arange(z, dtype=np.float32) * spacing[2]

    i_center = int(np.argmax(profile_z_center))
    i_roi = int(np.argmax(profile_z_roi))
    i_integral = int(np.argmax(profile_z_integral))

    d_center_mm = float(z_mm[i_center])
    d_roi_mm = float(z_mm[i_roi])
    d_integral_mm = float(z_mm[i_integral])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    ax = axes[0, 0]
    im0 = ax.imshow(ct_slice, cmap="gray")
    ax.set_title(f"CT corte axial z={z_mid}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    im1 = ax.imshow(dose_log, cmap="inferno")
    ax.set_title(f"Dosis log10 corte axial z={z_mid}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    ax.imshow(ct_norm, cmap="gray")
    overlay = ax.imshow(dose_norm, cmap="jet", alpha=0.45)
    # Contorno de ROI cilíndrica usada para depth-dose paralelo al haz.
    roi_for_plot = np.zeros((y, x), dtype=np.float32)
    roi_for_plot[roi_cyl] = 1.0
    ax.contour(roi_for_plot, levels=[0.5], colors="lime", linewidths=1.0)
    ax.set_title("Overlay CT + dosis normalizada")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    ax.plot(z_mm, profile_z_center_norm, linewidth=1.8, label="línea central")
    ax.plot(z_mm, profile_z_roi_norm, linewidth=2.2, label="ROI cilíndrica")
    ax.plot(z_mm, profile_z_integral_norm, linewidth=2.0, label="integral transversal")

    ax.axvline(d_center_mm, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(d_roi_mm, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(d_integral_mm, linestyle="--", linewidth=1.2, alpha=0.8)

    ax.scatter([d_center_mm], [profile_z_center_norm[i_center]], s=30)
    ax.scatter([d_roi_mm], [profile_z_roi_norm[i_roi]], s=30)
    ax.scatter([d_integral_mm], [profile_z_integral_norm[i_integral]], s=30)

    ax.text(d_center_mm, profile_z_center_norm[i_center] + 0.04, f"C {d_center_mm:.1f} mm", fontsize=8)
    ax.text(d_roi_mm, profile_z_roi_norm[i_roi] + 0.04, f"ROI {d_roi_mm:.1f} mm", fontsize=8)
    ax.text(d_integral_mm, profile_z_integral_norm[i_integral] + 0.04, f"INT {d_integral_mm:.1f} mm", fontsize=8)

    ax.set_title("Depth-dose paralelo al haz (z)")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Dosis normalizada")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera figura QC de CT y dosis")
    parser.add_argument("--ct", type=Path, required=True, help="Ruta CT .mhd")
    parser.add_argument("--dose", type=Path, required=True, help="Ruta dosis .mhd")
    parser.add_argument("--out", type=Path, required=True, help="Ruta PNG de salida")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_qc_figure(args.ct, args.dose, args.out)
    print(f"Figura QC guardada en: {args.out}")
