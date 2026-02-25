from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = np.max(x)
    if m <= 0:
        return np.zeros_like(x)
    return x / m


def main() -> None:
    parser = argparse.ArgumentParser(description="Verificar eje de depth-dose vs haz")
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--dose", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--beam-axis", type=str, default="z", choices=["x", "y", "z"])
    parser.add_argument("--title", type=str, default="")
    args = parser.parse_args()

    ct_img = sitk.ReadImage(str(args.ct))
    dose_img = sitk.ReadImage(str(args.dose))

    ct = sitk.GetArrayFromImage(ct_img).astype(np.float64)      # z,y,x
    dose = sitk.GetArrayFromImage(dose_img).astype(np.float64)  # z,y,x

    if ct.shape != dose.shape:
        raise ValueError(f"Shape mismatch: ct={ct.shape}, dose={dose.shape}")

    spacing = np.array(ct_img.GetSpacing(), dtype=np.float64)  # x,y,z
    origin = np.array(ct_img.GetOrigin(), dtype=np.float64)    # x,y,z

    zdim, ydim, xdim = dose.shape
    y_mid, x_mid = ydim // 2, xdim // 2

    # Profiles along each axis (integrated over transverse planes)
    p_z = dose.sum(axis=(1, 2))
    p_y = dose.sum(axis=(0, 2))
    p_x = dose.sum(axis=(0, 1))

    z_mm = origin[2] + np.arange(zdim) * spacing[2]
    y_mm = origin[1] + np.arange(ydim) * spacing[1]
    x_mm = origin[0] + np.arange(xdim) * spacing[0]

    p_z_n = normalize(p_z)
    p_y_n = normalize(p_y)
    p_x_n = normalize(p_x)

    # Imagen paralela al haz (si haz es z, usamos XZ en y fijo)
    ct_xz = ct[:, y_mid, :]
    dose_xz = np.log10(dose[:, y_mid, :] + 1e-12)

    x_min_mm = origin[0]
    x_max_mm = origin[0] + (xdim - 1) * spacing[0]
    z_min_mm = origin[2]
    z_max_mm = origin[2] + (zdim - 1) * spacing[2]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # Left: image + dose in parallel plane
    ax = axes[0]
    ax.imshow(ct_xz, cmap="gray", aspect="auto", extent=[x_min_mm, x_max_mm, z_max_mm, z_min_mm])
    ov = ax.imshow(dose_xz, cmap="jet", alpha=0.45, aspect="auto", extent=[x_min_mm, x_max_mm, z_max_mm, z_min_mm])
    ax.set_title("Plano paralelo al haz (XZ, y fijo)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.annotate(
        "dirección haz (+z)",
        xy=(x_max_mm - 30, z_min_mm + 80),
        xytext=(x_max_mm - 120, z_min_mm + 20),
        arrowprops=dict(arrowstyle="->", color="lime", lw=2),
        color="lime",
        fontsize=9,
    )
    fig.colorbar(ov, ax=ax, fraction=0.046, pad=0.04)

    # Right: depth-dose candidates by axis
    ax = axes[1]
    ax.plot(z_mm, p_z_n, label="perfil eje z (sum yx)", linewidth=2.2)
    ax.plot(y_mm, p_y_n, label="perfil eje y (sum zx)", linewidth=1.5)
    ax.plot(x_mm, p_x_n, label="perfil eje x (sum zy)", linewidth=1.5)

    if args.beam_axis == "z":
        ax.text(0.02, 0.95, "Depth-dose esperado: eje z", transform=ax.transAxes, fontsize=10, color="tab:green")

    ax.set_title("Comparación de perfiles por eje")
    ax.set_xlabel("Coordenada física (mm)")
    ax.set_ylabel("Dosis integrada normalizada")
    ax.grid(alpha=0.3)
    ax.legend()

    if args.title:
        fig.suptitle(args.title)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    plt.close(fig)

    print(f"Figura guardada en: {args.out}")
    print(f"Peak z @ {z_mm[int(np.argmax(p_z))]:.2f} mm")
    print(f"Peak y @ {y_mm[int(np.argmax(p_y))]:.2f} mm")
    print(f"Peak x @ {x_mm[int(np.argmax(p_x))]:.2f} mm")


if __name__ == "__main__":
    main()
