from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float64)
    amin = float(arr.min())
    amax = float(arr.max())
    if amax - amin < 1e-12:
        return np.zeros_like(arr)
    return (arr - amin) / (amax - amin)


def beamlet_spot_positions_mm(nx: int, ny: int, pitch_mm: float, center_x_mm: float, center_y_mm: float):
    x0 = center_x_mm - 0.5 * (nx - 1) * pitch_mm
    y0 = center_y_mm - 0.5 * (ny - 1) * pitch_mm
    spots = []
    for iy in range(ny):
        for ix in range(nx):
            spots.append((x0 + ix * pitch_mm, y0 + iy * pitch_mm))
    return spots


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay de trayectoria beamlet sobre CT + dosis")
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--dose", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--beamlet-nx", type=int, default=5)
    parser.add_argument("--beamlet-ny", type=int, default=5)
    parser.add_argument("--beamlet-pitch-mm", type=float, default=6.0)
    parser.add_argument("--center-x-mm", type=float, default=0.0)
    parser.add_argument("--center-y-mm", type=float, default=0.0)
    parser.add_argument("--source-z-mm", type=float, default=-300.0)
    args = parser.parse_args()

    ct_img = sitk.ReadImage(str(args.ct))
    dose_img = sitk.ReadImage(str(args.dose))

    ct = sitk.GetArrayFromImage(ct_img).astype(np.float64)     # z,y,x
    dose = sitk.GetArrayFromImage(dose_img).astype(np.float64) # z,y,x

    if ct.shape != dose.shape:
        raise ValueError(f"Shapes distintas CT={ct.shape} dose={dose.shape}")

    zdim, ydim, xdim = ct.shape
    spacing = np.array(ct_img.GetSpacing(), dtype=np.float64)  # x,y,z
    origin = np.array(ct_img.GetOrigin(), dtype=np.float64)    # x,y,z

    # z de mayor depósito integral para vista transversal representativa.
    depth_prof = dose.sum(axis=(1, 2))
    z_peak_idx = int(np.argmax(depth_prof))
    z_peak_mm = origin[2] + z_peak_idx * spacing[2]

    ct_ax = ct[z_peak_idx, :, :]
    dose_ax = dose[z_peak_idx, :, :]

    ct_ax_n = normalize(ct_ax)
    dose_ax_n = normalize(np.log10(dose_ax + 1e-12))

    # Vistas paralelas al haz
    x_center_idx = int(round((args.center_x_mm - origin[0]) / spacing[0]))
    y_center_idx = int(round((args.center_y_mm - origin[1]) / spacing[1]))
    x_center_idx = int(np.clip(x_center_idx, 0, xdim - 1))
    y_center_idx = int(np.clip(y_center_idx, 0, ydim - 1))

    # Usamos una losa de +/- n_slab para que se vea mejor la deposición en paralelo.
    n_slab = 2
    y0 = max(0, y_center_idx - n_slab)
    y1 = min(ydim, y_center_idx + n_slab + 1)
    x0 = max(0, x_center_idx - n_slab)
    x1 = min(xdim, x_center_idx + n_slab + 1)

    ct_xz = ct[:, y0:y1, :].mean(axis=1)
    dose_xz = dose[:, y0:y1, :].sum(axis=1)

    ct_yz = ct[:, :, x0:x1].mean(axis=2)
    dose_yz = dose[:, :, x0:x1].sum(axis=2)

    ct_xz_n = normalize(ct_xz)
    dose_xz_n = normalize(np.log10(dose_xz + 1e-12))
    ct_yz_n = normalize(ct_yz)
    dose_yz_n = normalize(np.log10(dose_yz + 1e-12))

    spots = beamlet_spot_positions_mm(
        nx=args.beamlet_nx,
        ny=args.beamlet_ny,
        pitch_mm=args.beamlet_pitch_mm,
        center_x_mm=args.center_x_mm,
        center_y_mm=args.center_y_mm,
    )

    # Conversión spots -> índices en corte axial al z de pico
    spot_xy_idx = []
    for sx_mm, sy_mm in spots:
        ix = (sx_mm - origin[0]) / spacing[0]
        iy = (sy_mm - origin[1]) / spacing[1]
        spot_xy_idx.append((ix, iy))

    x_positions_mm = sorted(set([s[0] for s in spots]))
    y_positions_mm = sorted(set([s[1] for s in spots]))
    x_positions_idx = [((x_mm - origin[0]) / spacing[0]) for x_mm in x_positions_mm]
    y_positions_idx = [((y_mm - origin[1]) / spacing[1]) for y_mm in y_positions_mm]

    # z en mm para anotación fuente
    z_min_mm = origin[2]
    z_max_mm = origin[2] + (zdim - 1) * spacing[2]
    x_min_mm = origin[0]
    x_max_mm = origin[0] + (xdim - 1) * spacing[0]
    y_min_mm = origin[1]
    y_max_mm = origin[1] + (ydim - 1) * spacing[1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # Transversal (axial)
    ax = axes[0]
    extent_ax = [x_min_mm, x_max_mm, y_max_mm, y_min_mm]
    ax.imshow(ct_ax_n, cmap="gray", extent=extent_ax)
    ov = ax.imshow(dose_ax_n, cmap="jet", alpha=0.45, extent=extent_ax)
    for ix, iy in spot_xy_idx:
        x_mm = origin[0] + ix * spacing[0]
        y_mm = origin[1] + iy * spacing[1]
        ax.scatter(x_mm, y_mm, s=18, c="white", marker="x", linewidths=0.8)
    ax.set_title(f"Transversal (axial) z={z_peak_idx} (~{z_peak_mm:.1f} mm)\nSpots beamlet proyectados")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.colorbar(ov, ax=ax, fraction=0.046, pad=0.04)

    # Paralela al haz: plano XZ (y fijo)
    ax = axes[1]
    extent_xz = [x_min_mm, x_max_mm, z_max_mm, z_min_mm]
    ax.imshow(ct_xz_n, cmap="gray", aspect="auto", extent=extent_xz)
    ov2 = ax.imshow(dose_xz_n, cmap="jet", alpha=0.45, aspect="auto", extent=extent_xz)
    for ix in x_positions_idx:
        x_mm = origin[0] + ix * spacing[0]
        ax.axvline(x_mm, color="white", linestyle="--", linewidth=0.8, alpha=0.8)

    ax.axhline(z_peak_mm, color="cyan", linestyle="-.", linewidth=1.2, alpha=0.9)
    ax.text(x_min_mm + 5, z_peak_mm + 5, f"pico ~ {z_peak_mm:.1f} mm", color="cyan", fontsize=8)

    # marca fuente: se muestra aunque esté fuera del rango del CT
    src_z_mm = args.source_z_mm
    for sx_mm, _sy_mm in spots:
        ax.plot([sx_mm, sx_mm], [src_z_mm, z_max_mm], color="lime", alpha=0.15, linewidth=0.7)
    ax.scatter([args.center_x_mm], [src_z_mm], c="lime", s=32, marker="o")
    ax.text(args.center_x_mm + 5, src_z_mm, "fuente", color="lime", fontsize=8)

    ax.set_ylim(max(z_max_mm, src_z_mm + 20), min(z_min_mm, src_z_mm - 20))
    ax.set_title("Paralela al haz (plano XZ, y fijo)\nLíneas verdes: rayos desde fuente")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    fig.colorbar(ov2, ax=ax, fraction=0.046, pad=0.04)

    # Paralela al haz: plano YZ (x fijo)
    ax = axes[2]
    extent_yz = [y_min_mm, y_max_mm, z_max_mm, z_min_mm]
    ax.imshow(ct_yz_n, cmap="gray", aspect="auto", extent=extent_yz)
    ov3 = ax.imshow(dose_yz_n, cmap="jet", alpha=0.45, aspect="auto", extent=extent_yz)
    for iy in y_positions_idx:
        y_mm = origin[1] + iy * spacing[1]
        ax.axvline(y_mm, color="white", linestyle="--", linewidth=0.8, alpha=0.8)

    ax.axhline(z_peak_mm, color="cyan", linestyle="-.", linewidth=1.2, alpha=0.9)
    for _sx_mm, sy_mm in spots:
        ax.plot([sy_mm, sy_mm], [src_z_mm, z_max_mm], color="lime", alpha=0.15, linewidth=0.7)
    ax.scatter([args.center_y_mm], [src_z_mm], c="lime", s=32, marker="o")
    ax.text(args.center_y_mm + 5, src_z_mm, "fuente", color="lime", fontsize=8)

    ax.set_ylim(max(z_max_mm, src_z_mm + 20), min(z_min_mm, src_z_mm - 20))
    ax.set_title("Paralela al haz (plano YZ, x fijo)\nLíneas verdes: rayos desde fuente")
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("z (mm)")
    fig.colorbar(ov3, ax=ax, fraction=0.046, pad=0.04)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    plt.close(fig)

    print(f"Figura guardada en: {args.out}")
    print(f"z_peak_idx={z_peak_idx}, z_peak_mm={z_peak_mm:.2f}")
    print(f"beamlet grid: {args.beamlet_nx}x{args.beamlet_ny}, pitch={args.beamlet_pitch_mm} mm")
    print(f"z range image: [{z_min_mm:.2f}, {z_max_mm:.2f}] mm, source_z={args.source_z_mm:.2f} mm")


if __name__ == "__main__":
    main()
