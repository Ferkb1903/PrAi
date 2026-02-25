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


def load_dose(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float64)  # z,y,x
    spacing = img.GetSpacing()  # x,y,z
    return arr, spacing


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara dos corridas de dosis")
    parser.add_argument("--dose-a", type=Path, required=True)
    parser.add_argument("--dose-b", type=Path, required=True)
    parser.add_argument("--label-a", type=str, default="run A")
    parser.add_argument("--label-b", type=str, default="run B")
    parser.add_argument("--events-a", type=float, default=1.0)
    parser.add_argument("--events-b", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    a, sp_a = load_dose(args.dose_a)
    b, sp_b = load_dose(args.dose_b)

    if a.shape != b.shape:
        raise ValueError(f"Shapes distintas: {a.shape} vs {b.shape}")

    z, y, x = a.shape
    z_mid = z // 2

    a_slice = a[z_mid]
    b_slice = b[z_mid]

    # Depth-dose integral transversal
    prof_a = a.sum(axis=(1, 2))
    prof_b = b.sum(axis=(1, 2))

    z_mm = np.arange(z, dtype=np.float64) * float(sp_a[2])

    # Escalado por número de eventos para comparar forma física
    scale = (args.events_a / args.events_b) if args.events_b > 0 else 1.0
    prof_b_scaled = prof_b * scale

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    im0 = axes[0, 0].imshow(np.log10(a_slice + 1e-12), cmap="inferno")
    axes[0, 0].set_title(f"{args.label_a} log10 dose (z={z_mid})")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(np.log10(b_slice + 1e-12), cmap="inferno")
    axes[0, 1].set_title(f"{args.label_b} log10 dose (z={z_mid})")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[1, 0].plot(z_mm, normalize(prof_a), label=f"{args.label_a} norm", linewidth=2)
    axes[1, 0].plot(z_mm, normalize(prof_b), label=f"{args.label_b} norm", linewidth=2)
    axes[1, 0].set_title("Depth-dose normalizado")
    axes[1, 0].set_xlabel("z (mm)")
    axes[1, 0].set_ylabel("norm")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(z_mm, prof_a, label=f"{args.label_a} raw", linewidth=2)
    axes[1, 1].plot(z_mm, prof_b_scaled, label=f"{args.label_b} escalado", linewidth=2)
    axes[1, 1].set_title("Depth-dose (comparación con escalado por eventos)")
    axes[1, 1].set_xlabel("z (mm)")
    axes[1, 1].set_ylabel("edep")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"Figura guardada en: {args.out}")


if __name__ == "__main__":
    main()
