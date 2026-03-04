from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def infer_ray_axis(ref: np.ndarray, beam_mask: np.ndarray | None) -> int:
    if beam_mask is not None and np.count_nonzero(beam_mask > 0.5) > 10:
        coords = np.argwhere(beam_mask > 0.5)
    else:
        thr = np.percentile(ref[ref > 0], 90) if np.any(ref > 0) else 0.0
        coords = np.argwhere(ref >= thr)
    if coords.shape[0] < 5:
        return 2
    ext = coords.max(axis=0) - coords.min(axis=0)
    return int(np.argmax(ext))


def plot_profiles_comparison(ref: np.ndarray, val: np.ndarray, pred: np.ndarray, axis: int, spacing: np.ndarray, out_dir: str) -> None:
    """Perfil longitudinal y transversal con ref/val/pred"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=140)

    # Eje del haz (longitudinal)
    axes_to_reduce = tuple(set([0, 1, 2]) - {axis})
    prof_ref = np.mean(ref, axis=axes_to_reduce)
    prof_val = np.mean(val, axis=axes_to_reduce)
    prof_pred = np.mean(pred, axis=axes_to_reduce)

    z = np.arange(len(prof_ref)) * float(spacing[axis])
    axes[0].plot(z, prof_ref, "k-", linewidth=2.2, label="Ref (1M)")
    axes[0].plot(z, prof_val, "b:", linewidth=2.0, label="Val (2k)")
    axes[0].plot(z, prof_pred, "r--", linewidth=2.0, label="Pred")
    axes[0].set_xlabel(f"Profundidad (mm, eje {axis})")
    axes[0].set_ylabel("Dosis (Gy)")
    axes[0].set_title("Perfil Longitudinal")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.25)

    # Transversal (corte central)
    center = ref.shape[axis] // 2
    if axis == 2:
        trans_ref = ref[center, :, :]
        trans_val = val[center, :, :]
        trans_pred = pred[center, :, :]
        lat_axis = 1
    elif axis == 1:
        trans_ref = ref[:, center, :]
        trans_val = val[:, center, :]
        trans_pred = pred[:, center, :]
        lat_axis = 2
    else:
        trans_ref = ref[:, :, center]
        trans_val = val[:, :, center]
        trans_pred = pred[:, :, center]
        lat_axis = 1

    prof_trans_ref = np.mean(trans_ref, axis=0)
    prof_trans_val = np.mean(trans_val, axis=0)
    prof_trans_pred = np.mean(trans_pred, axis=0)

    x = np.arange(len(prof_trans_ref)) * float(spacing[lat_axis])
    axes[1].plot(x, prof_trans_ref, "k-", linewidth=2.2, label="Ref (1M)")
    axes[1].plot(x, prof_trans_val, "b:", linewidth=2.0, label="Val (2k)")
    axes[1].plot(x, prof_trans_pred, "r--", linewidth=2.0, label="Pred")
    axes[1].set_xlabel("Posición lateral (mm)")
    axes[1].set_ylabel("Dosis (Gy)")
    axes[1].set_title("Perfil Transversal")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/01_profiles_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_dvh_comparison(ref: np.ndarray, val: np.ndarray, pred: np.ndarray, out_dir: str) -> None:
    """DVH de las 3 distribuciones"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)

    for d, label, style in [
        (ref, "Ref (1M)", "k-"),
        (val, "Val (2k)", "b:"),
        (pred, "Pred", "r--"),
    ]:
        dose_vals = np.sort(d.ravel())[::-1]
        vol_pct = np.arange(len(dose_vals)) / max(1, len(dose_vals)) * 100.0
        ax.plot(dose_vals, vol_pct, style, linewidth=2.0, label=label)

    ax.set_xlabel("Dosis (Gy)")
    ax.set_ylabel("Volumen (%)")
    ax.set_title("Histograma Dosis-Volumen (DVH)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/02_dvh_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_slices_comparison(ref: np.ndarray, val: np.ndarray, pred: np.ndarray, axis: int, out_dir: str) -> None:
    """Cortes 2D: ref, val, pred lado a lado"""
    center = ref.shape[axis] // 2

    if axis == 2:
        sl_ref, sl_val, sl_pred = ref[center, :, :], val[center, :, :], pred[center, :, :]
    elif axis == 1:
        sl_ref, sl_val, sl_pred = ref[:, center, :], val[:, center, :], pred[:, center, :]
    else:
        sl_ref, sl_val, sl_pred = ref[:, :, center], val[:, :, center], pred[:, :, center]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=140)
    vmin = 0.0
    vmax = max(float(np.max(sl_ref)), float(np.max(sl_val)), float(np.max(sl_pred)))

    im0 = axes[0].imshow(sl_ref, cmap="jet", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ref (1M)")
    plt.colorbar(im0, ax=axes[0], label="Gy")

    im1 = axes[1].imshow(sl_val, cmap="jet", vmin=vmin, vmax=vmax)
    axes[1].set_title("Val (2k)")
    plt.colorbar(im1, ax=axes[1], label="Gy")

    im2 = axes[2].imshow(sl_pred, cmap="jet", vmin=vmin, vmax=vmax)
    axes[2].set_title("Pred")
    plt.colorbar(im2, ax=axes[2], label="Gy")

    fig.tight_layout()
    fig.savefig(f"{out_dir}/03_slices_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_gamma_comparison(ref: np.ndarray, val: np.ndarray, pred: np.ndarray, out_dir: str) -> None:
    """Gamma index 1D: val vs ref, pred vs ref"""
    # Promedios sobre 2 ejes para obtener perfil 1D
    prof_ref = np.mean(ref, axis=(1, 2))
    prof_val = np.mean(val, axis=(1, 2))
    prof_pred = np.mean(pred, axis=(1, 2))

    def calc_gamma_1d(eval_dose: np.ndarray, ref_dose: np.ndarray, dd: float = 2.0, dta: float = 2.0) -> np.ndarray:
        gamma = np.ones_like(eval_dose, dtype=np.float64) * 2.0
        ref_max = float(np.max(ref_dose))
        dose_crit = (2.0 / 100.0) * ref_max if ref_max > 0 else 1e-8

        for i in range(len(eval_dose)):
            if ref_dose[i] < 0.1 * ref_max:
                continue
            dose_diff = np.abs(eval_dose - eval_dose[i])
            pos_diff = np.abs(np.arange(len(ref_dose), dtype=np.float64) - i)
            dd_term = (dose_diff / max(dose_crit, 1e-8)) ** 2
            dta_term = (pos_diff / max(dta, 1e-8)) ** 2
            g = np.sqrt(dd_term + dta_term)
            gamma[i] = float(np.min(g))
        return gamma

    gamma_val = calc_gamma_1d(prof_val, prof_ref)
    gamma_pred = calc_gamma_1d(prof_pred, prof_ref)

    z = np.arange(len(prof_ref), dtype=np.float64)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8), dpi=140)

    # Val vs Ref
    ax0.plot(z, prof_ref, "k-", linewidth=2.2, label="Ref (1M)")
    ax0.plot(z, prof_val, "b:", linewidth=2.0, label="Val (2k)")
    ax0.set_ylabel("Dosis (Gy)")
    ax0.set_title("Val (2k) vs Ref (1M)")
    ax0.legend(loc="upper left", fontsize=9)
    ax0.grid(alpha=0.25)

    ax0_twin = ax0.twinx()
    ax0_twin.plot(z, gamma_val, "b--", alpha=0.6, linewidth=2.0, label="Gamma (val)")
    ax0_twin.axhline(1.0, color="r", linestyle="--", alpha=0.5, linewidth=1.5)
    ax0_twin.set_ylabel("Gamma Index", color="b")
    ax0_twin.tick_params(axis="y", labelcolor="b")
    ax0_twin.legend(loc="upper right", fontsize=9)

    # Pred vs Ref
    ax1.plot(z, prof_ref, "k-", linewidth=2.2, label="Ref (1M)")
    ax1.plot(z, prof_pred, "r--", linewidth=2.0, label="Pred")
    ax1.set_ylabel("Dosis (Gy)")
    ax1.set_xlabel("Profundidad (voxeles)")
    ax1.set_title("Pred vs Ref (1M)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.25)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(z, gamma_pred, "r--", alpha=0.6, linewidth=2.0, label="Gamma (pred)")
    ax1_twin.axhline(1.0, color="g", linestyle="--", alpha=0.5, linewidth=1.5)
    ax1_twin.set_ylabel("Gamma Index", color="r")
    ax1_twin.tick_params(axis="y", labelcolor="r")
    ax1_twin.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/04_gamma_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive comparison: ref/val/pred in all plots")
    parser.add_argument("--input-npz", type=Path, required=True, help="NPZ with d_low, d_high, pred_phys")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/comprehensive_compare"))
    args = parser.parse_args()

    if not args.input_npz.exists():
        raise FileNotFoundError(f"NPZ no existe: {args.input_npz}")

    d = np.load(args.input_npz)

    ref = d["d_high"].astype(np.float64)
    val = d["d_low"].astype(np.float64)

    # Intenta encontrar pred bajo varios nombres
    pred_key = next((k for k in ["pred_phys", "pred", "prediction"] if k in d.files), None)
    if pred_key is None:
        raise KeyError("No pred_phys/pred found in NPZ")

    pred = d[pred_key].astype(np.float64)

    if ref.shape != val.shape or ref.shape != pred.shape:
        raise ValueError(f"Shape mismatch: ref={ref.shape} val={val.shape} pred={pred.shape}")

    spacing = d.get("spacing_mm", np.array([1.0, 1.0, 1.0])).astype(np.float64)
    if spacing.size != 3:
        spacing = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    beam_mask = None
    if "beam_mask" in d.files:
        bm = d["beam_mask"]
        if bm.shape == ref.shape:
            beam_mask = bm.astype(np.float64)

    beam_axis = infer_ray_axis(ref, beam_mask)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Generar todas las gráficas
    plot_profiles_comparison(ref, val, pred, beam_axis, spacing, str(args.out_dir))
    plot_dvh_comparison(ref, val, pred, str(args.out_dir))
    plot_slices_comparison(ref, val, pred, beam_axis, str(args.out_dir))
    plot_gamma_comparison(ref, val, pred, str(args.out_dir))

    summary = {
        "input_npz": str(args.input_npz),
        "pred_key": pred_key,
        "beam_axis": int(beam_axis),
        "spacing": list(spacing),
        "ref_shape": list(ref.shape),
        "outputs": [
            "01_profiles_comparison.png",
            "02_dvh_comparison.png",
            "03_slices_comparison.png",
            "04_gamma_comparison.png",
        ],
    }

    json_path = args.out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"✓ Todas las gráficas generadas en: {args.out_dir}")
    print(f"  - Perfiles (longitudinal + transversal)")
    print(f"  - DVH (dosis-volumen)")
    print(f"  - Cortes 2D (axiales)")
    print(f"  - Gamma index (val vs ref, pred vs ref)")


if __name__ == "__main__":
    main()
