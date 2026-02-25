from pathlib import Path
import argparse
import csv
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
        raise ValueError(f"No se pudo parsear energía desde {name}")
    return float(m.group(1))


def d_metrics(dose: np.ndarray, mask: np.ndarray) -> dict:
    vox = dose[mask]
    if vox.size == 0:
        return {"dmean": np.nan, "d95": np.nan, "d2": np.nan}
    dmean = float(np.mean(vox))
    d95 = float(np.percentile(vox, 5))
    d2 = float(np.percentile(vox, 98))
    return {"dmean": dmean, "d95": d95, "d2": d2}


def make_surrogate_masks(ct_zyx: np.ndarray) -> dict:
    return {
        "body": ct_zyx > -950,
        "soft_tissue": (ct_zyx > -300) & (ct_zyx <= 300),
        "bone": ct_zyx > 300,
        "air_lung": ct_zyx <= -300,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validación clínica low vs high con datos disponibles")
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--low-root", type=Path, required=True)
    parser.add_argument("--high-root", type=Path, required=True)
    parser.add_argument("--low-events", type=float, default=10000)
    parser.add_argument("--high-events", type=float, default=500000)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    ct_img = sitk.ReadImage(str(args.ct))
    ct = sitk.GetArrayFromImage(ct_img).astype(np.float32)  # z,y,x
    spacing_xyz = np.array(ct_img.GetSpacing(), dtype=np.float32)  # x,y,z

    masks = make_surrogate_masks(ct)
    scale = float(args.high_events) / float(args.low_events)

    rows = []

    dvh_series = {k: {"energy": [], "dmean_low": [], "dmean_high": [], "d95_low": [], "d95_high": [], "d2_low": [], "d2_high": []} for k in masks.keys()}

    for d in sorted(args.high_root.iterdir()):
        if not d.is_dir() or not d.name.startswith("E"):
            continue

        low_path = args.low_root / d.name / "dose_voxelized_ct_edep.mhd"
        high_path = d / "dose_voxelized_ct_edep.mhd"
        if not low_path.exists() or not high_path.exists():
            continue

        energy = parse_energy_from_dir(d.name)
        low = sitk.GetArrayFromImage(sitk.ReadImage(str(low_path))).astype(np.float32) * scale
        high = sitk.GetArrayFromImage(sitk.ReadImage(str(high_path))).astype(np.float32)

        mae = float(np.mean(np.abs(high - low)))
        rmse = float(np.sqrt(np.mean((high - low) ** 2)))
        nmae_peak = float(100.0 * mae / (np.max(high) + 1e-12))

        total_high = float(np.sum(high))
        total_low = float(np.sum(low))
        total_ratio = float(total_low / (total_high + 1e-12))
        peak_high = float(np.max(high))
        peak_low = float(np.max(low))
        peak_ratio = float(peak_low / (peak_high + 1e-12))

        high_peak = peak_high
        roi_1 = high > 0.01 * high_peak
        roi_10 = high > 0.10 * high_peak

        def masked_metrics(mask: np.ndarray) -> tuple[float, float, float]:
            diff = high[mask] - low[mask]
            mae_m = float(np.mean(np.abs(diff)))
            rmse_m = float(np.sqrt(np.mean(diff ** 2)))
            rel = np.abs(diff) / (np.abs(high[mask]) + 1e-12)
            rel_p95 = float(100.0 * np.percentile(rel, 95))
            return mae_m, rmse_m, rel_p95

        mae_roi1, rmse_roi1, rel95_roi1 = masked_metrics(roi_1)
        mae_roi10, rmse_roi10, rel95_roi10 = masked_metrics(roi_10)

        g = gamma_report(high, low, criteria=((3.0, 3.0), (2.0, 2.0)))
        g_roi1 = gamma_report(high[roi_1], low[roi_1], criteria=((3.0, 3.0), (2.0, 2.0)))
        g_roi10 = gamma_report(high[roi_10], low[roi_10], criteria=((3.0, 3.0), (2.0, 2.0)))

        # arrays z,y,x -> beam axis z => axis=0 and spacing_mm aligned as [z,y,x]
        spacing_zyx = np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], dtype=np.float32)
        d_err = float(distal_error_mm(high, low, spacing_mm=spacing_zyx, beam_axis=0))

        p_low = low.sum(axis=(1, 2))
        p_high = high.sum(axis=(1, 2))
        p_corr = float(np.corrcoef(p_low, p_high)[0, 1])
        valid = p_high > 0.05 * np.max(p_high)
        rel_depth = np.abs(p_low - p_high) / (p_high + 1e-12)
        depth_rel_mean = float(100.0 * np.mean(rel_depth[valid]))
        depth_rel_p95 = float(100.0 * np.percentile(rel_depth[valid], 95))

        for mask_name, mask in masks.items():
            m_low = d_metrics(low, mask)
            m_high = d_metrics(high, mask)
            dvh_series[mask_name]["energy"].append(energy)
            dvh_series[mask_name]["dmean_low"].append(m_low["dmean"])
            dvh_series[mask_name]["dmean_high"].append(m_high["dmean"])
            dvh_series[mask_name]["d95_low"].append(m_low["d95"])
            dvh_series[mask_name]["d95_high"].append(m_high["d95"])
            dvh_series[mask_name]["d2_low"].append(m_low["d2"])
            dvh_series[mask_name]["d2_high"].append(m_high["d2"])

        rows.append({
            "energy_mev": energy,
            "mae": mae,
            "rmse": rmse,
            "nmae_pct_of_peak": nmae_peak,
            "total_high": total_high,
            "total_low": total_low,
            "total_ratio_low_over_high": total_ratio,
            "peak_high": peak_high,
            "peak_low": peak_low,
            "peak_ratio_low_over_high": peak_ratio,
            "gamma_3pct": float(g["gamma_3pct"]),
            "gamma_2pct": float(g["gamma_2pct"]),
            "gamma_3pct_3mm_low10k_vs_high500k_whole": float(g["gamma_3pct"]),
            "gamma_2pct_2mm_low10k_vs_high500k_whole": float(g["gamma_2pct"]),
            "mae_roi1": mae_roi1,
            "rmse_roi1": rmse_roi1,
            "relerr_p95_roi1_pct": rel95_roi1,
            "gamma_3pct_roi1": float(g_roi1["gamma_3pct"]),
            "gamma_2pct_roi1": float(g_roi1["gamma_2pct"]),
            "mae_roi10": mae_roi10,
            "rmse_roi10": rmse_roi10,
            "relerr_p95_roi10_pct": rel95_roi10,
            "gamma_3pct_roi10": float(g_roi10["gamma_3pct"]),
            "gamma_2pct_roi10": float(g_roi10["gamma_2pct"]),
            "gamma_3pct_3mm_low10k_vs_high500k_roi10": float(g_roi10["gamma_3pct"]),
            "gamma_2pct_2mm_low10k_vs_high500k_roi10": float(g_roi10["gamma_2pct"]),
            "distal_error_mm": d_err,
            "depth_profile_corr": p_corr,
            "depth_rel_mean_pct": depth_rel_mean,
            "depth_rel_p95_pct": depth_rel_p95,
        })

    if not rows:
        raise RuntimeError("No se encontraron pares low/high para validar")

    rows.sort(key=lambda r: r["energy_mev"])

    csv_path = args.out_dir / "clinical_validation_low_vs_high.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "energy_mev",
                "mae",
                "rmse",
                "nmae_pct_of_peak",
                "total_high",
                "total_low",
                "total_ratio_low_over_high",
                "peak_high",
                "peak_low",
                "peak_ratio_low_over_high",
                "gamma_3pct",
                "gamma_2pct",
                "gamma_3pct_3mm_low10k_vs_high500k_whole",
                "gamma_2pct_2mm_low10k_vs_high500k_whole",
                "mae_roi1",
                "rmse_roi1",
                "relerr_p95_roi1_pct",
                "gamma_3pct_roi1",
                "gamma_2pct_roi1",
                "mae_roi10",
                "rmse_roi10",
                "relerr_p95_roi10_pct",
                "gamma_3pct_roi10",
                "gamma_2pct_roi10",
                "gamma_3pct_3mm_low10k_vs_high500k_roi10",
                "gamma_2pct_2mm_low10k_vs_high500k_roi10",
                "distal_error_mm",
                "depth_profile_corr",
                "depth_rel_mean_pct",
                "depth_rel_p95_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Plot 1: core metrics vs energy
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    es = [r["energy_mev"] for r in rows]

    ax[0, 0].plot(es, [r["gamma_3pct"] for r in rows], "o-", label="3%/3mm whole (10k vs 500k)")
    ax[0, 0].plot(es, [r["gamma_2pct"] for r in rows], "s-", label="2%/2mm whole (10k vs 500k)")
    ax[0, 0].plot(es, [r["gamma_3pct_roi10"] for r in rows], "o--", label="3%/3mm ROI>10%")
    ax[0, 0].plot(es, [r["gamma_2pct_roi10"] for r in rows], "s--", label="2%/2mm ROI>10%")
    ax[0, 0].set_title("Gamma pass-rate: low10k (scaled) vs high500k")
    ax[0, 0].set_xlabel("Energía (MeV)")
    ax[0, 0].set_ylabel("Passing rate (%)")
    ax[0, 0].grid(alpha=0.3)
    ax[0, 0].legend()

    ax[0, 1].plot(es, [r["distal_error_mm"] for r in rows], "o-")
    ax[0, 1].set_title("Error distal")
    ax[0, 1].set_xlabel("Energía (MeV)")
    ax[0, 1].set_ylabel("mm")
    ax[0, 1].grid(alpha=0.3)

    ax[1, 0].plot(es, [r["depth_rel_mean_pct"] for r in rows], "o-", label="mean")
    ax[1, 0].plot(es, [r["depth_rel_p95_pct"] for r in rows], "s-", label="p95")
    ax[1, 0].set_title("Error relativo depth-dose")
    ax[1, 0].set_xlabel("Energía (MeV)")
    ax[1, 0].set_ylabel("%")
    ax[1, 0].grid(alpha=0.3)
    ax[1, 0].legend()

    ax[1, 1].plot(es, [r["nmae_pct_of_peak"] for r in rows], "o-", label="whole")
    ax[1, 1].plot(es, [r["relerr_p95_roi1_pct"] for r in rows], "s-", label="p95 ROI>1%")
    ax[1, 1].plot(es, [r["relerr_p95_roi10_pct"] for r in rows], "^-", label="p95 ROI>10%")
    ax[1, 1].set_title("Error relativo")
    ax[1, 1].set_xlabel("Energía (MeV)")
    ax[1, 1].set_ylabel("%")
    ax[1, 1].grid(alpha=0.3)
    ax[1, 1].legend()

    fig.savefig(args.out_dir / "clinical_validation_metrics.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
    ax.plot(es, [r["gamma_3pct_3mm_low10k_vs_high500k_whole"] for r in rows], "o-", label="3%/3mm whole")
    ax.plot(es, [r["gamma_2pct_2mm_low10k_vs_high500k_whole"] for r in rows], "s-", label="2%/2mm whole")
    ax.plot(es, [r["gamma_3pct_3mm_low10k_vs_high500k_roi10"] for r in rows], "o--", label="3%/3mm ROI>10%")
    ax.plot(es, [r["gamma_2pct_2mm_low10k_vs_high500k_roi10"] for r in rows], "s--", label="2%/2mm ROI>10%")
    ax.set_title("Gamma explicit: 10k vs 500k")
    ax.set_xlabel("Energía (MeV)")
    ax.set_ylabel("Passing rate (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(args.out_dir / "clinical_validation_gamma_10k_vs_500k.png", dpi=170)
    plt.close(fig)

    # Plot 1b: explicit low vs high comparisons
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    ax[0].plot(es, [r["total_high"] for r in rows], "o-", label="High total")
    ax[0].plot(es, [r["total_low"] for r in rows], "s--", label="Low total (scaled)")
    ax[0].set_title("Total deposited dose")
    ax[0].set_xlabel("Energía (MeV)")
    ax[0].set_ylabel("a.u.")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    ax[1].plot(es, [100.0 * r["total_ratio_low_over_high"] for r in rows], "o-", label="Total ratio")
    ax[1].plot(es, [100.0 * r["peak_ratio_low_over_high"] for r in rows], "s-", label="Peak ratio")
    ax[1].set_title("Low / High ratio")
    ax[1].set_xlabel("Energía (MeV)")
    ax[1].set_ylabel("%")
    ax[1].grid(alpha=0.3)
    ax[1].legend()

    fig.savefig(args.out_dir / "clinical_validation_low_high_ratio.png", dpi=170)
    plt.close(fig)

    # Plot 1c: depth-dose overlays (normalized)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for idx, r in enumerate(rows):
        energy = r["energy_mev"]
        low_path = args.low_root / f"E{int(energy)}" / "dose_voxelized_ct_edep.mhd"
        high_path = args.high_root / f"E{int(energy)}" / "dose_voxelized_ct_edep.mhd"
        low = sitk.GetArrayFromImage(sitk.ReadImage(str(low_path))).astype(np.float32) * scale
        high = sitk.GetArrayFromImage(sitk.ReadImage(str(high_path))).astype(np.float32)
        p_low = low.sum(axis=(1, 2))
        p_high = high.sum(axis=(1, 2))
        p_low = p_low / (np.max(p_low) + 1e-12)
        p_high = p_high / (np.max(p_high) + 1e-12)
        z = np.arange(p_high.shape[0]) * float(spacing_xyz[2])
        c = cmap(idx % 10)
        ax.plot(z, p_high, color=c, linestyle="-", linewidth=2, label=f"E{int(energy)} high")
        ax.plot(z, p_low, color=c, linestyle="--", linewidth=1.5, label=f"E{int(energy)} low")
    ax.set_title("Depth-dose overlay (normalized)")
    ax.set_xlabel("Depth z (mm)")
    ax.set_ylabel("Normalized dose")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(args.out_dir / "clinical_validation_depthdose_overlay.png", dpi=170)
    plt.close(fig)

    # Plot 2: surrogate DVH metrics by structure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()
    for i, (mask_name, series) in enumerate(dvh_series.items()):
        if i >= 4:
            break
        order = np.argsort(series["energy"])
        e = np.array(series["energy"])[order]

        d95_low = np.array(series["d95_low"])[order]
        d95_high = np.array(series["d95_high"])[order]
        dmean_low = np.array(series["dmean_low"])[order]
        dmean_high = np.array(series["dmean_high"])[order]

        axes[i].plot(e, d95_high, "o-", label="D95 high")
        axes[i].plot(e, d95_low, "o--", label="D95 low")
        axes[i].plot(e, dmean_high, "s-", label="Dmean high")
        axes[i].plot(e, dmean_low, "s--", label="Dmean low")
        axes[i].set_title(f"DVH surrogate: {mask_name}")
        axes[i].set_xlabel("Energía (MeV)")
        axes[i].set_ylabel("Dose (a.u.)")
        axes[i].grid(alpha=0.3)
        if i == 0:
            axes[i].legend(fontsize=8)

    fig.savefig(args.out_dir / "clinical_validation_dvh_surrogate.png", dpi=170)
    plt.close(fig)

    # Markdown report
    report = args.out_dir / "clinical_validation_report.md"
    mean_vals = {
        "mae": float(np.mean([r["mae"] for r in rows])),
        "rmse": float(np.mean([r["rmse"] for r in rows])),
        "nmae": float(np.mean([r["nmae_pct_of_peak"] for r in rows])),
        "g3": float(np.mean([r["gamma_3pct"] for r in rows])),
        "g2": float(np.mean([r["gamma_2pct"] for r in rows])),
        "total_ratio": float(np.mean([r["total_ratio_low_over_high"] for r in rows])),
        "peak_ratio": float(np.mean([r["peak_ratio_low_over_high"] for r in rows])),
        "depth_corr": float(np.mean([r["depth_profile_corr"] for r in rows])),
        "g3_roi10": float(np.mean([r["gamma_3pct_roi10"] for r in rows])),
        "g2_roi10": float(np.mean([r["gamma_2pct_roi10"] for r in rows])),
        "rel95_roi1": float(np.mean([r["relerr_p95_roi1_pct"] for r in rows])),
        "rel95_roi10": float(np.mean([r["relerr_p95_roi10_pct"] for r in rows])),
        "dr": float(np.mean([r["distal_error_mm"] for r in rows])),
        "depth_mean": float(np.mean([r["depth_rel_mean_pct"] for r in rows])),
        "depth_p95": float(np.mean([r["depth_rel_p95_pct"] for r in rows])),
    }

    with open(report, "w", encoding="utf-8") as f:
        f.write("# Clinical Validation (low vs high)\n\n")
        f.write(f"- CT: {args.ct}\n")
        f.write(f"- low root: {args.low_root} (events={args.low_events})\n")
        f.write(f"- high root: {args.high_root} (events={args.high_events})\n")
        f.write(f"- scaling applied: low * ({args.high_events}/{args.low_events})\n\n")
        f.write("## Gamma parameters used\n")
        f.write("- Reference dose (d_ref): high 500k\n")
        f.write("- Evaluation dose (d_eval): low 10k scaled x50\n")
        f.write("- Criteria shown: 3%/3mm and 2%/2mm\n")
        f.write("- Note: implementation is dose-difference simplified (distance term not explicitly searched).\n\n")

        f.write("## Global summary\n")
        f.write(f"- MAE mean: {mean_vals['mae']:.4f}\n")
        f.write(f"- RMSE mean: {mean_vals['rmse']:.4f}\n")
        f.write(f"- NMAE (% peak) mean: {mean_vals['nmae']:.4f}\n")
        f.write(f"- Gamma 3% mean: {mean_vals['g3']:.3f}%\n")
        f.write(f"- Gamma 2% mean: {mean_vals['g2']:.3f}%\n")
        f.write(f"- Total dose ratio mean (low/high): {100.0 * mean_vals['total_ratio']:.3f}%\n")
        f.write(f"- Peak dose ratio mean (low/high): {100.0 * mean_vals['peak_ratio']:.3f}%\n")
        f.write(f"- Depth profile correlation mean: {mean_vals['depth_corr']:.5f}\n")
        f.write(f"- Gamma 3% mean (ROI >10% peak): {mean_vals['g3_roi10']:.3f}%\n")
        f.write(f"- Gamma 2% mean (ROI >10% peak): {mean_vals['g2_roi10']:.3f}%\n")
        f.write(f"- Relative error p95 mean (ROI >1% peak): {mean_vals['rel95_roi1']:.3f}%\n")
        f.write(f"- Relative error p95 mean (ROI >10% peak): {mean_vals['rel95_roi10']:.3f}%\n")
        f.write(f"- Distal error mean: {mean_vals['dr']:.3f} mm\n")
        f.write(f"- Depth relative mean error: {mean_vals['depth_mean']:.3f}%\n")
        f.write(f"- Depth relative p95 error: {mean_vals['depth_p95']:.3f}%\n\n")

        f.write("## Explicit low vs high by energy\n")
        for r in rows:
            f.write(
                f"- E{int(r['energy_mev'])}: total low/high={100.0 * r['total_ratio_low_over_high']:.2f}%, "
                f"peak low/high={100.0 * r['peak_ratio_low_over_high']:.2f}%, "
                f"depth corr={r['depth_profile_corr']:.5f}, "
                f"gamma 3%/3mm whole={r['gamma_3pct_3mm_low10k_vs_high500k_whole']:.3f}%, "
                f"gamma 2%/2mm whole={r['gamma_2pct_2mm_low10k_vs_high500k_whole']:.3f}%, "
                f"gamma 3%/3mm ROI10={r['gamma_3pct_3mm_low10k_vs_high500k_roi10']:.3f}%, "
                f"gamma 2%/2mm ROI10={r['gamma_2pct_2mm_low10k_vs_high500k_roi10']:.3f}%\n"
            )

        f.write("## Important caveats\n")
        f.write("- Gamma used here is simplified dose-difference only (no full 3D distance-to-agreement search).\n")
        f.write("- DVH is surrogate based on CT-threshold masks (no RTSTRUCT available yet).\n")
        f.write("- Results are valid for current setup/beamlet protocol and should be extended with more patients.\n")

    print(f"Saved CSV: {csv_path}")
    print(f"Saved report: {report}")
    print(f"Saved figures: {args.out_dir / 'clinical_validation_metrics.png'} and {args.out_dir / 'clinical_validation_dvh_surrogate.png'}")


if __name__ == "__main__":
    main()
