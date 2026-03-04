from __future__ import annotations

import argparse
import csv
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


def nmae(eval_dose: np.ndarray, ref_dose: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        a = eval_dose.ravel()
        b = ref_dose.ravel()
    else:
        a = eval_dose[mask]
        b = ref_dose[mask]
    denom = max(float(np.max(ref_dose)), 1e-8)
    return float(np.mean(np.abs(a - b)) / denom)


def dxx(dose_vals: np.ndarray, volume_percent: float) -> float:
    if dose_vals.size == 0:
        return 0.0
    p = max(0.0, min(100.0, 100.0 - volume_percent))
    return float(np.percentile(dose_vals, p))


def dvh_curve(dose_vals: np.ndarray, bins: np.ndarray) -> np.ndarray:
    if dose_vals.size == 0:
        return np.zeros_like(bins)
    s = np.sort(dose_vals)
    n = s.size
    idx = np.searchsorted(s, bins, side="left")
    return (n - idx) / max(1, n) * 100.0


def gamma_3d_same_grid(
    eval_dose: np.ndarray,
    ref_dose: np.ndarray,
    spacing_mm: np.ndarray,
    mask: np.ndarray,
    dose_percent: float,
    dta_mm: float,
    max_voxels: int,
    seed: int,
) -> tuple[float, float, int]:
    ref_max = max(float(np.max(ref_dose)), 1e-8)
    dose_crit = (dose_percent / 100.0) * ref_max

    dz, dy, dx = [max(float(v), 1e-6) for v in spacing_mm]
    rz = int(np.ceil(dta_mm / dz))
    ry = int(np.ceil(dta_mm / dy))
    rx = int(np.ceil(dta_mm / dx))

    offsets = []
    for oz in range(-rz, rz + 1):
        for oy in range(-ry, ry + 1):
            for ox in range(-rx, rx + 1):
                dist = np.sqrt((oz * dz) ** 2 + (oy * dy) ** 2 + (ox * dx) ** 2)
                if dist <= dta_mm + 1e-9:
                    offsets.append((oz, oy, ox, dist))

    candidates = np.argwhere(mask)
    if candidates.shape[0] == 0:
        return 0.0, 0.0, 0

    if candidates.shape[0] > max_voxels:
        rng = np.random.default_rng(seed)
        sel = rng.choice(candidates.shape[0], size=max_voxels, replace=False)
        candidates = candidates[sel]

    zmax, ymax, xmax = ref_dose.shape
    gammas = np.empty(candidates.shape[0], dtype=np.float64)

    for i, (z, y, x) in enumerate(candidates):
        ref_val = ref_dose[z, y, x]
        best = np.inf
        for oz, oy, ox, dist in offsets:
            zz = z + oz
            yy = y + oy
            xx = x + ox
            if zz < 0 or yy < 0 or xx < 0 or zz >= zmax or yy >= ymax or xx >= xmax:
                continue
            dd = (eval_dose[zz, yy, xx] - ref_val) / dose_crit
            rr = dist / max(dta_mm, 1e-8)
            g = np.sqrt(dd * dd + rr * rr)
            if g < best:
                best = g
        gammas[i] = best

    pass_rate = float(np.mean(gammas <= 1.0) * 100.0)
    return pass_rate, float(np.mean(gammas)), int(gammas.size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare low-stat vs high-stat dose quality with quantitative metrics")
    parser.add_argument("--npz-eval", type=Path, required=True, help="NPZ of evaluated/noisy dose (e.g., 2k)")
    parser.add_argument("--npz-ref", type=Path, default=None, help="NPZ of reference dose (e.g., 1M). If omitted, use same NPZ")
    parser.add_argument("--eval-key", type=str, default="d_low")
    parser.add_argument("--ref-key", type=str, default="d_high")
    parser.add_argument("--beam-mask-key", type=str, default="beam_mask")
    parser.add_argument("--spacing-key", type=str, default="spacing_mm")
    parser.add_argument("--dose-scale-eval", type=float, default=1.0, help="Optional multiplier for eval dose before comparing")
    parser.add_argument("--dose-threshold-percent", type=float, default=50.0, help="Mask threshold for high-dose region metrics")
    parser.add_argument("--gamma-dose-percent", type=float, default=2.0)
    parser.add_argument("--gamma-dta-mm", type=float, default=2.0)
    parser.add_argument("--gamma-max-voxels", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/quality_compare"))
    args = parser.parse_args()

    if not args.npz_eval.exists():
        raise FileNotFoundError(f"No existe npz-eval: {args.npz_eval}")

    npz_ref = args.npz_ref if args.npz_ref is not None else args.npz_eval
    if not npz_ref.exists():
        raise FileNotFoundError(f"No existe npz-ref: {npz_ref}")

    de = np.load(args.npz_eval)
    dr = np.load(npz_ref)

    if args.eval_key not in de.files:
        raise KeyError(f"{args.eval_key} no existe en {args.npz_eval}")
    if args.ref_key not in dr.files:
        raise KeyError(f"{args.ref_key} no existe en {npz_ref}")

    eval_dose = de[args.eval_key].astype(np.float64) * float(args.dose_scale_eval)
    ref_dose = dr[args.ref_key].astype(np.float64)

    if eval_dose.shape != ref_dose.shape:
        raise ValueError(f"Shape mismatch eval={eval_dose.shape} ref={ref_dose.shape}")

    spacing = dr[args.spacing_key].astype(np.float64) if args.spacing_key in dr.files else np.array([2.0, 2.0, 2.0], dtype=np.float64)
    if spacing.size != 3:
        spacing = np.array([2.0, 2.0, 2.0], dtype=np.float64)

    beam_mask = None
    if args.beam_mask_key in dr.files:
        bm = dr[args.beam_mask_key]
        if bm.shape == ref_dose.shape:
            beam_mask = bm.astype(np.float64)

    ray_axis = infer_ray_axis(ref_dose, beam_mask)

    thr = (args.dose_threshold_percent / 100.0) * max(float(np.max(ref_dose)), 1e-8)
    high_mask = ref_dose >= thr

    # uncertainty map if available
    uncertainty_stats = {}
    unc_candidates = ["uncertainty", "d_low_unc", "dose_unc", "d_unc"]
    unc_key = next((k for k in unc_candidates if k in de.files and de[k].shape == eval_dose.shape), None)
    if unc_key is not None:
        unc = de[unc_key].astype(np.float64)
        if np.max(unc) > 1.5:
            unc = unc / 100.0
        unc_roi = unc[high_mask]
        uncertainty_stats = {
            "unc_key": unc_key,
            "unc_mean_in_high_mask": float(np.mean(unc_roi)) if unc_roi.size else 0.0,
            "unc_p95_in_high_mask": float(np.percentile(unc_roi, 95)) if unc_roi.size else 0.0,
        }

    nmae_all = nmae(eval_dose, ref_dose, None)
    nmae_high = nmae(eval_dose, ref_dose, high_mask)

    gamma_pass, gamma_mean, gamma_n = gamma_3d_same_grid(
        eval_dose=eval_dose,
        ref_dose=ref_dose,
        spacing_mm=spacing,
        mask=high_mask,
        dose_percent=float(args.gamma_dose_percent),
        dta_mm=float(args.gamma_dta_mm),
        max_voxels=int(args.gamma_max_voxels),
        seed=int(args.seed),
    )

    peak_idx = np.unravel_index(int(np.argmax(ref_dose)), ref_dose.shape)

    line_sel = [slice(None), slice(None), slice(None)]
    for ax in range(3):
        if ax != ray_axis:
            line_sel[ax] = peak_idx[ax]
    line_sel = tuple(line_sel)

    eval_line = eval_dose[line_sel]
    ref_line = ref_dose[line_sel]

    # transverse on first non-ray axis
    trans_axis = 0 if ray_axis != 0 else 1
    line_sel_t = [peak_idx[0], peak_idx[1], peak_idx[2]]
    line_sel_t[trans_axis] = slice(None)
    line_sel_t = tuple(line_sel_t)
    eval_line_t = eval_dose[line_sel_t]
    ref_line_t = ref_dose[line_sel_t]

    # roughness in plateau where ref line is mid-dose
    ref_line_n = ref_line / max(float(np.max(ref_line)), 1e-8)
    plateau_mask = (ref_line_n >= 0.2) & (ref_line_n <= 0.6)
    rough_eval = float(np.std(eval_line[plateau_mask])) if np.any(plateau_mask) else float(np.std(eval_line))
    rough_ref = float(np.std(ref_line[plateau_mask])) if np.any(plateau_mask) else float(np.std(ref_line))

    # DVH-like for Body and High-mask
    body_mask = np.ones_like(ref_dose, dtype=bool)
    region_masks = {
        "Body": body_mask,
        f"HighMask_{int(args.dose_threshold_percent)}pct": high_mask,
    }

    dose_max = max(float(np.max(ref_dose)), float(np.max(eval_dose)), 1e-8) * 1.05
    bins = np.linspace(0.0, dose_max, 300)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # profiles figure
    fig1, ax1 = plt.subplots(1, 2, figsize=(12, 4.5), dpi=140)
    depth_mm = np.arange(ref_line.shape[0]) * float(spacing[ray_axis])
    trans_mm = np.arange(ref_line_t.shape[0]) * float(spacing[trans_axis])

    ax1[0].plot(depth_mm, ref_line, label="Ref (1M)", lw=2)
    ax1[0].plot(depth_mm, eval_line, label="Eval (2k)", lw=1.6)
    ax1[0].set_title("Beam-axis profile")
    ax1[0].set_xlabel("Depth (mm)")
    ax1[0].set_ylabel("Dose")
    ax1[0].grid(alpha=0.25)
    ax1[0].legend()

    ax1[1].plot(trans_mm, ref_line_t, label="Ref (1M)", lw=2)
    ax1[1].plot(trans_mm, eval_line_t, label="Eval (2k)", lw=1.6)
    ax1[1].set_title("Transverse profile")
    ax1[1].set_xlabel("Distance (mm)")
    ax1[1].set_ylabel("Dose")
    ax1[1].grid(alpha=0.25)
    ax1[1].legend()

    fig1.tight_layout()
    prof_png = args.out_dir / "profiles_compare.png"
    fig1.savefig(prof_png, bbox_inches="tight")
    plt.close(fig1)

    # DVH figure
    fig2, ax2 = plt.subplots(figsize=(9, 6), dpi=140)
    colors = {"Body": "tab:blue", f"HighMask_{int(args.dose_threshold_percent)}pct": "tab:red"}

    dvh_table_rows = []
    for rname, rmask in region_masks.items():
        ref_vals = ref_dose[rmask]
        eval_vals = eval_dose[rmask]
        ref_curve = dvh_curve(ref_vals, bins)
        eval_curve = dvh_curve(eval_vals, bins)

        ax2.plot(bins, ref_curve, color=colors.get(rname, None), lw=2.0, linestyle="-", label=f"{rname} | Ref")
        ax2.plot(bins, eval_curve, color=colors.get(rname, None), lw=1.6, linestyle=":", label=f"{rname} | Eval")

        row = {
            "region": rname,
            "n_voxels": int(ref_vals.size),
            "D95_ref": dxx(ref_vals, 95.0),
            "D95_eval": dxx(eval_vals, 95.0),
            "Dmax_ref": float(np.max(ref_vals)) if ref_vals.size else 0.0,
            "Dmax_eval": float(np.max(eval_vals)) if eval_vals.size else 0.0,
        }
        dvh_table_rows.append(row)

    ax2.set_title("DVH-like: Ref vs Eval")
    ax2.set_xlabel("Dose")
    ax2.set_ylabel("Volume (%)")
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    dvh_png = args.out_dir / "dvh_compare.png"
    fig2.savefig(dvh_png, bbox_inches="tight")
    plt.close(fig2)

    # save csv table
    csv_path = args.out_dir / "dvh_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dvh_table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dvh_table_rows)

    summary = {
        "eval_npz": str(args.npz_eval),
        "ref_npz": str(npz_ref),
        "eval_key": args.eval_key,
        "ref_key": args.ref_key,
        "dose_scale_eval": float(args.dose_scale_eval),
        "ray_axis": int(ray_axis),
        "gamma": {
            "criteria": f"{args.gamma_dose_percent}%/{args.gamma_dta_mm}mm",
            "pass_rate_percent": float(gamma_pass),
            "gamma_mean": float(gamma_mean),
            "n_voxels": int(gamma_n),
        },
        "nmae": {
            "all": float(nmae_all),
            f"high_mask_{int(args.dose_threshold_percent)}pct": float(nmae_high),
        },
        "roughness": {
            "eval_plateau_std": float(rough_eval),
            "ref_plateau_std": float(rough_ref),
        },
        "uncertainty": uncertainty_stats,
        "outputs": {
            "profiles_png": str(prof_png),
            "dvh_png": str(dvh_png),
            "dvh_metrics_csv": str(csv_path),
        },
    }

    json_path = args.out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Done quality comparison")
    print(f"Profiles: {prof_png}")
    print(f"DVH: {dvh_png}")
    print(f"Summary: {json_path}")


if __name__ == "__main__":
    main()
