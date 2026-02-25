from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.gamma import gamma_report
from src.metrics.distal_range import distal_error_mm


@dataclass(frozen=True)
class RunKey:
    n_events: int
    seed: int


def parse_energy(name: str) -> int:
    m = re.match(r"E([0-9]+)", name)
    if not m:
        raise ValueError(f"Invalid energy dir: {name}")
    return int(m.group(1))


def parse_run_dir(name: str) -> RunKey:
    m = re.match(r"N([0-9]+)_seed([0-9]+)", name)
    if not m:
        raise ValueError(f"Invalid run dir (expected N<events>_seed<seed>): {name}")
    return RunKey(n_events=int(m.group(1)), seed=int(m.group(2)))


def load_dose(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def metrics_for_pair(
    low: np.ndarray,
    high: np.ndarray,
    spacing_xyz: np.ndarray,
    low_events: int,
    high_events: int,
) -> dict:
    scale = float(high_events) / float(low_events)
    low = low * scale

    high_peak = float(np.max(high))
    roi10 = high > 0.10 * high_peak

    diff = high - low
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # whole-volume gamma
    g_whole = gamma_report(high, low, criteria=((3.0, 3.0), (2.0, 2.0)))
    # clinically meaningful ROI gamma
    g_roi10 = gamma_report(high[roi10], low[roi10], criteria=((3.0, 3.0), (2.0, 2.0)))

    spacing_zyx = np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], dtype=np.float32)
    d_err_mm = float(distal_error_mm(high, low, spacing_mm=spacing_zyx, beam_axis=0))

    p_low = low.sum(axis=(1, 2))
    p_high = high.sum(axis=(1, 2))
    valid = p_high > 0.05 * np.max(p_high)
    rel_depth = np.abs(p_low - p_high) / (p_high + 1e-12)
    depth_rel_p95 = float(100.0 * np.percentile(rel_depth[valid], 95))

    return {
        "mae": mae,
        "rmse": rmse,
        "gamma_3_whole": float(g_whole["gamma_3pct"]),
        "gamma_2_whole": float(g_whole["gamma_2pct"]),
        "gamma_3_roi10": float(g_roi10["gamma_3pct"]),
        "gamma_2_roi10": float(g_roi10["gamma_2pct"]),
        "distal_error_mm": d_err_mm,
        "depth_rel_p95_pct": depth_rel_p95,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess minimum sufficient N events with seed stability")
    parser.add_argument("--sweep-root", type=Path, required=False, help="Root with N<events>_seed<seed>/E*/dose.mhd")
    parser.add_argument("--single-low-root", type=Path, required=False, help="Single low root with E*/dose.mhd")
    parser.add_argument("--single-low-events", type=int, required=False, help="Events for --single-low-root")
    parser.add_argument("--single-low-seed", type=int, default=0, help="Seed label for --single-low-root")
    parser.add_argument("--high-root", type=Path, required=True, help="Reference high-dose root with E*/dose.mhd")
    parser.add_argument("--high-events", type=int, default=500000)
    parser.add_argument("--out-dir", type=Path, required=True)

    parser.add_argument("--thr-gamma3-roi10", type=float, default=95.0)
    parser.add_argument("--thr-gamma2-roi10", type=float, default=85.0)
    parser.add_argument("--thr-distal-mm", type=float, default=1.5)
    parser.add_argument("--thr-depth-p95", type=float, default=20.0)

    parser.add_argument("--min-seeds", type=int, default=3)
    args = parser.parse_args()

    if args.sweep_root is None and args.single_low_root is None:
        raise ValueError("Provide either --sweep-root or --single-low-root")
    if args.sweep_root is not None and not args.sweep_root.exists():
        raise FileNotFoundError(f"Sweep root not found: {args.sweep_root}")
    if args.single_low_root is not None:
        if not args.single_low_root.exists():
            raise FileNotFoundError(f"Single low root not found: {args.single_low_root}")
        if args.single_low_events is None:
            raise ValueError("--single-low-events is required with --single-low-root")
    if not args.high_root.exists():
        raise FileNotFoundError(f"High root not found: {args.high_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    high_dirs = [d for d in sorted(args.high_root.iterdir()) if d.is_dir() and d.name.startswith("E")]
    if not high_dirs:
        raise RuntimeError("No E* dirs found in high root")

    high_by_energy = {}
    spacing_xyz = None
    for d in high_dirs:
        energy = parse_energy(d.name)
        p = d / "dose_voxelized_ct_edep.mhd"
        if not p.exists():
            continue
        img = sitk.ReadImage(str(p))
        high_by_energy[energy] = sitk.GetArrayFromImage(img).astype(np.float32)
        if spacing_xyz is None:
            spacing_xyz = np.array(img.GetSpacing(), dtype=np.float32)

    rows = []
    run_specs = []
    if args.sweep_root is not None:
        for run in sorted(args.sweep_root.iterdir()):
            if not run.is_dir():
                continue
            try:
                key = parse_run_dir(run.name)
            except ValueError:
                continue
            run_specs.append((run, key.n_events, key.seed))

    if args.single_low_root is not None:
        run_specs.append((args.single_low_root, int(args.single_low_events), int(args.single_low_seed)))

    for run, n_events, seed in run_specs:
        for e, high in high_by_energy.items():
            low_path = run / f"E{e}" / "dose_voxelized_ct_edep.mhd"
            if not low_path.exists():
                continue
            low = load_dose(low_path)
            m = metrics_for_pair(
                low=low,
                high=high,
                spacing_xyz=spacing_xyz,
                low_events=n_events,
                high_events=args.high_events,
            )
            rows.append(
                {
                    "n_events": n_events,
                    "seed": seed,
                    "energy_mev": e,
                    **m,
                }
            )

    if not rows:
        raise RuntimeError("No comparable runs found in sweep root")

    per_seed_csv = args.out_dir / "event_sufficiency_per_seed_energy.csv"
    with open(per_seed_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "n_events",
                "seed",
                "energy_mev",
                "mae",
                "rmse",
                "gamma_3_whole",
                "gamma_2_whole",
                "gamma_3_roi10",
                "gamma_2_roi10",
                "distal_error_mm",
                "depth_rel_p95_pct",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    # Aggregate by n_events over all seeds and energies
    from collections import defaultdict

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["n_events"]].append(r)

    summary_rows = []
    for n in sorted(grouped.keys()):
        rs = grouped[n]
        seeds = sorted({int(r["seed"]) for r in rs})
        n_seeds = len(seeds)

        # mean across all seed-energy cases
        g3 = np.array([r["gamma_3_roi10"] for r in rs], dtype=np.float32)
        g2 = np.array([r["gamma_2_roi10"] for r in rs], dtype=np.float32)
        dr = np.array([r["distal_error_mm"] for r in rs], dtype=np.float32)
        dp95 = np.array([r["depth_rel_p95_pct"] for r in rs], dtype=np.float32)

        # conservative criterion: p10 for gamma, p90 for errors
        g3_p10 = float(np.percentile(g3, 10))
        g2_p10 = float(np.percentile(g2, 10))
        dr_p90 = float(np.percentile(dr, 90))
        dp95_p90 = float(np.percentile(dp95, 90))

        pass_flag = (
            n_seeds >= args.min_seeds
            and g3_p10 >= args.thr_gamma3_roi10
            and g2_p10 >= args.thr_gamma2_roi10
            and dr_p90 <= args.thr_distal_mm
            and dp95_p90 <= args.thr_depth_p95
        )

        summary_rows.append(
            {
                "n_events": n,
                "n_seeds": n_seeds,
                "gamma3_roi10_mean": float(np.mean(g3)),
                "gamma3_roi10_p10": g3_p10,
                "gamma2_roi10_mean": float(np.mean(g2)),
                "gamma2_roi10_p10": g2_p10,
                "distal_mm_mean": float(np.mean(dr)),
                "distal_mm_p90": dr_p90,
                "depth_rel_p95_mean": float(np.mean(dp95)),
                "depth_rel_p95_p90": dp95_p90,
                "pass": int(pass_flag),
            }
        )

    summary_csv = args.out_dir / "event_sufficiency_summary_by_events.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "n_events",
                "n_seeds",
                "gamma3_roi10_mean",
                "gamma3_roi10_p10",
                "gamma2_roi10_mean",
                "gamma2_roi10_p10",
                "distal_mm_mean",
                "distal_mm_p90",
                "depth_rel_p95_mean",
                "depth_rel_p95_p90",
                "pass",
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)

    chosen = [r for r in summary_rows if int(r["pass"]) == 1]
    min_sufficient = int(chosen[0]["n_events"]) if chosen else None

    report = args.out_dir / "event_sufficiency_report.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write("# Event Sufficiency Report\n\n")
        if args.sweep_root is not None:
            f.write(f"- sweep root: {args.sweep_root}\n")
        if args.single_low_root is not None:
            f.write(f"- single low root: {args.single_low_root} (events={args.single_low_events}, seed={args.single_low_seed})\n")
        f.write(f"- high root: {args.high_root} (events={args.high_events})\n")
        f.write("- decision logic: pass if conservative quantiles satisfy all thresholds\n")
        f.write(f"- min seeds required: {args.min_seeds}\n\n")

        f.write("## Thresholds\n")
        f.write(f"- gamma 3%/3mm ROI>10% (p10) >= {args.thr_gamma3_roi10}\n")
        f.write(f"- gamma 2%/2mm ROI>10% (p10) >= {args.thr_gamma2_roi10}\n")
        f.write(f"- distal error mm (p90) <= {args.thr_distal_mm}\n")
        f.write(f"- depth relative p95 % (p90) <= {args.thr_depth_p95}\n\n")

        f.write("## By events\n")
        for r in summary_rows:
            f.write(
                f"- N={int(r['n_events'])}: seeds={int(r['n_seeds'])}, "
                f"g3_p10={r['gamma3_roi10_p10']:.2f}, g2_p10={r['gamma2_roi10_p10']:.2f}, "
                f"distal_p90={r['distal_mm_p90']:.2f} mm, depthP95_p90={r['depth_rel_p95_p90']:.2f}%, "
                f"pass={int(r['pass'])}\n"
            )

        f.write("\n## Decision\n")
        if min_sufficient is None:
            f.write("- No N tested meets all criteria. Increase events and/or relax thresholds based on protocol.\n")
        else:
            f.write(f"- Minimum sufficient events in tested set: **{min_sufficient}**\n")

    print(f"Saved per-seed CSV: {per_seed_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved report: {report}")
    if min_sufficient is None:
        print("Decision: no sufficient N found")
    else:
        print(f"Decision: minimum sufficient N = {min_sufficient}")


if __name__ == "__main__":
    main()
