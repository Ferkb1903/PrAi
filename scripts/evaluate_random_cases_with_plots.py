from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io_npz import load_case_npz
from src.model.resunet3d import ResidualUNet3D


def resolve_npz_path(path: Path, manifest_dir: Path) -> Path | None:
    if path.exists():
        return path

    candidates: list[Path] = []
    candidates.append(manifest_dir / path.name)

    if not path.is_absolute():
        candidates.append((manifest_dir / path).resolve())

    parts = list(path.parts)
    filtered_parts = [part for part in parts if not part.startswith("chunk_")]
    if filtered_parts != parts:
        try:
            candidates.append(Path(*filtered_parts))
        except TypeError:
            pass

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def infer_ray_axis(mask: np.ndarray, spacing_mm: np.ndarray, fallback_axis: int) -> int:
    coords = np.argwhere(mask > 0.5)
    if coords.shape[0] < 10:
        return int(fallback_axis)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ext_vox = (maxs - mins + 1).astype(np.float64)
    ext_mm = ext_vox * spacing_mm.astype(np.float64)
    return int(np.argmax(ext_mm))


def gamma_1d(
    ref: np.ndarray,
    evalv: np.ndarray,
    spacing_mm_axis: float,
    dose_percent: float = 3.0,
    dta_mm: float = 3.0,
) -> tuple[np.ndarray, float]:
    ref = np.asarray(ref, dtype=np.float64)
    evalv = np.asarray(evalv, dtype=np.float64)

    n = ref.shape[0]
    dose_crit = (dose_percent / 100.0) * max(float(np.max(np.abs(ref))), 1e-8)
    dta_crit = max(float(dta_mm), 1e-8)

    pos = np.arange(n, dtype=np.float64) * float(spacing_mm_axis)
    gamma = np.empty(n, dtype=np.float64)

    for index in range(n):
        dd = (evalv - ref[index]) / dose_crit
        dr = (pos - pos[index]) / dta_crit
        gamma[index] = np.sqrt(np.min(dd * dd + dr * dr))

    pass_rate = float(np.mean(gamma <= 1.0) * 100.0)
    return gamma, pass_rate


def parse_offsets_mm(raw: str) -> list[float]:
    values = [float(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        return [0.0, 2.5, 5.0]
    return values


def extract_oriented_slice(volume: np.ndarray, fixed_axis: int, fixed_index: int, ray_axis: int) -> np.ndarray:
    plane = np.take(volume, indices=fixed_index, axis=fixed_axis)
    remaining_axes = [axis for axis in range(3) if axis != fixed_axis]
    ray_pos = remaining_axes.index(ray_axis)
    if ray_pos != 0:
        plane = np.moveaxis(plane, ray_pos, 0)
    return plane


def build_model_from_checkpoint(checkpoint: dict, device: torch.device) -> tuple[ResidualUNet3D, str]:
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    base_channels = int(ckpt_args.get("base_channels", 24))
    use_se_blocks = bool(ckpt_args.get("use_se_blocks", False))
    model_variant = str(ckpt_args.get("model_variant", "resunet_delta"))
    residual_mode = model_variant == "resunet_delta"

    model = ResidualUNet3D(
        in_channels=4,
        base_channels=base_channels,
        residual=residual_mode,
        use_se=use_se_blocks,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, model_variant


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate random NPZ cases and generate depth-profile + gamma plots")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True, help="One or more checkpoints")
    parser.add_argument("--num-cases", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/random_eval"))
    parser.add_argument("--low-scale-factor", type=float, default=200.0)
    parser.add_argument("--dose-percent", type=float, default=3.0)
    parser.add_argument("--dta-mm", type=float, default=3.0)
    parser.add_argument("--disable-beam-mask", action="store_true")
    parser.add_argument("--with-beamlet-style", action="store_true", help="Genera panel estilo CT/Low/High/Pred con cortes center y off-center")
    parser.add_argument("--offsets-mm", type=str, default="0,2.5,5", help="Offsets en mm para cortes off-center (ej: 0,2.5,5)")
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest no encontrado: {args.manifest}")

    for checkpoint in args.checkpoint:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint}")

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"Manifest vacío: {args.manifest}")

    manifest_dir = args.manifest.parent.resolve()
    resolved_paths: list[Path] = []
    missing = 0
    for row in rows:
        path = Path(row["npz_path"])
        resolved = resolve_npz_path(path, manifest_dir)
        if resolved is None:
            missing += 1
            continue
        resolved_paths.append(resolved)

    if not resolved_paths:
        raise RuntimeError("No se encontraron NPZ válidos en el manifest")

    random.seed(args.seed)
    sample_count = min(args.num_cases, len(resolved_paths))
    selected_paths = random.sample(resolved_paths, k=sample_count)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models: list[dict] = []
    for checkpoint_path in args.checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model, model_variant = build_model_from_checkpoint(checkpoint, device)
        models.append(
            {
                "path": checkpoint_path,
                "name": checkpoint_path.stem,
                "variant": model_variant,
                "model": model,
            }
        )

    print(f"Device: {device}")
    print(f"Manifest rows: {len(rows)} | resolved: {len(resolved_paths)} | missing: {missing}")
    print(f"Evaluating random cases: {sample_count}")

    case_rows: list[dict] = []
    offsets_mm = parse_offsets_mm(args.offsets_mm)

    for case_idx, npz_path in enumerate(selected_paths, start=1):
        case = load_case_npz(npz_path)

        d_low = case.d_low.astype(np.float32)
        d_high = case.d_high.astype(np.float32)
        spr = case.spr.astype(np.float32)
        spacing_mm = np.asarray(case.spacing_mm, dtype=np.float32)
        fallback_axis = int(case.beam_axis) if case.beam_axis is not None else 2
        beam_mask = (
            case.beam_mask.astype(np.float32)
            if (case.beam_mask is not None and not args.disable_beam_mask)
            else np.ones_like(d_low, dtype=np.float32)
        )

        ray_axis = infer_ray_axis(beam_mask, spacing_mm, fallback_axis)

        e0_map = np.full_like(d_low, fill_value=float(case.e0_mev), dtype=np.float32)
        x_np = np.stack([d_low, spr, e0_map, beam_mask], axis=0)[None, ...].astype(np.float32)
        x_t = torch.from_numpy(x_np).to(device)

        low_scaled = d_low.astype(np.float64) * float(args.low_scale_factor)
        high_ref = d_high.astype(np.float64)

        other_axes = tuple(ax for ax in range(3) if ax != ray_axis)
        low_depth = (low_scaled * beam_mask).sum(axis=other_axes)
        high_depth = (high_ref * beam_mask).sum(axis=other_axes)
        depth_mm = np.arange(high_depth.shape[0]) * float(spacing_mm[ray_axis])
        norm = max(float(np.max(high_depth)), 1e-8)

        per_model_results: list[dict] = []

        for model_idx, model_item in enumerate(models):
            model = model_item["model"]
            variant = model_item["variant"]

            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                out = model(x_t)
                pred_t = x_t[:, 0:1, ...] + out if variant == "resunet_delta" else out

            pred = pred_t[0, 0].detach().float().cpu().numpy().astype(np.float64)
            pred_scaled = pred * float(args.low_scale_factor)
            pred_depth = (pred_scaled * beam_mask).sum(axis=other_axes)

            gamma_vals, gamma_pass = gamma_1d(
                high_depth,
                pred_depth,
                spacing_mm_axis=float(spacing_mm[ray_axis]),
                dose_percent=float(args.dose_percent),
                dta_mm=float(args.dta_mm),
            )

            l1_depth = float(np.mean(np.abs(pred_depth - high_depth)))

            metric_row = {
                "case_index": case_idx,
                "npz_path": str(npz_path),
                "checkpoint": str(model_item["path"]),
                "checkpoint_name": model_item["name"],
                "model_variant": variant,
                "ray_axis": ray_axis,
                "spacing_mm_axis": float(spacing_mm[ray_axis]),
                "gamma_pass_rate": gamma_pass,
                "gamma_mean": float(np.mean(gamma_vals)),
                "gamma_max": float(np.max(gamma_vals)),
                "depth_l1": l1_depth,
            }
            case_rows.append(metric_row)

            per_model_results.append(
                {
                    "metric_row": metric_row,
                    "model_name": model_item["name"],
                    "variant": variant,
                    "pred_scaled": pred_scaled,
                    "pred_depth": pred_depth,
                    "gamma_vals": gamma_vals,
                    "gamma_pass": gamma_pass,
                }
            )

        fig, axes = plt.subplots(len(models), 2, figsize=(12, max(4, 4 * len(models))), dpi=140, squeeze=False)
        for model_idx, model_result in enumerate(per_model_results):
            ax0 = axes[model_idx, 0]
            ax0.plot(depth_mm, low_depth / norm, label=f"low*{args.low_scale_factor:g}", lw=1.2)
            ax0.plot(depth_mm, high_depth / norm, label="high", lw=2.0)
            ax0.plot(depth_mm, model_result["pred_depth"] / norm, label=f"pred*{args.low_scale_factor:g}", lw=1.6)
            ax0.set_ylabel("Norm dose")
            ax0.set_title(f"{model_result['model_name']} | axis={ray_axis}")
            ax0.grid(alpha=0.25)
            ax0.legend(fontsize=7)

            ax1 = axes[model_idx, 1]
            ax1.plot(depth_mm, model_result["gamma_vals"], color="tab:purple", lw=1.4, label="gamma")
            ax1.axhline(1.0, color="tab:red", ls="--", lw=1.1, label="pass")
            ax1.set_title(f"Gamma pass: {model_result['gamma_pass']:.2f}%")
            ax1.set_ylabel("Gamma")
            ax1.grid(alpha=0.25)
            ax1.legend(fontsize=7)

        for col in range(2):
            axes[-1, col].set_xlabel("Depth (mm)")

        fig.suptitle(f"Case {case_idx:02d}: {npz_path.name}", fontsize=10)
        fig.tight_layout()
        plot_path = plots_dir / f"case_{case_idx:02d}_{npz_path.stem}_depth_gamma.png"
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

        if args.with_beamlet_style:
            coords = np.argwhere(beam_mask > 0.5)
            transverse_axes = [axis for axis in range(3) if axis != ray_axis]
            if coords.shape[0] >= 10:
                mins = coords.min(axis=0)
                maxs = coords.max(axis=0)
                ext_mm = (maxs - mins + 1).astype(np.float64) * spacing_mm.astype(np.float64)
                shift_axis = max(transverse_axes, key=lambda axis: ext_mm[axis])
                center_index = int(round(float(np.mean(coords[:, shift_axis]))))
            else:
                shift_axis = transverse_axes[0]
                center_index = d_low.shape[shift_axis] // 2

            if len(transverse_axes) == 1:
                inplane_axis = transverse_axes[0]
            else:
                inplane_axis = transverse_axes[0] if transverse_axes[1] == shift_axis else transverse_axes[1]

            spr_min = float(np.percentile(spr, 5))
            spr_max = float(np.percentile(spr, 95))

            num_cols = 3 + len(per_model_results)
            fig2, axes2 = plt.subplots(
                len(offsets_mm),
                num_cols,
                figsize=(3.2 * num_cols, 3.4 * len(offsets_mm)),
                dpi=140,
                squeeze=False,
            )
            fig2.suptitle(
                f"Case {case_idx:02d} | {npz_path.name} | ray axis={ray_axis}",
                fontsize=10,
            )

            for row_idx, offset_mm in enumerate(offsets_mm):
                shift_vox = int(round(offset_mm / max(float(spacing_mm[shift_axis]), 1e-6)))
                slice_index = int(np.clip(center_index + shift_vox, 0, d_low.shape[shift_axis] - 1))

                spr_slice = extract_oriented_slice(spr, shift_axis, slice_index, ray_axis)
                low_slice = extract_oriented_slice(low_scaled, shift_axis, slice_index, ray_axis)
                high_slice = extract_oriented_slice(high_ref, shift_axis, slice_index, ray_axis)
                pred_slices = [
                    extract_oriented_slice(model_result["pred_scaled"], shift_axis, slice_index, ray_axis)
                    for model_result in per_model_results
                ]

                dose_arrays = [low_slice, high_slice] + pred_slices
                dose_stack = np.concatenate([array.ravel() for array in dose_arrays])
                dose_max = float(np.percentile(dose_stack, 99.5)) if np.any(np.isfinite(dose_stack)) else 1.0
                dose_max = max(dose_max, 1e-6)

                panels = [spr_slice, low_slice, high_slice] + pred_slices
                titles = ["Material Density (SPR)", "Low Noise", "High Noise"] + [
                    f"Pred: {model_result['model_name']}"
                    for model_result in per_model_results
                ]

                for col_idx, (panel, title) in enumerate(zip(panels, titles)):
                    ax = axes2[row_idx, col_idx]
                    if col_idx == 0:
                        ax.imshow(panel, cmap="gray", vmin=spr_min, vmax=spr_max, origin="lower", aspect="auto")
                    else:
                        ax.imshow(panel, cmap="viridis", vmin=0.0, vmax=dose_max, origin="lower", aspect="auto")

                    if row_idx == 0:
                        ax.set_title(title, fontsize=9)
                    if col_idx == 0:
                        label = "Beamlet Center" if abs(offset_mm) < 1e-6 else f"{offset_mm:g}mm Off-Center"
                        ax.set_ylabel(label, fontsize=9)

                    ax.set_xticks([])
                    ax.set_yticks([])

            fig2.tight_layout()
            beamlet_plot = plots_dir / f"case_{case_idx:02d}_{npz_path.stem}_beamlet_style_multi_model.png"
            fig2.savefig(beamlet_plot, bbox_inches="tight")
            plt.close(fig2)

        print(f"[{case_idx:02d}/{sample_count}] {npz_path.name} -> {plot_path.name}")

    summary_csv = args.out_dir / "per_case_metrics.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(case_rows)

    by_checkpoint: dict[str, dict[str, list[float]]] = {}
    for row in case_rows:
        key = row["checkpoint_name"]
        bucket = by_checkpoint.setdefault(key, {"gamma_pass_rate": [], "gamma_mean": [], "depth_l1": []})
        bucket["gamma_pass_rate"].append(float(row["gamma_pass_rate"]))
        bucket["gamma_mean"].append(float(row["gamma_mean"]))
        bucket["depth_l1"].append(float(row["depth_l1"]))

    aggregate = {}
    for key, vals in by_checkpoint.items():
        aggregate[key] = {
            "n_cases": len(vals["gamma_pass_rate"]),
            "gamma_pass_rate_mean": float(np.mean(vals["gamma_pass_rate"])),
            "gamma_pass_rate_std": float(np.std(vals["gamma_pass_rate"])),
            "gamma_mean_mean": float(np.mean(vals["gamma_mean"])),
            "depth_l1_mean": float(np.mean(vals["depth_l1"])),
        }

    summary_json = args.out_dir / "aggregate_metrics.json"
    summary_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Per-case metrics: {summary_csv}")
    print(f"Aggregate metrics: {summary_json}")


if __name__ == "__main__":
    main()
