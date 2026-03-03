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
    candidates = [manifest_dir / path.name]
    if not path.is_absolute():
        candidates.append((manifest_dir / path).resolve())
    for cand in candidates:
        if cand.exists():
            return cand
    return None


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


def infer_predictions(
    case,
    checkpoints: list[Path],
    device: torch.device,
    low_scale_factor: float,
) -> list[tuple[str, np.ndarray]]:
    d_low = case.d_low.astype(np.float32)
    spr = case.spr.astype(np.float32)
    beam_mask = case.beam_mask.astype(np.float32) if case.beam_mask is not None else np.ones_like(d_low, dtype=np.float32)
    e0_map = np.full_like(d_low, fill_value=float(case.e0_mev), dtype=np.float32)

    x_np = np.stack([d_low, spr, e0_map, beam_mask], axis=0)[None, ...].astype(np.float32)
    x_t = torch.from_numpy(x_np).to(device)

    preds: list[tuple[str, np.ndarray]] = []
    for ckpt_path in checkpoints:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model, model_variant = build_model_from_checkpoint(checkpoint, device)

        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"), dtype=torch.bfloat16):
            model_out = model(x_t)
            pred = x_t[:, 0:1, ...] + model_out if model_variant == "resunet_delta" else model_out

        pred_np = pred[0, 0].detach().float().cpu().numpy().astype(np.float64)
        pred_np = pred_np * float(low_scale_factor)
        preds.append((ckpt_path.stem, pred_np))

    return preds


def build_auto_regions(d_high: np.ndarray, spr: np.ndarray, beam_mask: np.ndarray) -> dict[str, np.ndarray]:
    body = spr > np.percentile(spr, 2)
    beam = beam_mask > 0.5

    high_vals = d_high[body]
    if high_vals.size == 0:
        high_vals = d_high.ravel()

    p95 = np.percentile(high_vals, 95)
    p99 = np.percentile(high_vals, 99)

    regions = {
        "Body": body,
        "Beam Path": beam,
        "High Dose Core (p99+)": (d_high >= p99) & body,
        "Mid Dose Shell (p95-99)": (d_high >= p95) & (d_high < p99) & body,
    }

    for name, mask in list(regions.items()):
        if np.count_nonzero(mask) == 0:
            regions.pop(name)

    return regions


def dvh_curve(dose: np.ndarray, mask: np.ndarray, bins: np.ndarray) -> np.ndarray:
    vals = dose[mask]
    if vals.size == 0:
        return np.zeros_like(bins)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(bins)

    # volume receiving >= dose threshold
    vals_sorted = np.sort(vals)
    n = vals_sorted.size
    idx = np.searchsorted(vals_sorted, bins, side="left")
    frac = (n - idx) / max(1, n)
    return frac * 100.0


def pick_case_from_manifest(manifest: Path, seed: int) -> Path:
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"Manifest vacío: {manifest}")

    paths: list[Path] = []
    for row in rows:
        p = Path(row["npz_path"])
        r = resolve_npz_path(p, manifest.parent.resolve())
        if r is not None:
            paths.append(r)

    if not paths:
        raise RuntimeError("No se pudo resolver ningún NPZ desde el manifest")

    random.seed(seed)
    return random.choice(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="DVH-style evaluation plot (GT vs Noisy vs Predictions)")
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--input-npz", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low-scale-factor", type=float, default=1.0)
    parser.add_argument("--max-dose", type=float, default=0.0, help="Max dose on x-axis; <=0 auto")
    parser.add_argument("--num-bins", type=int, default=300)
    parser.add_argument("--out-png", type=Path, default=Path("outputs/dvh_like/dvh_like_plot.png"))
    parser.add_argument("--out-json", type=Path, default=Path("outputs/dvh_like/dvh_like_summary.json"))
    args = parser.parse_args()

    if args.input_npz is None and args.manifest is None:
        raise ValueError("Debes pasar --input-npz o --manifest")

    for ckpt in args.checkpoint:
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {ckpt}")

    if args.input_npz is not None:
        npz_path = args.input_npz
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ no encontrado: {npz_path}")
    else:
        npz_path = pick_case_from_manifest(args.manifest, args.seed)

    case = load_case_npz(npz_path)
    d_low = case.d_low.astype(np.float64) * float(args.low_scale_factor)
    d_high = case.d_high.astype(np.float64)
    spr = case.spr.astype(np.float64)
    beam_mask = case.beam_mask.astype(np.float64) if case.beam_mask is not None else np.ones_like(d_high, dtype=np.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = infer_predictions(case, args.checkpoint, device, args.low_scale_factor)

    regions = build_auto_regions(d_high, spr, beam_mask)
    if not regions:
        raise RuntimeError("No hay regiones válidas para DVH")

    x_max = float(args.max_dose)
    if x_max <= 0:
        peak = max(
            float(np.max(d_high)),
            float(np.max(d_low)),
            max(float(np.max(pv)) for _, pv in preds),
        )
        x_max = peak * 1.05

    bins = np.linspace(0.0, x_max, int(args.num_bins), dtype=np.float64)

    color_cycle = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]

    fig, ax = plt.subplots(figsize=(12, 8), dpi=140)
    summary: dict[str, dict] = {
        "npz": str(npz_path),
        "device": str(device),
        "regions": {},
    }

    pred_styles = ["--", "-.", (0, (5, 2)), (0, (3, 2, 1, 2))]

    for region_idx, (region_name, region_mask) in enumerate(regions.items()):
        color = color_cycle[region_idx % len(color_cycle)]

        dvh_gt = dvh_curve(d_high, region_mask, bins)
        dvh_noisy = dvh_curve(d_low, region_mask, bins)
        ax.plot(bins, dvh_gt, color=color, lw=2.0, linestyle="-", label=f"{region_name} | Ground Truth")
        ax.plot(bins, dvh_noisy, color=color, lw=1.4, linestyle=":", alpha=0.95, label=f"{region_name} | Noisy")

        summary["regions"][region_name] = {
            "n_voxels": int(np.count_nonzero(region_mask)),
            "gt_mean": float(np.mean(d_high[region_mask])) if np.count_nonzero(region_mask) else 0.0,
            "noisy_mean": float(np.mean(d_low[region_mask])) if np.count_nonzero(region_mask) else 0.0,
        }

        for pred_idx, (pred_name, pred_dose) in enumerate(preds):
            style = pred_styles[pred_idx % len(pred_styles)]
            dvh_pred = dvh_curve(pred_dose, region_mask, bins)
            ax.plot(
                bins,
                dvh_pred,
                color=color,
                lw=1.6,
                linestyle=style,
                label=f"{region_name} | {pred_name}",
            )
            summary["regions"][region_name][f"pred_mean_{pred_name}"] = (
                float(np.mean(pred_dose[region_mask])) if np.count_nonzero(region_mask) else 0.0
            )

    ax.set_title("DVH-style Evaluation: Ground Truth vs Noisy vs Predictions")
    ax.set_xlabel("Dose")
    ax.set_ylabel("Volume (%)")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)

    # compact legend outside
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    fig.tight_layout()

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, bbox_inches="tight")
    plt.close(fig)

    summary["out_png"] = str(args.out_png)
    summary["checkpoints"] = [str(c) for c in args.checkpoint]
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"NPZ: {npz_path}")
    print(f"Saved DVH plot: {args.out_png}")
    print(f"Saved summary: {args.out_json}")


if __name__ == "__main__":
    main()
