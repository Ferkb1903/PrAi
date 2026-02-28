#!/usr/bin/env python
"""
Generate depth profile analysis (Bragg peaks, dose difference) with predictions.
Compares ground truth vs model predictions on test cases.
Usage:
  python scripts/analyze_predictions_depth.py --checkpoint best.pt --manifest data/training_npz/spot_campaign_v2_parallel/manifest_test.csv --n-cases 5
"""

import argparse
import csv
import os
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io_npz import load_case_npz
from src.data.preprocess import maybe_crop_bev
from src.model.resunet3d import ResidualUNet3D


def ensure_safe_runtime_dirs() -> None:
    user = os.environ.get("USER", "user")
    base_tmp = Path(f"/tmp/miopen_cache_{user}")
    base_tmp.mkdir(parents=True, exist_ok=True)
    os.chmod(base_tmp, 0o700)

    tmpdir = os.environ.get("TMPDIR", "").strip()
    if not tmpdir or not Path(tmpdir).is_dir():
        os.environ["TMPDIR"] = str(base_tmp)

    miopen_db = base_tmp / "miopen_db"
    miopen_db.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MIOPEN_USER_DB_PATH", str(miopen_db))
    os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", str(base_tmp))


class ManifestNPZDataset(Dataset):
    def __init__(self, manifest_csv: Path, use_bev_crop: bool = True, crop_size: tuple[int, int, int] = (96, 96, 96)) -> None:
        rows = list(csv.DictReader(manifest_csv.open(encoding="utf-8")))
        self.paths = [Path(r["npz_path"]) for r in rows]
        self.manifest_dir = manifest_csv.parent.resolve()
        self.use_bev_crop = use_bev_crop
        self.crop_size = crop_size
        self.bad_paths_reported: set[Path] = set()
        
        if not self.paths:
            raise ValueError(f"Manifest vacío: {manifest_csv}")
        
        self.paths = self._filter_invalid_paths(self.paths)
        if not self.paths:
            raise ValueError(f"No hay NPZ válidos tras filtrar: {manifest_csv}")

    def _filter_invalid_paths(self, paths: list[Path]) -> list[Path]:
        valid: list[Path] = []
        for path in paths:
            resolved = self._resolve_npz_path(path, raise_if_missing=False)
            if resolved is None or not zipfile.is_zipfile(resolved):
                continue
            valid.append(resolved)
        return valid

    def _resolve_npz_path(self, path: Path, raise_if_missing: bool = True) -> Path | None:
        if path.exists():
            return path

        candidates: list[Path] = [
            self.manifest_dir / path.name,
        ]
        if not path.is_absolute():
            candidates.append((self.manifest_dir / path).resolve())

        parts = list(path.parts)
        filtered_parts = [p for p in parts if not p.startswith("chunk_")]
        if filtered_parts != parts:
            try:
                candidates.append(Path(*filtered_parts))
            except TypeError:
                pass

        for candidate in candidates:
            if candidate.exists():
                return candidate

        if raise_if_missing:
            raise FileNotFoundError(f"NPZ not found: {path}")
        return None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        n = len(self.paths)
        last_error: Exception | None = None

        for attempt in range(3):
            real_idx = (idx + attempt) % n
            npz_path = self.paths[real_idx]
            try:
                case = load_case_npz(npz_path)
                break
            except Exception as exc:
                last_error = exc
                if npz_path not in self.bad_paths_reported:
                    print(f"[Dataset] Saltando NPZ: {npz_path}")
                    self.bad_paths_reported.add(npz_path)
                continue
        else:
            raise RuntimeError(f"No se pudo cargar muestra tras 3 intentos: {last_error}")

        d_low = case.d_low
        spr = case.spr
        d_high = case.d_high
        beam_mask = case.beam_mask if case.beam_mask is not None else np.ones_like(d_low, dtype=np.float32)

        # Original shape before crops
        orig_shape = d_low.shape

        if self.use_bev_crop:
            d_low, spr, d_high = maybe_crop_bev(d_low, spr, d_high, crop_size=self.crop_size, enabled=True)
            beam_mask = maybe_crop_bev(beam_mask, beam_mask, beam_mask, crop_size=self.crop_size, enabled=True)[0]

        e0_map = np.full_like(d_low, fill_value=float(case.e0_mev), dtype=np.float32)
        x = np.stack([d_low, spr, e0_map, beam_mask.astype(np.float32)], axis=0).astype(np.float32)

        return {
            "x": torch.from_numpy(x),
            "d_low": torch.from_numpy(d_low[None, ...].astype(np.float32)),
            "target": torch.from_numpy(d_high[None, ...].astype(np.float32)),
            "beam_mask": torch.from_numpy(beam_mask[None, ...].astype(np.float32)),
            "spr": torch.from_numpy(spr[None, ...].astype(np.float32)),
            "e0_mev": float(case.e0_mev),
            "npz_path": str(npz_path),
            "orig_shape": orig_shape,
        }


def find_peak_location(dose_2d: np.ndarray) -> tuple[int, int]:
    """Find XY location of maximum dose (Bragg peak)."""
    if dose_2d.size == 0:
        return 0, 0
    flat_idx = np.argmax(dose_2d)
    y, x = np.unravel_index(flat_idx, dose_2d.shape)
    return x, y


def analyze_depth_profiles(
    case_idx: int,
    d_low: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    spr: np.ndarray,
    beam_mask: np.ndarray,
    e0_mev: float,
    npz_name: str,
    out_dir: Path,
) -> dict:
    """Generate depth profile analysis (Bragg peaks, dose profiles)."""
    
    # Find peak location in ground truth (2D max intensity projection)
    target_xy_max = np.max(target, axis=0)  # Project to XY
    peak_x, peak_y = find_peak_location(target_xy_max)
    
    # Extract depth profiles at peak location
    d_low_profile = d_low[:, peak_y, peak_x]
    target_profile = target[:, peak_y, peak_x]
    pred_profile = pred[:, peak_y, peak_x]
    spr_profile = spr[:, peak_y, peak_x]
    
    # Smooth profiles for visualization
    from scipy.ndimage import gaussian_filter1d
    kernel_size = max(1, d_low.shape[0] // 50)
    target_smooth = gaussian_filter1d(target_profile, sigma=kernel_size / 4)
    pred_smooth = gaussian_filter1d(pred_profile, sigma=kernel_size / 4)
    
    # Depth index
    depth_idx = np.arange(len(d_low_profile))
    
    # Compute error
    err_profile = np.abs(pred_profile - target_profile)
    
    # Metrics
    metrics = {
        "case": case_idx,
        "npz_name": npz_name,
        "peak_x": int(peak_x),
        "peak_y": int(peak_y),
        "target_peak_dose": float(np.max(target_profile)),
        "pred_peak_dose": float(np.max(pred_profile)),
        "peak_error": float(np.abs(np.max(pred_profile) - np.max(target_profile))),
        "mean_error": float(np.mean(err_profile)),
        "max_error": float(np.max(err_profile)),
        "bragg_peak_target": float(np.argmax(target_profile)) if len(target_profile) > 0 else np.nan,
        "bragg_peak_pred": float(np.argmax(pred_profile)) if len(pred_profile) > 0 else np.nan,
    }
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Row 1: Projections and 2D slices
    ax = axes[0, 0]
    im = ax.imshow(target_xy_max, cmap="viridis")
    ax.plot(peak_x, peak_y, "r+", markersize=15, markeredgewidth=2)
    ax.set_title("Ground Truth Max Intensity (XY)")
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 1]
    im = ax.imshow(np.max(pred, axis=0), cmap="viridis")
    ax.plot(peak_x, peak_y, "r+", markersize=15, markeredgewidth=2)
    ax.set_title("Prediction Max Intensity (XY)")
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 2]
    beam_mask_xy = np.max(beam_mask, axis=0)
    ax.imshow(beam_mask_xy, cmap="gray", alpha=0.7)
    ax.plot(peak_x, peak_y, "r+", markersize=15, markeredgewidth=2)
    ax.set_title(f"Beam Mask (Peak at x={peak_x}, y={peak_y})")
    
    # Row 2: Depth profiles
    ax = axes[1, 0]
    ax.plot(depth_idx, d_low_profile, label="D_low (input)", alpha=0.7)
    ax.plot(depth_idx, target_smooth, label="Ground Truth (smooth)", linewidth=2)
    ax.plot(depth_idx, pred_smooth, label="Prediction (smooth)", linewidth=2, linestyle="--")
    ax.set_xlabel("Depth Index")
    ax.set_ylabel("Dose")
    ax.set_title("Depth Profiles at Bragg Peak")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(depth_idx, target_profile - pred_profile, label="GT - Pred", color="purple", linewidth=2)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.fill_between(depth_idx, target_profile - pred_profile, 0, alpha=0.3, color="purple")
    ax.set_xlabel("Depth Index")
    ax.set_ylabel("Dose Difference")
    ax.set_title("Dose Difference (GT - Pred)")
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    ax.plot(depth_idx, spr_profile, color="brown", linewidth=2)
    ax.set_xlabel("Depth Index")
    ax.set_ylabel("SPR")
    ax.set_title("SPR Along Depth")
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Case {case_idx}: {npz_name} | E0={e0_mev:.0f} MeV | Peak Error={metrics['peak_error']:.4f}")
    save_path = out_dir / f"depth_analysis_{case_idx:03d}_{Path(npz_name).stem}.png"
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {save_path.name}")
    
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze predictions with depth profiles (Bragg peaks)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--n-cases", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--crop-size", type=str, default="96,96,96")
    parser.add_argument("--no-bev-crop", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("predictions_depth"))
    args = parser.parse_args()

    ensure_safe_runtime_dirs()
    print(f"Runtime TMPDIR: {os.environ.get('TMPDIR')}")

    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not args.manifest.exists():
        print(f"ERROR: Manifest not found: {args.manifest}")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    crop_tokens = [int(x.strip()) for x in args.crop_size.split(",") if x.strip()]
    if len(crop_tokens) != 3:
        raise ValueError("--crop-size debe tener formato D,H,W")
    crop_size = (crop_tokens[0], crop_tokens[1], crop_tokens[2])

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if "args" in checkpoint and "base_channels" in checkpoint["args"]:
        base_channels = checkpoint["args"]["base_channels"]
    else:
        base_channels = args.base_channels
    
    model = ResidualUNet3D(in_channels=4, base_channels=base_channels, residual=True).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"[Checkpoint] Loaded: {args.checkpoint} (epoch {checkpoint.get('epoch', '?')})")

    # Load dataset
    dataset = ManifestNPZDataset(args.manifest, use_bev_crop=not args.no_bev_crop, crop_size=crop_size)
    n_cases = min(args.n_cases, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"[Dataset] {len(dataset)} total samples; analyzing {n_cases}")

    # Generate analysis
    case_count = 0
    all_metrics = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Analyzing"):
            if case_count >= n_cases:
                break

            x = batch["x"].to(device, non_blocking=True)
            d_low = batch["d_low"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            beam_mask = batch["beam_mask"].to(device, non_blocking=True)
            spr = batch["spr"].to(device, non_blocking=True)
            e0_mev = batch["e0_mev"][0] if isinstance(batch["e0_mev"], torch.Tensor) else batch["e0_mev"]
            npz_path = batch["npz_path"][0]

            try:
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
                    delta = model(x)
                    pred = d_low + delta

                # Move to CPU & numpy
                d_low_np = d_low[0, 0].cpu().numpy()
                target_np = target[0, 0].cpu().numpy()
                pred_np = pred[0, 0].cpu().numpy()
                beam_mask_np = beam_mask[0, 0].cpu().numpy()
                spr_np = spr[0, 0].cpu().numpy()

                # Analyze
                npz_name = Path(npz_path).name
                metrics = analyze_depth_profiles(
                    case_count, d_low_np, target_np, pred_np, spr_np, beam_mask_np, float(e0_mev), npz_name, args.out_dir
                )
                all_metrics.append(metrics)
                
                case_count += 1
            except Exception as exc:
                print(f"[!] Error on case {case_count}: {exc}")
                import traceback
                traceback.print_exc()
                continue

    # Save metrics
    if all_metrics:
        import json
        metrics_path = args.out_dir / "depth_metrics.jsonl"
        with open(metrics_path, "w") as f:
            for m in all_metrics:
                f.write(json.dumps(m) + "\n")
        print(f"\n[Metrics] Saved to: {metrics_path}")
        
        # Print summary
        peak_errors = [m["peak_error"] for m in all_metrics]
        mean_errors = [m["mean_error"] for m in all_metrics]
        print(f"\n  Peak Error:  {np.mean(peak_errors):.6f} ± {np.std(peak_errors):.6f}")
        print(f"  Mean Error:  {np.mean(mean_errors):.6f} ± {np.std(mean_errors):.6f}")
    
    print(f"\n[Done] Analysis saved to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
