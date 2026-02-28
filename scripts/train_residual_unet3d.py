from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import zipfile
from datetime import datetime
from pathlib import Path

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
            raise ValueError(f"No hay NPZ válidos tras filtrar archivos corruptos en: {manifest_csv}")

    def _filter_invalid_paths(self, paths: list[Path]) -> list[Path]:
        valid: list[Path] = []
        missing = 0
        corrupt = 0

        for path in paths:
            resolved = self._resolve_npz_path(path, raise_if_missing=False)
            if resolved is None:
                missing += 1
                continue
            if not zipfile.is_zipfile(resolved):
                corrupt += 1
                continue
            valid.append(resolved)

        if missing > 0 or corrupt > 0:
            print(f"[Dataset] Filtrado inicial: {len(valid)} válidos, {missing} missing, {corrupt} corruptos")

        return valid

    def _resolve_npz_path(self, path: Path, raise_if_missing: bool = True) -> Path | None:
        if path.exists():
            return path

        candidates: list[Path] = []

        # Common case after merging chunk outputs:
        # manifest still references .../chunk_xx/file.npz but file moved to root dir.
        candidates.append(self.manifest_dir / path.name)

        # If path is relative, also try relative to manifest dir.
        if not path.is_absolute():
            candidates.append((self.manifest_dir / path).resolve())

        # Try removing any chunk_* directory from the original path.
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
            raise FileNotFoundError(f"NPZ not found: {path} (also tried {len(candidates)} fallback paths)")
        return None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        n = len(self.paths)
        last_error: Exception | None = None

        for attempt in range(3):
            real_idx = (idx + attempt) % n
            npz_path = self.paths[real_idx]
            try:
                case = load_case_npz(npz_path)
                break
            except (FileNotFoundError, zipfile.BadZipFile, OSError, ValueError) as exc:
                last_error = exc
                if npz_path not in self.bad_paths_reported:
                    print(f"[Dataset] Saltando NPZ inválido: {npz_path} ({type(exc).__name__})")
                    self.bad_paths_reported.add(npz_path)
                continue
        else:
            raise RuntimeError(f"No se pudo cargar muestra tras 3 intentos (idx={idx}): {last_error}")

        d_low = case.d_low
        spr = case.spr
        d_high = case.d_high
        beam_mask = case.beam_mask if case.beam_mask is not None else np.ones_like(d_low, dtype=np.float32)

        if self.use_bev_crop:
            d_low, spr, d_high = maybe_crop_bev(d_low, spr, d_high, crop_size=self.crop_size, enabled=True)
            beam_mask = maybe_crop_bev(beam_mask, beam_mask, beam_mask, crop_size=self.crop_size, enabled=True)[0]

        e0_map = np.full_like(d_low, fill_value=float(case.e0_mev), dtype=np.float32)
        x = np.stack([d_low, spr, e0_map, beam_mask.astype(np.float32)], axis=0).astype(np.float32)

        return {
            "x": torch.from_numpy(x),
            "d_low": torch.from_numpy(d_low[None, ...].astype(np.float32)),
            "target": torch.from_numpy(d_high[None, ...].astype(np.float32)),
        }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    desc: str = "",
    exp_loss_alpha: float = 0.0,
    exp_loss_max_weight: float = 10.0,
    use_beam_mask: bool = True,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    
    # Use BF16 mixed precision (safe for ROCm)
    autocast_enabled = device.type == "cuda"

    loss_sum = 0.0
    baseline_l1_sum = 0.0
    n_batches = 0
    nan_batches = 0
    oom_batches = 0

    with tqdm(loader, desc=desc, disable=False) as pbar:
        for batch_idx, batch in enumerate(pbar, start=1):
            x = batch["x"].to(device, non_blocking=True)
            d_low = batch["d_low"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            beam_mask = x[:, 3:4, ...] if use_beam_mask else torch.ones_like(target)
            beam_mask = beam_mask.to(target.dtype)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            try:
                # Use BF16 for forward/backward (faster, safe on ROCm)
                with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled, dtype=torch.bfloat16):
                    delta = model(x)
                    pred = d_low + delta
                    diff = torch.abs(pred - target)
                    weights = beam_mask.to(diff.dtype)

                    if exp_loss_alpha > 0.0:
                        norm_target = target / torch.clamp(target.abs().max(), min=1e-3)
                        exp_w = torch.exp(exp_loss_alpha * norm_target)
                        if exp_loss_max_weight > 0:
                            exp_w = torch.clamp(exp_w, max=exp_loss_max_weight)
                        weights = weights * exp_w

                    weighted_diff = (diff * weights).float()
                    denom = torch.clamp(weights.sum().float(), min=1.0)
                    loss = weighted_diff.sum() / denom

                    baseline_diff = torch.abs(d_low - target) * beam_mask
                    baseline_denom = torch.clamp(beam_mask.sum().float(), min=1.0)
                    baseline_l1_sum += float((baseline_diff.sum() / baseline_denom).item())
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" in msg or "hip out of memory" in msg:
                    oom_batches += 1
                    print(f"\n[WARNING] OOM en batch {batch_idx}; se salta batch y se libera cache ({oom_batches} OOMs)")
                    if is_train and optimizer is not None:
                        optimizer.zero_grad(set_to_none=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    if oom_batches > 20:
                        raise RuntimeError("Demasiados OOM en una época (>20). Reduce --batch-size o --crop-size.") from exc
                    continue
                raise
            
            # Detect NaN
            if torch.isnan(loss):
                nan_batches += 1
                print(f"\n[WARNING] NaN detected in batch {n_batches + 1}")
                print(f"  delta range: [{delta.min():.4f}, {delta.max():.4f}]")
                print(f"  pred range:  [{pred.min():.4f}, {pred.max():.4f}]")
                print(f"  target range: [{target.min():.4f}, {target.max():.4f}]")
                if nan_batches > 5:
                    raise RuntimeError(f"Too many NaN batches ({nan_batches}). Stopping.")
                continue  # Skip this batch

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()

            loss_sum += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=f"{loss_sum / max(1, n_batches):.6f}")

    if nan_batches > 0:
        print(f"\n[!] Warning: {nan_batches} batches had NaN loss")
    if oom_batches > 0:
        print(f"\n[!] Warning: {oom_batches} batches skipped by OOM")
    
    return {
        "loss": loss_sum / max(1, n_batches),
        "baseline_l1": baseline_l1_sum / max(1, n_batches),
        "nan_batches": float(nan_batches),
        "oom_batches": float(oom_batches),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    exp_loss_alpha: float = 0.0,
    exp_loss_max_weight: float = 10.0,
    use_beam_mask: bool = True,
    desc: str = "Eval",
) -> dict[str, float]:
    with torch.no_grad():
        return run_epoch(
            model,
            loader,
            None,
            device,
            None,
            desc=desc,
            exp_loss_alpha=exp_loss_alpha,
            exp_loss_max_weight=exp_loss_max_weight,
            use_beam_mask=use_beam_mask,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento Residual 3D U-Net (D_pred = D_low + Net)")
    parser.add_argument("--manifest-train", type=Path, required=True)
    parser.add_argument("--manifest-val", type=Path, required=True)
    parser.add_argument("--manifest-test", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 = main process only, safer)")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate (BF16 safe)")
    parser.add_argument("--min-lr", type=float, default=5e-6, help="Min LR for cosine scheduler")
    parser.add_argument("--lr-scheduler", type=str, choices=["none", "cosine", "exp"], default="cosine")
    parser.add_argument("--lr-gamma", type=float, default=0.97, help="Gamma for exp scheduler")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-bev-crop", action="store_true")
    parser.add_argument("--crop-size", type=str, default="96,96,96")
    parser.add_argument("--exp-loss-alpha", type=float, default=0.0, help="Exponential voxel weighting (0 disables)")
    parser.add_argument("--exp-loss-max-weight", type=float, default=25.0, help="Clip for exponential loss weights (<=0 disables)")
    parser.add_argument("--no-loss-beam-mask", action="store_true", help="No beam-mask weighting (use full volume)")
    parser.add_argument("--use-se-blocks", action="store_true", help="Enable SE attention blocks")
    parser.add_argument("--se-reduction", type=int, default=8, help="Channel reduction for SE blocks")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/resunet3d"))
    parser.add_argument("--save-every", type=int, default=1, help="Guardar checkpoint cada N épocas")
    parser.add_argument("--resume", type=Path, default=None, help="Ruta a checkpoint .pt para reanudar entrenamiento")
    args = parser.parse_args()

    ensure_safe_runtime_dirs()
    print(f"Runtime TMPDIR: {os.environ.get('TMPDIR')}")
    print(f"MIOPEN_USER_DB_PATH: {os.environ.get('MIOPEN_USER_DB_PATH')}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    crop_tokens = [int(x.strip()) for x in args.crop_size.split(",") if x.strip()]
    if len(crop_tokens) != 3:
        raise ValueError("--crop-size debe tener formato D,H,W")
    crop_size = (crop_tokens[0], crop_tokens[1], crop_tokens[2])

    train_ds = ManifestNPZDataset(args.manifest_train, use_bev_crop=not args.no_bev_crop, crop_size=crop_size)
    val_ds = ManifestNPZDataset(args.manifest_val, use_bev_crop=not args.no_bev_crop, crop_size=crop_size)
    test_ds = None
    if args.manifest_test is not None and args.manifest_test.exists():
        test_ds = ManifestNPZDataset(args.manifest_test, use_bev_crop=not args.no_bev_crop, crop_size=crop_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualUNet3D(
        in_channels=4,
        base_channels=args.base_channels,
        residual=True,
        use_se=args.use_se_blocks,
        se_reduction=args.se_reduction,
    ).to(device)
    
    # Use Adam with conservative settings
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay
    )
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
    elif args.lr_scheduler == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    # Re-enable GradScaler for BF16 mixed precision
    scaler = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else None

    start_epoch = 1
    best_val = float("inf")

    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"Checkpoint para reanudar no existe: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        resumed_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = resumed_epoch + 1
        best_val = float(checkpoint.get("best_val_l1", float("inf")))
        run_dir = args.resume.parent
        print(f"[Resume] Checkpoint: {args.resume}")
        print(f"[Resume] Última época guardada: {resumed_epoch}; reanudando en época {start_epoch}")
        print(f"[Resume] best_val_l1 previo: {best_val:.6f}")
    else:
        run_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    print(f"Run dir: {run_dir}")

    if start_epoch > args.epochs:
        print(f"[Resume] Nada que entrenar: start_epoch={start_epoch} > epochs={args.epochs}")
        return

    metrics_mode = "a" if args.resume is not None and metrics_path.exists() else "w"
    with metrics_path.open(metrics_mode, encoding="utf-8") as mf:
        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                scaler,
                desc=f"Epoch {epoch:03d}/Train",
                exp_loss_alpha=args.exp_loss_alpha,
                exp_loss_max_weight=args.exp_loss_max_weight,
                use_beam_mask=not args.no_loss_beam_mask,
            )
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                exp_loss_alpha=args.exp_loss_alpha,
                exp_loss_max_weight=args.exp_loss_max_weight,
                use_beam_mask=not args.no_loss_beam_mask,
                desc=f"Epoch {epoch:03d}/Val",
            )

            current_lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                current_lr = scheduler.get_last_lr()[0]

            row = {
                "epoch": epoch,
                "train_l1": train_metrics["loss"],
                "train_baseline": train_metrics["baseline_l1"],
                "val_l1": val_metrics["loss"],
                "val_baseline": val_metrics["baseline_l1"],
                "lr": current_lr,
            }
            mf.write(json.dumps(row) + "\n")
            mf.flush()

            print(
                f"Epoch {epoch:03d} | lr={current_lr:.6f} | "
                f"train_l1={train_metrics['loss']:.6f} | train_baseline={train_metrics['baseline_l1']:.6f} | "
                f"val_l1={val_metrics['loss']:.6f} | val_baseline={val_metrics['baseline_l1']:.6f}"
            )

            if scheduler is not None:
                scheduler.step()

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "epoch": epoch,
                        "best_val_l1": best_val,
                        "args": vars(args),
                    },
                    run_dir / "best.pt",
                )

            if args.save_every > 0 and (epoch % args.save_every == 0):
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "epoch": epoch,
                        "best_val_l1": best_val,
                        "args": vars(args),
                    },
                    run_dir / f"epoch_{epoch:03d}.pt",
                )

    test_l1 = None
    test_baseline = None
    if test_loader is not None:
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            exp_loss_alpha=args.exp_loss_alpha,
            exp_loss_max_weight=args.exp_loss_max_weight,
            use_beam_mask=not args.no_loss_beam_mask,
            desc="Test",
        )
        test_l1 = test_metrics["loss"]
        test_baseline = test_metrics["baseline_l1"]

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": max(args.epochs, start_epoch - 1),
            "best_val_l1": best_val,
            "test_l1": test_l1,
            "test_baseline": test_baseline,
            "args": vars(args),
        },
        run_dir / "last.pt",
    )

    print(f"Run dir: {run_dir}")
    print(f"Best val L1: {best_val:.6f}")
    if test_l1 is not None:
        print(f"Test L1: {test_l1:.6f} (baseline={test_baseline:.6f})")


if __name__ == "__main__":
    main()
