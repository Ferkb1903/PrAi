from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io_npz import load_case_npz
from src.model.resunet3d import ResidualUNet3D


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
    parser = argparse.ArgumentParser(description="Run inference on one NPZ and save outputs to a new NPZ")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument("--low-physical-factor", type=float, default=200.0)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.input_npz.exists():
        raise FileNotFoundError(f"Input NPZ not found: {args.input_npz}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, model_variant = build_model_from_checkpoint(checkpoint, device)

    case = load_case_npz(args.input_npz)
    d_low = case.d_low.astype(np.float32)
    spr = case.spr.astype(np.float32)
    d_high = case.d_high.astype(np.float32)
    beam_mask = case.beam_mask.astype(np.float32) if case.beam_mask is not None else np.ones_like(d_low, dtype=np.float32)
    e0_map = np.full_like(d_low, fill_value=float(case.e0_mev), dtype=np.float32)

    x = np.stack([d_low, spr, e0_map, beam_mask], axis=0)[None, ...].astype(np.float32)
    x_t = torch.from_numpy(x).to(device)

    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"), dtype=torch.bfloat16):
        out = model(x_t)
        pred_t = x_t[:, 0:1, ...] + out if model_variant == "resunet_delta" else out

    pred = pred_t[0, 0].detach().float().cpu().numpy().astype(np.float32)

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        d_low=d_low,
        d_high=d_high,
        pred=pred,
        d_low_phys=(d_low * float(args.low_physical_factor)).astype(np.float32),
        pred_phys=(pred * float(args.low_physical_factor)).astype(np.float32),
        spr=spr,
        beam_mask=beam_mask,
        e0_mev=np.asarray(case.e0_mev, dtype=np.float32),
        spacing_mm=np.asarray(case.spacing_mm, dtype=np.float32),
        beam_axis=np.asarray(case.beam_axis if case.beam_axis is not None else 2, dtype=np.int32),
        model_variant=np.asarray(model_variant),
        checkpoint=np.asarray(str(args.checkpoint)),
    )

    print(f"Device: {device}")
    print(f"Model variant: {model_variant}")
    print(f"Saved: {args.out_npz}")
    print(f"d_low max: {float(np.max(d_low)):.6f}")
    print(f"pred  max: {float(np.max(pred)):.6f}")
    print(f"d_high max: {float(np.max(d_high)):.6f}")


if __name__ == "__main__":
    main()
