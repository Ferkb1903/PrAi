from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import defaults
from src.data.dataset import ProtonDoseDataset
from src.metrics.distal_range import distal_error_mm
from src.metrics.gamma import gamma_report
from src.models.model_factory import build_model


def resolve_device() -> torch.device:
    """Selecciona dispositivo de inferencia."""
    if defaults.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_latest_checkpoint(model: torch.nn.Module, checkpoint_dir: Path) -> None:
    """Carga el checkpoint más reciente según nombre de época."""
    ckpts = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No se encontraron checkpoints en {checkpoint_dir}")
    latest = ckpts[-1]
    payload = torch.load(latest, map_location="cpu")
    model.load_state_dict(payload["model_state"])


def evaluate() -> Dict[str, float]:
    """Evalúa el modelo en test y devuelve métricas agregadas."""
    device = resolve_device()
    test_ds = ProtonDoseDataset(defaults.TEST_DIR, predict_residual=defaults.PREDICT_RESIDUAL)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=defaults.NUM_WORKERS)

    model = build_model().to(device)
    load_latest_checkpoint(model, defaults.CHECKPOINT_DIR)
    model.eval()

    gamma3_values = []
    gamma2_values = []
    distal_errors = []
    mae_values = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            d_low = batch["d_low"].to(device)
            d_high = batch["d_high"].to(device)
            spacing_mm = batch["spacing_mm"][0].cpu().numpy()
            beam_axis = int(batch["beam_axis"][0].item())

            pred = model(x)
            # Si entrenamos residual, reconstruimos dosis absoluta sumando d_low.
            d_pred = d_low + pred if defaults.PREDICT_RESIDUAL else pred

            ref_np = d_high[0, 0].cpu().numpy().astype(np.float32)
            pred_np = d_pred[0, 0].cpu().numpy().astype(np.float32)

            g = gamma_report(ref_np, pred_np)
            gamma3_values.append(g["gamma_3pct"])
            gamma2_values.append(g["gamma_2pct"])

            distal_errors.append(distal_error_mm(ref_np, pred_np, spacing_mm, beam_axis))
            mae_values.append(float(np.mean(np.abs(ref_np - pred_np))))

    report = {
        "gamma_3pct_mean": float(np.mean(gamma3_values)),
        "gamma_2pct_mean": float(np.mean(gamma2_values)),
        "distal_error_mm_mean": float(np.mean(distal_errors)),
        "mae_mean": float(np.mean(mae_values)),
    }
    return report


if __name__ == "__main__":
    result = evaluate()
    for key, value in result.items():
        print(f"{key}: {value:.6f}")
