from typing import Dict

import torch
import torch.nn.functional as F

from src.config import defaults


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Error base voxel a voxel."""
    return F.mse_loss(pred, target)


def _make_distal_weight_mask(
    target: torch.Tensor,
    beam_axis: torch.Tensor,
    distal_fraction: float,
) -> torch.Tensor:
    """Construye máscara de pesos para enfatizar la región distal.

    La región distal se define como la última fracción del volumen sobre `beam_axis`.
    """

    batch_size = target.shape[0]
    weight = torch.ones_like(target)

    for i in range(batch_size):
        axis = int(beam_axis[i].item())
        spatial_len = target.shape[2 + axis]
        distal_len = max(1, int(spatial_len * distal_fraction))

        if axis == 0:
            weight[i, :, -distal_len:, :, :] = 2.0
        elif axis == 1:
            weight[i, :, :, -distal_len:, :] = 2.0
        else:
            weight[i, :, :, :, -distal_len:] = 2.0

    return weight


def bragg_weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beam_axis: torch.Tensor,
    distal_fraction: float = defaults.BRAGG_DISTAL_FRACTION,
) -> torch.Tensor:
    """MSE ponderada para proteger la zona del pico/fall-off distal."""
    weight = _make_distal_weight_mask(target, beam_axis, distal_fraction)
    err2 = (pred - target) ** 2
    return (weight * err2).mean()


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penaliza discrepancias en gradientes espaciales (preserva bordes/fall-off)."""
    grad_pred_z = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    grad_true_z = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]

    grad_pred_y = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
    grad_true_y = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]

    grad_pred_x = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
    grad_true_x = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]

    loss_z = F.l1_loss(grad_pred_z, grad_true_z)
    loss_y = F.l1_loss(grad_pred_y, grad_true_y)
    loss_x = F.l1_loss(grad_pred_x, grad_true_x)
    return (loss_z + loss_y + loss_x) / 3.0


def total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beam_axis: torch.Tensor,
    lambda_bragg: float = defaults.LAMBDA_BRAGG,
    lambda_gradient: float = defaults.LAMBDA_GRADIENT,
) -> Dict[str, torch.Tensor]:
    """Combina términos físicos: MSE + Bragg + Gradiente.

    Retorna cada componente para facilitar inspección durante entrenamiento.
    """
    l_mse = mse_loss(pred, target)
    l_bragg = bragg_weighted_loss(pred, target, beam_axis)
    l_grad = gradient_loss(pred, target)
    l_total = l_mse + lambda_bragg * l_bragg + lambda_gradient * l_grad
    return {
        "total": l_total,
        "mse": l_mse,
        "bragg": l_bragg,
        "grad": l_grad,
    }
